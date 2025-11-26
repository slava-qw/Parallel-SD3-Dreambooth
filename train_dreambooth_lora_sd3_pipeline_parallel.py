#!/usr/bin/env python
# coding=utf-8
# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import copy
import gc
import hashlib
import logging
import math
import os
import random
import shutil
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.utils.checkpoint
import transformers
from huggingface_hub import create_repo, upload_folder
from peft import LoraConfig, set_peft_model_state_dict
from peft.utils import get_peft_model_state_dict
from PIL import Image
from PIL.ImageOps import exif_transpose
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms.functional import crop
from tqdm.auto import tqdm

import psutil
import pynvml
import subprocess
import time

import diffusers
from diffusers.models.modeling_outputs import Transformer2DModelOutput


from diffusers import (
    AutoencoderKL,
    FlowMatchEulerDiscreteScheduler,
    SD3Transformer2DModel,
    StableDiffusion3Pipeline,
)
from diffusers.optimization import get_scheduler
from diffusers.training_utils import (
    cast_training_params,
    compute_density_for_timestep_sampling,
    compute_loss_weighting_for_sd3,
)
from diffusers.utils import (
    check_min_version,
    convert_unet_state_dict_to_peft,
    is_wandb_available,
)
from diffusers.utils.hub_utils import load_or_create_model_card, populate_model_card
from diffusers.utils.torch_utils import is_compiled_module
from torch.utils.tensorboard import SummaryWriter
from peft import inject_adapter_in_model

# this part is to use for shared cluster
# tried to measure utilization correctly
def get_process_gpu_sm_util(pid):
    try:
        result = subprocess.run(
            ['nvidia-smi', 'pmon', '-s', 'u', '-c', '1'],
            capture_output=True, text=True, timeout=2
        )
        gpu_utils = {}
        for line in result.stdout.split('\n'):
            parts = line.split()
            if len(parts) >= 4 and parts[1] == str(pid):
                gpu_idx = int(parts[0])
                sm_util = parts[3]
                if sm_util != '-':
                    gpu_utils[gpu_idx] = int(sm_util)
        return gpu_utils
    except Exception:
        return {}


if is_wandb_available():
    import wandb

check_min_version("0.30.0.dev0")

logger = logging.getLogger(__name__)


class PipelineParallelSD3Transformer(nn.Module):

    def __init__(
        self,
        base_transformer: SD3Transformer2DModel,
        devices: list,
        chunks: int = 4,
    ):
        super().__init__()

        self.devices = devices
        self.num_stages = len(devices)
        self.chunks = chunks
        self.config = base_transformer.config
        self.out_channels = base_transformer.out_channels
        self.inner_dim = base_transformer.inner_dim
        self.pos_embed = base_transformer.pos_embed.to(devices[0])
        self.time_text_embed = base_transformer.time_text_embed.to(devices[0])
        self.context_embedder = base_transformer.context_embedder.to(devices[0])
        self.block_stages = nn.ModuleList()
        num_blocks = len(base_transformer.transformer_blocks)
        blocks_per_stage = num_blocks // self.num_stages
        remainder = num_blocks % self.num_stages

        start_idx = 0
        for i in range(self.num_stages):
            end_idx = start_idx + blocks_per_stage + (1 if i < remainder else 0)
            stage_blocks = nn.ModuleList()

            for j in range(start_idx, end_idx):
                block = base_transformer.transformer_blocks[j].to(devices[i])
                stage_blocks.append(block)

            self.block_stages.append(stage_blocks)
            start_idx = end_idx

        self.norm_out = base_transformer.norm_out.to(devices[-1])
        self.proj_out = base_transformer.proj_out.to(devices[-1])

        logger.info(f"Pipeline parallel setup: {self.num_stages} stages across {devices}")
        for i, stage in enumerate(self.block_stages):
            logger.info(f"  Stage {i} ({devices[i]}): {len(stage)} blocks")

    def add_adapter(self, lora_config):
        """Apply LoRA adapters to all transformer blocks."""
        for stage_idx, stage_blocks in enumerate(self.block_stages):
            for block in stage_blocks:
                inject_adapter_in_model(lora_config, block, adapter_name="default")

    def enable_gradient_checkpointing(self):
        for stage_blocks in self.block_stages:
            for block in stage_blocks:
                block.gradient_checkpointing = True

    def _forward_stage(self, stage_idx, hidden_states, encoder_hidden_states, temb):
        device = self.devices[stage_idx]

        ###
        dtype = next(self.block_stages[stage_idx][0].parameters()).dtype
        ###

        hidden_states = hidden_states.to(device=device, dtype=dtype)
        encoder_hidden_states = encoder_hidden_states.to(device=device, dtype=dtype)
        temb = temb.to(device=device, dtype=dtype)
        for block in self.block_stages[stage_idx]:
            encoder_hidden_states, hidden_states = block(
                hidden_states=hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                temb=temb,
            )

        return hidden_states, encoder_hidden_states, temb

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor = None,
        pooled_projections: torch.Tensor = None,
        timestep: torch.Tensor = None,
        # joint_attention_kwargs=None,
        return_dict: bool = True,
    ):
        """
        GPipe-style micro-batching
        """
        first_device = self.devices[0]
        last_device = self.devices[-1]

        hidden_states = hidden_states.to(first_device)
        encoder_hidden_states = encoder_hidden_states.to(first_device)
        pooled_projections = pooled_projections.to(first_device)
        timestep = timestep.to(first_device)

        batch_size, channels, height, width = hidden_states.shape

        hidden_states = self.pos_embed(hidden_states)
        temb = self.time_text_embed(timestep, pooled_projections)
        encoder_hidden_states = self.context_embedder(encoder_hidden_states)

        if self.training and self.chunks > 1 and batch_size >= self.chunks:
            outputs = self._forward_with_microbatches(
                hidden_states, encoder_hidden_states, temb, batch_size
            )
            hidden_states = outputs
        else:
            for stage_idx in range(self.num_stages):
                hidden_states, encoder_hidden_states, temb = self._forward_stage(
                    stage_idx, hidden_states, encoder_hidden_states, temb
                )
        hidden_states = hidden_states.to(last_device)
        temb = temb.to(last_device)
        hidden_states = self.norm_out(hidden_states, temb)
        hidden_states = self.proj_out(hidden_states)
        patch_size = self.config.patch_size
        out_channels = self.out_channels
        height_out = height // patch_size
        width_out = width // patch_size

        hidden_states = hidden_states.reshape(
            batch_size, height_out, width_out, patch_size, patch_size, out_channels
        )
        hidden_states = hidden_states.permute(0, 5, 1, 3, 2, 4)
        hidden_states = hidden_states.reshape(
            batch_size, out_channels, height_out * patch_size, width_out * patch_size
        )

        if return_dict:
            return Transformer2DModelOutput(sample=hidden_states)

        return (hidden_states,)

    def _forward_with_microbatches(self, hidden_states, encoder_hidden_states, temb, batch_size):
        # chunk_size = batch_size // self.chunks
        num_chunks = self.chunks

        hidden_chunks = hidden_states.chunk(num_chunks, dim=0)
        encoder_chunks = encoder_hidden_states.chunk(num_chunks, dim=0)
        temb_chunks = temb.chunk(num_chunks, dim=0)

        outputs = []

        for chunk_idx in range(num_chunks):
            h = hidden_chunks[chunk_idx]
            e = encoder_chunks[chunk_idx]
            t = temb_chunks[chunk_idx]

            for stage_idx in range(self.num_stages):
                h, e, t = self._forward_stage(stage_idx, h, e, t)

            outputs.append(h)

        return torch.cat(outputs, dim=0)

    def parameters(self, recurse=True):
        for param in self.pos_embed.parameters(recurse):
            yield param
        for param in self.time_text_embed.parameters(recurse):
            yield param
        for param in self.context_embedder.parameters(recurse):
            yield param
        for stage_blocks in self.block_stages:
            for block in stage_blocks:
                for param in block.parameters(recurse):
                    yield param
        for param in self.norm_out.parameters(recurse):
            yield param
        for param in self.proj_out.parameters(recurse):
            yield param

    def named_parameters(self, prefix='', recurse=True):
        for name, param in self.pos_embed.named_parameters(prefix=f'{prefix}pos_embed', recurse=recurse):
            yield name, param
        for name, param in self.time_text_embed.named_parameters(prefix=f'{prefix}time_text_embed', recurse=recurse):
            yield name, param
        for name, param in self.context_embedder.named_parameters(prefix=f'{prefix}context_embedder', recurse=recurse):
            yield name, param
        for stage_idx, stage_blocks in enumerate(self.block_stages):
            for block_idx, block in enumerate(stage_blocks):
                for name, param in block.named_parameters(prefix=f'{prefix}block_stages.{stage_idx}.{block_idx}', recurse=recurse):
                    yield name, param
        for name, param in self.norm_out.named_parameters(prefix=f'{prefix}norm_out', recurse=recurse):
            yield name, param
        for name, param in self.proj_out.named_parameters(prefix=f'{prefix}proj_out', recurse=recurse):
            yield name, param

    def train(self, mode=True):
        super().train(mode)
        self.pos_embed.train(mode)
        self.time_text_embed.train(mode)
        self.context_embedder.train(mode)
        for stage_blocks in self.block_stages:
            for block in stage_blocks:
                block.train(mode)
        self.norm_out.train(mode)
        self.proj_out.train(mode)
        return self

    def eval(self):
        return self.train(False)


def save_model_card(
    repo_id: str,
    images=None,
    base_model: str = None,
    train_text_encoder=False,
    instance_prompt=None,
    validation_prompt=None,
    repo_folder=None,
):
    widget_dict = []
    if images is not None:
        for i, image in enumerate(images):
            image.save(os.path.join(repo_folder, f"image_{i}.png"))
            widget_dict.append(
                {"text": validation_prompt if validation_prompt else " ", "output": {"url": f"image_{i}.png"}}
            )

    model_description = f"""
# SD3 DreamBooth LoRA (Pipeline Parallel) - {repo_id}

<Gallery />

## Model description

These are {repo_id} DreamBooth weights for {base_model}.

The weights were trained using [DreamBooth](https://dreambooth.github.io/) with Pipeline Parallelism.

LoRA for the text encoder was enabled: {train_text_encoder}.

## Trigger words

You should use {instance_prompt} to trigger the image generation.

## Download model

[Download]({repo_id}/tree/main) them in the Files & versions tab.

## License

Please adhere to the licensing terms as described [here](https://huggingface.co/stabilityai/stable-diffusion-3-medium/blob/main/LICENSE).
"""
    model_card = load_or_create_model_card(
        repo_id_or_path=repo_id,
        from_training=True,
        license="openrail++",
        base_model=base_model,
        prompt=instance_prompt,
        model_description=model_description,
        widget=widget_dict,
    )
    tags = [
        "text-to-image",
        "diffusers-training",
        "diffusers",
        "lora",
        "sd3",
        "sd3-diffusers",
        "template:sd-lora",
        "pipeline-parallel",
    ]

    model_card = populate_model_card(model_card, tags=tags)
    model_card.save(os.path.join(repo_folder, "README.md"))


def log_validation(
    pipeline,
    args,
    pipeline_args,
    epoch,
    writer,
    is_final_validation=False,
):
    logger.info(
        f"Running validation... \n Generating {args.num_validation_images} images with prompt:"
        f" {args.validation_prompt}."
    )
    pipeline.enable_model_cpu_offload()
    pipeline.set_progress_bar_config(disable=True)

    generator = torch.Generator(device="cuda").manual_seed(args.seed) if args.seed else None
    autocast_ctx = nullcontext()

    with autocast_ctx:
        images = [pipeline(**pipeline_args, generator=generator).images[0] for _ in range(args.num_validation_images)]

    phase_name = "test" if is_final_validation else "validation"
    if args.report_to == "tensorboard":
        np_images = np.stack([np.asarray(img) for img in images])
        writer.add_images(phase_name, np_images, epoch, dataformats="NHWC")
    if args.report_to == "wandb":
        wandb.log(
            {
                phase_name: [
                    wandb.Image(image, caption=f"{i}: {args.validation_prompt}") for i, image in enumerate(images)
                ]
            }
        )
    # elif args.report_to == "tensorboard":
    #     writer.add_scalar()

    del pipeline
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return images


def parse_args(input_args=None):
    parser = argparse.ArgumentParser(description="SD3 DreamBooth LoRA training with Pipeline Parallelism.")
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default=None,
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        required=False,
        help="Revision of pretrained model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--variant",
        type=str,
        default=None,
        help="Variant of the model files of the pretrained model identifier from huggingface.co/models, 'e.g.' fp16",
    )
    parser.add_argument(
        "--instance_data_dir",
        type=str,
        default=None,
        help=("A folder containing the training data. "),
    )
    parser.add_argument(
        "--data_df_path",
        type=str,
        default=None,
        help=("Path to the parquet file serialized with compute_embeddings.py."),
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help="The directory where the downloaded models and datasets will be stored.",
    )
    parser.add_argument(
        "--instance_prompt",
        type=str,
        default=None,
        required=True,
        help="The prompt with identifier specifying the instance, e.g. 'photo of a TOK dog', 'in the style of TOK'",
    )
    parser.add_argument(
        "--max_sequence_length",
        type=int,
        default=77,
        help="Maximum sequence length to use with with the T5 text encoder",
    )
    parser.add_argument(
        "--validation_prompt",
        type=str,
        default=None,
        help="A prompt that is used during validation to verify that the model is learning.",
    )
    parser.add_argument(
        "--num_validation_images",
        type=int,
        default=4,
        help="Number of images that should be generated during validation with `validation_prompt`.",
    )
    parser.add_argument(
        "--validation_epochs",
        type=int,
        default=50,
        help=(
            "Run dreambooth validation every X epochs. Dreambooth validation consists of running the prompt"
            " `args.validation_prompt` multiple times: `args.num_validation_images`."
        ),
    )
    parser.add_argument(
        "--rank",
        type=int,
        default=4,
        help=("The dimension of the LoRA update matrices."),
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="sd3-dreambooth-lora-pipeline",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
    parser.add_argument(
        "--resolution",
        type=int,
        default=512,
        help=(
            "The resolution for input images, all the images in the train/validation dataset will be resized to this"
            " resolution"
        ),
    )
    parser.add_argument(
        "--center_crop",
        default=False,
        action="store_true",
        help=(
            "Whether to center crop the input images to the resolution. If not set, the images will be randomly"
            " cropped. The images will be resized to the resolution first before cropping."
        ),
    )
    parser.add_argument(
        "--random_flip",
        action="store_true",
        help="whether to randomly flip images horizontally",
    )
    parser.add_argument(
        "--train_batch_size", type=int, default=4, help="Batch size (per device) for the training dataloader."
    )
    parser.add_argument("--num_train_epochs", type=int, default=1)
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform.  If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--checkpointing_steps",
        type=int,
        default=500,
        help=(
            "Save a checkpoint of the training state every X updates. These checkpoints can be used both as final"
            " checkpoints in case they are better than the last checkpoint, and are also suitable for resuming"
            " training using `--resume_from_checkpoint`."
        ),
    )
    parser.add_argument(
        "--checkpoints_total_limit",
        type=int,
        default=None,
        help=("Max number of checkpoints to store."),
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help=(
            "Whether training should be resumed from a previous checkpoint. Use a path saved by"
            ' `--checkpointing_steps`, or `"latest"` to automatically select the last available checkpoint.'
        ),
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
        help="Initial learning rate to use.",
    )
    parser.add_argument(
        "--scale_lr",
        action="store_true",
        default=False,
        help="Scale the learning rate by the number of GPUs, gradient accumulation steps, and batch size.",
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="constant",
        help=(
            'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
            ' "constant", "constant_with_warmup"]'
        ),
    )
    parser.add_argument(
        "--lr_warmup_steps", type=int, default=500, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument(
        "--lr_num_cycles",
        type=int,
        default=1,
        help="Number of hard resets of the lr in cosine_with_restarts scheduler.",
    )
    parser.add_argument("--lr_power", type=float, default=1.0, help="Power factor of the polynomial scheduler.")
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=0,
        help=(
            "Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process."
        ),
    )
    parser.add_argument(
        "--weighting_scheme",
        type=str,
        default="logit_normal",
        choices=["sigma_sqrt", "logit_normal", "mode", "cosmap"],
    )
    parser.add_argument(
        "--logit_mean", type=float, default=0.0, help="mean to use when using the `'logit_normal'` weighting scheme."
    )
    parser.add_argument(
        "--logit_std", type=float, default=1.0, help="std to use when using the `'logit_normal'` weighting scheme."
    )
    parser.add_argument(
        "--mode_scale",
        type=float,
        default=1.29,
        help="Scale of mode weighting scheme. Only effective when using the `'mode'` as the `weighting_scheme`.",
    )
    parser.add_argument(
        "--use_8bit_adam",
        action="store_true",
        help="Whether or not to use 8-bit Adam from bitsandbytes. Ignored if optimizer is not set to AdamW",
    )

    parser.add_argument("--adam_beta1", type=float, default=0.9, help="The beta1 parameter for the Adam optimizer.")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="The beta2 parameter for the Adam optimizer.")
    parser.add_argument("--adam_weight_decay", type=float, default=1e-04, help="Weight decay to use for unet params")
    parser.add_argument(
        "--adam_epsilon",
        type=float,
        default=1e-08,
        help="Epsilon value for the Adam optimizer.",
    )
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument("--push_to_hub", action="store_true", help="Whether to push the model to the Hub.")
    parser.add_argument("--hub_token", type=str, default=None, help="The token to use to push to the Model Hub.")
    parser.add_argument(
        "--hub_model_id",
        type=str,
        default=None,
        help="The name of the repository to keep in sync with the local `output_dir`.",
    )
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help=(
            "[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to"
            " *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***."
        ),
    )
    parser.add_argument(
        "--allow_tf32",
        action="store_true",
        help=(
            "Whether or not to allow TF32 on Ampere GPUs. Can be used to speed up training. For more information, see"
            " https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices"
        ),
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="tensorboard",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`'
            ' (default), `"wandb"` and `"comet_ml"`. Use `"all"` to report to all integrations.'
        ),
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default=None,
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
            " 1.10.and an Nvidia Ampere GPU.  Default to the value of accelerate config of the current system or the"
            " flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config."
        ),
    )
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")

    # Pipeline parallelism specific arguments
    parser.add_argument(
        "--num_pipeline_stages",
        type=int,
        default=1,
        help="Number of pipeline stages (GPUs) to use for pipeline parallelism.",
    )
    parser.add_argument(
        "--pipeline_chunks",
        type=int,
        default=4,
        help="Number of micro-batches for GPipe-style pipeline parallelism.",
    )

    parser.add_argument(
        "--wandb_key",
        type=str,
        default=None,
        help='Wandb API key.',
    )
    parser.add_argument("--wandb_run_id", type=str, default=None,
                        help="Existing W&B run ID to resume logging into.")
    parser.add_argument("--wandb_project_name", type=str, default=None,
                        help="(Optional) Explicit project name when resuming/starting.")
    parser.add_argument("--tracker_name", type=str, default=None, help="Project tracker name")

    if input_args is not None:
        args = parser.parse_args(input_args)
    else:
        args = parser.parse_args()

    if args.instance_data_dir is None:
        raise ValueError("Specify `instance_data_dir`.")

    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    return args


class DreamBoothDataset(Dataset):
    """
    A dataset to prepare the instance and class images with the prompts for fine-tuning the model.
    It pre-processes the images.
    """

    def __init__(
        self,
        data_df_path,
        instance_data_root,
        instance_prompt,
        args,
        size=1024,
        center_crop=False,
    ):
        # Logistics
        self.size = size
        self.center_crop = center_crop

        self.args = args

        self.instance_prompt = instance_prompt
        self.instance_data_root = Path(instance_data_root)
        if not self.instance_data_root.exists():
            raise ValueError("Instance images root doesn't exists.")

        # Load images.
        self.instance_images = [Image.open(path) for path in list(Path(instance_data_root).iterdir())]
        self.image_hashes = [self.generate_image_hash(path) for path in list(Path(instance_data_root).iterdir())]
        # Image transformations
        self.pixel_values = self.apply_image_transformations(
            instance_images=self.instance_images, size=size, center_crop=center_crop
        )

        # Map hashes to embeddings.
        self.data_dict = self.map_image_hash_embedding(data_df_path=data_df_path)

        self.num_instance_images = len(self.instance_images)

    def __len__(self):
        return len(self.instance_images)

    def __getitem__(self, index):
        example = {}
        instance_image = self.pixel_values[index % self.num_instance_images]
        image_hash = self.image_hashes[index % self.num_instance_images]
        prompt_embeds, pooled_prompt_embeds = self.data_dict[image_hash]
        example["instance_images"] = instance_image
        example["prompt_embeds"] = prompt_embeds
        example["pooled_prompt_embeds"] = pooled_prompt_embeds
        return example

    def apply_image_transformations(self, instance_images, size, center_crop):
        pixel_values = []

        train_resize = transforms.Resize(size, interpolation=transforms.InterpolationMode.BILINEAR)
        train_crop = transforms.CenterCrop(size) if center_crop else transforms.RandomCrop(size)
        train_flip = transforms.RandomHorizontalFlip(p=1.0)
        train_transforms = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )
        for image in instance_images:
            image = exif_transpose(image)
            if not image.mode == "RGB":
                image = image.convert("RGB")
            image = train_resize(image)
            if self.args.random_flip and random.random() < 0.5:
                # flip
                image = train_flip(image)
            if self.args.center_crop:
                y1 = max(0, int(round((image.height - self.args.resolution) / 2.0)))
                x1 = max(0, int(round((image.width - self.args.resolution) / 2.0)))
                image = train_crop(image)
            else:
                y1, x1, h, w = train_crop.get_params(image, (self.args.resolution, self.args.resolution))
                image = crop(image, y1, x1, h, w)
            image = train_transforms(image)
            pixel_values.append(image)

        return pixel_values

    def convert_to_torch_tensor(self, embeddings: list):
        prompt_embeds = embeddings[0]
        pooled_prompt_embeds = embeddings[1]
        prompt_embeds = np.array(prompt_embeds).reshape(154, 4096)
        pooled_prompt_embeds = np.array(pooled_prompt_embeds).reshape(2048)
        return torch.from_numpy(prompt_embeds), torch.from_numpy(pooled_prompt_embeds)

    def map_image_hash_embedding(self, data_df_path):
        hashes_df = pd.read_parquet(data_df_path)
        data_dict = {}
        for i, row in hashes_df.iterrows():
            embeddings = [row["prompt_embeds"], row["pooled_prompt_embeds"]]
            prompt_embeds, pooled_prompt_embeds = self.convert_to_torch_tensor(embeddings=embeddings)
            data_dict.update({row["image_hash"]: (prompt_embeds, pooled_prompt_embeds)})
        return data_dict

    def generate_image_hash(self, image_path):
        with open(image_path, "rb") as f:
            img_data = f.read()
        return hashlib.sha256(img_data).hexdigest()


def collate_fn(examples):
    pixel_values = [example["instance_images"] for example in examples]
    prompt_embeds = [example["prompt_embeds"] for example in examples]
    pooled_prompt_embeds = [example["pooled_prompt_embeds"] for example in examples]

    pixel_values = torch.stack(pixel_values)
    pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()
    prompt_embeds = torch.stack(prompt_embeds)
    pooled_prompt_embeds = torch.stack(pooled_prompt_embeds)

    batch = {
        "pixel_values": pixel_values,
        "prompt_embeds": prompt_embeds,
        "pooled_prompt_embeds": pooled_prompt_embeds,
    }
    return batch


def main(args):
    if args.report_to == "wandb" and args.hub_token is not None:
        raise ValueError(
            "You cannot use both --report_to=wandb and --hub_token due to a security risk of exposing your token."
            " Please use `hf auth login` to authenticate with the Hub."
        )

    if torch.backends.mps.is_available() and args.mixed_precision == "bf16":
        # due to pytorch#99272, MPS does not yet support bfloat16.
        raise ValueError(
            "Mixed precision training with bfloat16 is not supported on MPS. Please use fp16 (recommended) or fp32 instead."
        )

    # logging_dir = Path(args.output_dir, args.logging_dir)
    os.makedirs(args.output_dir, exist_ok=True)

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )

    if args.seed is not None:
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)

    if args.push_to_hub:
        repo_id = create_repo(
            repo_id=args.hub_model_id or Path(args.output_dir).name,
            exist_ok=True,
        ).repo_id

    num_gpus = torch.cuda.device_count()
    if args.num_pipeline_stages > num_gpus:
        raise ValueError(f"Requested {args.num_pipeline_stages} stages but only {num_gpus} GPUs available")

    devices = [torch.device(f"cuda:{i}") for i in range(args.num_pipeline_stages)]
    logger.info(f"Using {args.num_pipeline_stages} GPUs for pipeline parallelism: {devices}")

    weight_dtype = torch.float32
    if args.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif args.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    # Load scheduler and VAE
    noise_scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="scheduler"
    )
    noise_scheduler_copy = copy.deepcopy(noise_scheduler)

    vae = AutoencoderKL.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="vae",
        revision=args.revision,
        variant=args.variant,
    )

    vae.requires_grad_(False)
    vae.to(devices[0], dtype=torch.float32)

    base_transformer = SD3Transformer2DModel.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="transformer",
        revision=args.revision,
        variant=args.variant,
        torch_dtype=weight_dtype,
    )

    transformer = PipelineParallelSD3Transformer(
        base_transformer,
        devices=devices,
        chunks=args.pipeline_chunks,
    )

    # Free the base transformer
    del base_transformer
    gc.collect()
    torch.cuda.empty_cache()

    # Freeze all parameters initially
    for param in transformer.parameters():
        param.requires_grad = False

    # now we will add new LoRA weights to the attention layers
    transformer_lora_config = LoraConfig(
        r=args.rank,
        lora_alpha=args.rank,
        init_lora_weights="gaussian",
        target_modules=["to_k", "to_q", "to_v", "to_out.0"],
    )

    for stage_blocks in transformer.block_stages:
        for block in stage_blocks:
            inject_adapter_in_model(transformer_lora_config, block, adapter_name="default")

    if args.gradient_checkpointing:
        transformer.enable_gradient_checkpointing()

    if args.allow_tf32 and torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True

    lora_parameters = [p for p in transformer.parameters() if p.requires_grad]
    num_trainable = sum(p.numel() for p in lora_parameters)
    logger.info(f"Number of trainable parameters: {num_trainable}")

    if args.scale_lr:
        args.learning_rate = (
            args.learning_rate * args.gradient_accumulation_steps * args.train_batch_size * args.num_pipeline_stages
        )

    # Make sure the trainable params are in float32.
    if args.mixed_precision == "fp16":
        for param in lora_parameters:
            param.data = param.data.to(torch.float32)

    if args.optimizer.lower() == "adamw":
        if args.use_8bit_adam:
            try:
                import bitsandbytes as bnb
            except ImportError:
                raise ImportError(
                    "To use 8-bit Adam, please install the bitsandbytes library: `pip install bitsandbytes`."
                )

            optimizer_class = bnb.optim.AdamW8bit
        else:
            optimizer_class = torch.optim.AdamW

        optimizer = optimizer_class(
            lora_parameters,
            lr=args.learning_rate,
            betas=(args.adam_beta1, args.adam_beta2),
            weight_decay=args.adam_weight_decay,
            eps=args.adam_epsilon,
        )

    # Dataset and DataLoader
    train_dataset = DreamBoothDataset(
        data_df_path=args.data_df_path,
        instance_data_root=args.instance_data_dir,
        instance_prompt=args.instance_prompt,
        args=args,
        size=args.resolution,
        center_crop=args.center_crop,
    )

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.train_batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=args.dataloader_num_workers,
    )

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps,
        num_training_steps=args.max_train_steps,
        num_cycles=args.lr_num_cycles,
        power=args.lr_power,
    )

    # Initialize wandb
    if args.report_to == "wandb":
        if not is_wandb_available():
            raise ImportError("Install wandb: `pip install wandb`")
        wandb.login(key=args.wandb_key)
        wandb.init(
            project=args.wandb_project_name or "sd3-dreambooth-pipeline",
            id=args.wandb_run_id,
            resume="must" if args.wandb_run_id else None,
            config=vars(args),
        )
    elif args.report_to == "tensorboard":
        writer = SummaryWriter(log_dir=args.output_dir + "/logs")

    # Initialize GPU monitoring (done for shared cluster)
    pynvml.nvmlInit()
    gpu_handles = [pynvml.nvmlDeviceGetHandleByIndex(i) for i in range(args.num_pipeline_stages)]

    # Train!
    total_batch_size = args.train_batch_size * args.gradient_accumulation_steps

    global_step = 0
    first_epoch = 0
    logger.info("***** Running training with Pipeline Parallelism *****")
    logger.info(f"  Num trainable parameters = {num_trainable}")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num epochs = {args.num_train_epochs}")
    logger.info(f"  Batch size = {args.train_batch_size}")
    logger.info(f"  Total train batch size (w. accumulation) = {total_batch_size}")
    logger.info(f"  Gradient accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    logger.info(f"  Pipeline stages = {args.num_pipeline_stages}")
    logger.info(f"  Micro-batches (chunks) = {args.pipeline_chunks}")

    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint != "latest":
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            # Get the mos recent checkpoint
            dirs = os.listdir(args.output_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if len(dirs) > 0 else None

        if path is None:
            logger.info(f"Checkpoint '{args.resume_from_checkpoint}' does not exist. Starting new training.")
            args.resume_from_checkpoint = None
        else:
            logger.info(f"Resuming from checkpoint {path}")
            checkpoint = torch.load(os.path.join(args.output_dir, path, "checkpoint.pt"))
            # Load LoRA state dict here
            global_step = int(path.split("-")[1])
            first_epoch = global_step // num_update_steps_per_epoch

    progress_bar = tqdm(
        range(args.max_train_steps),
        initial=global_step,
        desc="Steps",
    )

    def get_sigmas(timesteps, n_dim=4, dtype=torch.float32):
        sigmas = noise_scheduler_copy.sigmas.to(device=devices[0], dtype=dtype)
        schedule_timesteps = noise_scheduler_copy.timesteps.to(devices[0])
        timesteps = timesteps.to(devices[0])
        step_indices = [(schedule_timesteps == t).nonzero().item() for t in timesteps]

        sigma = sigmas[step_indices].flatten()
        while len(sigma.shape) < n_dim:
            sigma = sigma.unsqueeze(-1)
        return sigma

    for epoch in range(first_epoch, args.num_train_epochs):
        transformer.train()

        for step, batch in enumerate(train_dataloader):
            step_start_time = time.perf_counter()
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            start_event.record()

            pixel_values = batch["pixel_values"].to(device=devices[0], dtype=vae.dtype)

             # Convert images to latent space
            with torch.no_grad():
                model_input = vae.encode(pixel_values).latent_dist.sample()
                model_input = model_input * vae.config.scaling_factor
                model_input = model_input.to(dtype=weight_dtype)

            # Sample noise that we'll add to the latents
            noise = torch.randn_like(model_input)
            bsz = model_input.shape[0]

            # Sample a random timestep for each image
                # for weighting schemes where we sample timesteps non-uniformly
            u = compute_density_for_timestep_sampling(
                weighting_scheme=args.weighting_scheme,
                batch_size=bsz,
                logit_mean=args.logit_mean,
                logit_std=args.logit_std,
                mode_scale=args.mode_scale,
            )
            indices = (u * noise_scheduler_copy.config.num_train_timesteps).long()
            timesteps = noise_scheduler_copy.timesteps[indices].to(device=model_input.device)

            # Add noise according to flow matching.
            sigmas = get_sigmas(timesteps, n_dim=model_input.ndim, dtype=model_input.dtype)
            noisy_model_input = sigmas * noise + (1.0 - sigmas) * model_input

            # Predict the noise residual
            prompt_embeds = batch["prompt_embeds"].to(device=devices[0], dtype=weight_dtype)
            pooled_prompt_embeds = batch["pooled_prompt_embeds"].to(device=devices[0], dtype=weight_dtype)
            model_pred = transformer(
                hidden_states=noisy_model_input,
                timestep=timesteps,
                encoder_hidden_states=prompt_embeds,
                pooled_projections=pooled_prompt_embeds,
                return_dict=False,
            )[0]

            # FIX LATER!!!
            model_pred = model_pred.to(devices[0])

            sigmas = sigmas.to(devices[0])
            noisy_model_input = noisy_model_input.to(devices[0])

            # Follow: Section 5 of https://huggingface.co/papers/2206.00364.
                # Preconditioning of the model outputs.
            model_pred = model_pred * (-sigmas) + noisy_model_input

            # these weighting schemes use a uniform timestep sampling
                # and instead post-weight the loss
            weighting = compute_loss_weighting_for_sd3(weighting_scheme=args.weighting_scheme, sigmas=sigmas)

            # flow matching loss
            target = model_input.to(devices[0])

            # Compute regular loss.
            loss = torch.mean(
                (weighting.float() * (model_pred.float() - target.float()) ** 2).reshape(target.shape[0], -1),
                1,
            )
            loss = loss.mean()

            # Scale loss for gradient accumulation
            loss = loss / (args.gradient_accumulation_steps * args.pipeline_chunks)
            loss.backward()

            # End GPU timing
            end_event.record()
            torch.cuda.synchronize()
            gpu_compute_time_ms = start_event.elapsed_time(end_event)
            wall_time_ms = (time.perf_counter() - step_start_time) * 1000
            gpu_utilization_estimate = (gpu_compute_time_ms / wall_time_ms) * 100 if wall_time_ms > 0 else 0

            if (step + 1) % args.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(lora_parameters, args.max_grad_norm)

                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

                progress_bar.update(1)
                global_step += 1

                logs = {"loss": loss.detach().item() * args.gradient_accumulation_steps, "lr": lr_scheduler.get_last_lr()[0]}
                progress_bar.set_postfix(**logs)
                print(logs["loss"])

                if args.report_to == "wandb":
                    wandb.log(logs, step=global_step)
                elif args.report_to == "tensorboard":
                    writer.add_scalar("train/loss", logs["loss"], global_step)
                    writer.add_scalar("train/lr", logs["lr"], global_step)

                    # Log GPU compute utilization
                    writer.add_scalar("gpu/compute_time_ms", gpu_compute_time_ms, global_step)
                    writer.add_scalar("gpu/utilization_estimate", gpu_utilization_estimate, global_step)

                    # Log CPU metrics
                    writer.add_scalar("system/cpu_percent", psutil.cpu_percent(), global_step)
                    writer.add_scalar("system/ram_percent", psutil.virtual_memory().percent, global_step)

                    # Log GPU metrics for each pipeline stage (process-specific)
                    pid = os.getpid()
                    sm_utils = get_process_gpu_sm_util(pid)
                    for gpu_idx, handle in enumerate(gpu_handles):
                        # Get process-specific memory via pynvml
                        process_mem_gb = 0
                        try:
                            processes = pynvml.nvmlDeviceGetComputeRunningProcesses(handle)
                            for proc in processes:
                                if proc.pid == pid:
                                    process_mem_gb = proc.usedGpuMemory / 1e9
                                    break
                        except pynvml.NVMLError:
                            pass

                        # PyTorch memory tracking (most accurate for your process)
                        torch_allocated_gb = torch.cuda.memory_allocated(gpu_idx) / 1e9
                        torch_reserved_gb = torch.cuda.memory_reserved(gpu_idx) / 1e9

                        writer.add_scalar(f"gpu{gpu_idx}/process_memory_gb", process_mem_gb, global_step)
                        writer.add_scalar(f"gpu{gpu_idx}/torch_allocated_gb", torch_allocated_gb, global_step)
                        writer.add_scalar(f"gpu{gpu_idx}/torch_reserved_gb", torch_reserved_gb, global_step)

                        # Log per-process SM utilization (nvtop-style)
                        if gpu_idx in sm_utils:
                            writer.add_scalar(f"gpu{gpu_idx}/sm_utilization", sm_utils[gpu_idx], global_step)

                if global_step % args.checkpointing_steps == 0:
                    if args.checkpoints_total_limit is not None:
                        checkpoints = os.listdir(args.output_dir)
                        checkpoints = [d for d in checkpoints if d.startswith("checkpoint")]
                        checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))

                        if len(checkpoints) >= args.checkpoints_total_limit:
                            num_to_remove = len(checkpoints) - args.checkpoints_total_limit + 1
                            for removing_checkpoint in checkpoints[:num_to_remove]:
                                shutil.rmtree(os.path.join(args.output_dir, removing_checkpoint))

                    save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                    os.makedirs(save_path, exist_ok=True)

                    # Save LoRA weights
                    lora_state_dict = {}
                    for name, param in transformer.named_parameters():
                        if param.requires_grad:
                            lora_state_dict[name] = param.cpu().clone()

                    torch.save({
                        "lora_state_dict": lora_state_dict,
                        "optimizer": optimizer.state_dict(),
                        "lr_scheduler": lr_scheduler.state_dict(),
                        "global_step": global_step,
                    }, os.path.join(save_path, "checkpoint.pt"))

                    logger.info(f"Saved checkpoint to {save_path}")

            if global_step >= args.max_train_steps:
                break

        # # Validation
        # if args.validation_prompt is not None and epoch % args.validation_epochs == 0:
        #     # Consolidate LoRA weights to a single transformer for validation
        #     val_transformer = SD3Transformer2DModel.from_pretrained(
        #         args.pretrained_model_name_or_path,
        #         subfolder="transformer",
        #         revision=args.revision,
        #         variant=args.variant,
        #         torch_dtype=weight_dtype,
        #     )

        #     # Apply LoRA config
        #     val_transformer.add_adapter(transformer_lora_config)

        #     # Copy LoRA weights from pipeline parallel model
        #     lora_state_dict = {}
        #     for stage_idx, stage_blocks in enumerate(transformer.block_stages):
        #         for block_idx, block in enumerate(stage_blocks):
        #             for name, param in block.named_parameters():
        #                 if param.requires_grad and 'lora' in name:
        #                     # Map back to original block index
        #                     orig_block_idx = sum(len(transformer.block_stages[i]) for i in range(stage_idx)) + block_idx
        #                     orig_name = f"transformer_blocks.{orig_block_idx}.{name}"
        #                     lora_state_dict[orig_name] = param.data.cpu()

        #     # Load state dict
        #     val_transformer.load_state_dict(lora_state_dict, strict=False)

        #     pipeline = StableDiffusion3Pipeline.from_pretrained(
        #         args.pretrained_model_name_or_path,
        #         vae=vae,
        #         transformer=val_transformer,
        #         revision=args.revision,
        #         variant=args.variant,
        #         torch_dtype=weight_dtype,
        #     )
        #     pipeline_args = {"prompt": args.validation_prompt}
        #     images = log_validation(
        #         pipeline=pipeline,
        #         args=args,
        #         pipeline_args=pipeline_args,
        #         epoch=epoch,
        #         writer=writer,
        #     )
        #     del val_transformer, pipeline
        #     torch.cuda.empty_cache()
        #     gc.collect()

    logger.info("Saving final LoRA weights...")
    lora_state_dict = {}
    for name, param in transformer.named_parameters():
        if param.requires_grad:
            lora_state_dict[f"transformer.{name}"] = param.cpu().clone()

    final_save_path = os.path.join(args.output_dir, "pytorch_lora_weights.safetensors")
    from safetensors.torch import save_file
    save_file(lora_state_dict, final_save_path)

    logger.info(f"Training complete. LoRA weights saved to {final_save_path}")

    # Final validation
    # if args.validation_prompt and args.num_validation_images > 0:
    #     pipeline = StableDiffusion3Pipeline.from_pretrained(
    #         args.pretrained_model_name_or_path,
    #         revision=args.revision,
    #         variant=args.variant,
    #         torch_dtype=weight_dtype,
    #     )
    #     pipeline.load_lora_weights(args.output_dir)

    #     pipeline_args = {"prompt": args.validation_prompt}
    #     images = log_validation(
    #         pipeline=pipeline,
    #         args=args,
    #         pipeline_args=pipeline_args,
    #         epoch=epoch,
    #         is_final_validation=True,
    #     )

    #     if args.push_to_hub:
    #         save_model_card(
    #             repo_id,
    #             images=images,
    #             base_model=args.pretrained_model_name_or_path,
    #             instance_prompt=args.instance_prompt,
    #             validation_prompt=args.validation_prompt,
    #             repo_folder=args.output_dir,
    #         )
    #         upload_folder(
    #             repo_id=repo_id,
    #             folder_path=args.output_dir,
    #             commit_message="End of training",
    #             ignore_patterns=["step_*", "epoch_*"],
    #         )

    if args.report_to == "wandb":
        wandb.finish()
    elif args.report_to == "tensorboard":
        writer.close()
        pynvml.nvmlShutdown()


if __name__ == "__main__":
    args = parse_args()
    main(args)
