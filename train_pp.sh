#!/bin/bash

# export ROCR_VISIBLE_DEVICES=0,1,2,3
# export HIP_VISIBLE_DEVICES=0,1,2,3
export ROCR_VISIBLE_DEVICES=0,1
export HIP_VISIBLE_DEVICES=0,1
# export ROCR_VISIBLE_DEVICES=0
# export HIP_VISIBLE_DEVICES=0


unset CUDA_VISIBLE_DEVICES  # ROCm stack ignores CUDA var
unset HSA_OVERRIDE_GFX_VERSION
unset AMD_SERIALIZE_KERNEL
unset TORCH_USE_HIP_DSA
unset HSA_ENABLE_SDMA
unset HSA_ENABLE_INTERRUPT
unset AMD_DIRECT_DISPATCH

# export HIP_LAUNCH_BLOCKING=1   # optional: keep only while debugging

# MIOpen environment variables to fix miopenStatusInternalError
export MIOPEN_DISABLE_CACHE=1
export MIOPEN_USER_DB_PATH=""
export MIOPEN_CACHE_DIR=""
export MIOPEN_DEBUG_FIND_MODE=1
export MIOPEN_FIND_MODE=1

# Ensure ROCm in the path (adapt if your ROCm lives elsewhere)
export ROCM_PATH=/opt/rocm
export LD_LIBRARY_PATH=$ROCM_PATH/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
export PATH=$ROCM_PATH/bin:$PATH

# --------------------------------------------------------------------------------------------------------------------

export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=8
export NUMEXPR_MAX_THREADS=8


export DATA_BASE="/data"
export HOME="/home/spotai_small/kuvshinova"
export WANDB_DIR="$HOME/.cache/wandb"
export WANDB_CACHE_DIR="$HOME/.cache/wandb"
# mkdir -p "$DATA_BASE/.cache" "$DATA_BASE/.conda" "$DATA_BASE/.pip" "$DATA_BASE/.huggingface"
# export XDG_CACHE_HOME="$DATA_BASE/.cache"
export HF_HOME="$HOME/.cache/huggingface"
export HF_HUB_CACHE="$HOME/.cache/huggingface/hub"
export TRANSFORMERS_CACHE="$HOME/.cache/huggingface/transformers"
# export TORCH_HOME="$DATA_BASE/.cache/torch"
# export PIP_CACHE_DIR="$DATA_BASE/.pip/cache"
# export CONDA_PKGS_DIRS="$DATA_BASE/.conda/pkgs" 
export HF_TOKEN="YOUR_TOKEN"
export WANDB_API_KEY="YOUR_KEY"


# Pipeline Parallel training (no accelerate needed - manages GPUs internally)
python train_dreambooth_lora_sd3_pipeline_parallel.py \
  --pretrained_model_name_or_path="stabilityai/stable-diffusion-3-medium-diffusers" \
  --instance_data_dir="$HOME/Parallel-SD3-Dreambooth-dev42/data/images" \
  --data_df_path="$HOME/Parallel-SD3-Dreambooth-dev42/data/style_embeddings.parquet" \
  --output_dir="trained-sd3-lora-pipeline-parallel_3gpu_test_fix" \
  --mixed_precision="fp16" \
  --instance_prompt="a photo with sks style" \
  --resolution=1024 \
  --train_batch_size=8 \
  --gradient_accumulation_steps=1 \
  --learning_rate=1e-4 \
  --report_to="tensorboard" \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --num_train_epochs=50 \
  --seed="0" \
  --num_pipeline_stages=3 \
  --pipeline_chunks=4 \
  --gradient_checkpointing # \
  # --wandb_key $WANDB_API_KEY \
  # --wandb_project_name "sd3-dreambooth-pipeline"
