#!/bin/bash
# Launch distributed training with FSDP (for large models)
# Usage: ./scripts/launch_fsdp.sh [NUM_GPUS]

NUM_GPUS=${1:-4}

echo "🚀 Launching FSDP training on $NUM_GPUS GPUs..."

torchrun \
    --nproc_per_node=$NUM_GPUS \
    --master_port=29500 \
    src/training/train.py \
    --model_name meta-llama/Llama-3.2-3B \
    --strategy fsdp \
    --use_lora \
    --use_amp \
    --batch_size 2 \
    --epochs 3

echo "✅ Training complete!"
