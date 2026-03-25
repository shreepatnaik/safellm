#!/bin/bash
# Launch distributed training with DDP
# Usage: ./scripts/launch_ddp.sh [NUM_GPUS]

NUM_GPUS=${1:-2}

echo "🚀 Launching DDP training on $NUM_GPUS GPUs..."

torchrun \
    --nproc_per_node=$NUM_GPUS \
    --master_port=29500 \
    src/training/train.py \
    --model_name meta-llama/Llama-3.2-1B \
    --strategy ddp \
    --use_lora \
    --use_amp \
    --batch_size 4 \
    --epochs 3

echo "✅ Training complete!"
