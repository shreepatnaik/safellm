"""Distributed fine-tuning of LLMs with DDP/FSDP, LoRA, and AMP.

Usage:
    # Single GPU
    python src/training/train.py --model_name meta-llama/Llama-3.2-1B

    # Multi-GPU DDP
    torchrun --nproc_per_node=2 src/training/train.py --strategy ddp

    # Multi-GPU FSDP with LoRA
    torchrun --nproc_per_node=4 src/training/train.py --strategy fsdp --use_lora --use_amp
"""
import os
import sys
import json
import time
import argparse
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from tqdm import tqdm

# Optional imports — gracefully handle missing GPU/FSDP
try:
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
    from torch.distributed.fsdp import ShardingStrategy
    HAS_FSDP = True
except ImportError:
    HAS_FSDP = False

try:
    from torch.cuda.amp import GradScaler, autocast
    HAS_AMP = True
except ImportError:
    HAS_AMP = False

from transformers import AutoModelForCausalLM, AutoTokenizer

try:
    from peft import LoraConfig, get_peft_model, TaskType
    HAS_PEFT = True
except ImportError:
    HAS_PEFT = False

from dataset import InstructionDataset


def setup_distributed():
    """Initialize distributed process group."""
    if "RANK" not in os.environ:
        return 0, 1, torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dist.init_process_group("nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    torch.cuda.set_device(rank)
    device = torch.device(f"cuda:{rank}")
    return rank, world_size, device


def cleanup():
    """Clean up distributed process group."""
    if dist.is_initialized():
        dist.destroy_process_group()


def log(rank, msg):
    """Only print from rank 0."""
    if rank == 0:
        print(msg)


def load_model(model_name, device, use_lora=False, dtype=torch.bfloat16):
    """Load base model with optional LoRA adapters."""
    log(0, f"Loading model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=dtype,
        trust_remote_code=True,
    )

    if use_lora:
        if not HAS_PEFT:
            raise ImportError("peft is required for LoRA. Install: pip install peft")
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=16,
            lora_alpha=32,
            lora_dropout=0.05,
            target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
        )
        model = get_peft_model(model, lora_config)
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())
        log(0, f"LoRA enabled: {trainable:,} / {total:,} trainable params ({100*trainable/total:.2f}%)")

    return model, tokenizer


def wrap_model(model, device, strategy, rank):
    """Wrap model with DDP or FSDP."""
    model = model.to(device)

    if strategy == "ddp":
        model = DDP(model, device_ids=[rank], find_unused_parameters=False)
        log(rank, "Wrapped model with DDP")
    elif strategy == "fsdp":
        if not HAS_FSDP:
            raise ImportError("FSDP requires PyTorch 2.0+")
        model = FSDP(
            model,
            sharding_strategy=ShardingStrategy.FULL_SHARD,
            use_orig_params=True,
            device_id=rank,
        )
        log(rank, "Wrapped model with FSDP")
    else:
        log(rank, "Single GPU mode (no wrapping)")

    return model


def train_epoch(model, loader, optimizer, scaler, device, use_amp, rank, epoch):
    """Run one training epoch."""
    model.train()
    total_loss = 0
    num_batches = 0

    pbar = tqdm(loader, desc=f"Epoch {epoch}", disable=(rank != 0))
    for batch in pbar:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        optimizer.zero_grad()

        if use_amp and HAS_AMP:
            with autocast(device_type="cuda", dtype=torch.bfloat16):
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        total_loss += loss.item()
        num_batches += 1

        if rank == 0:
            pbar.set_postfix({"loss": f"{loss.item():.4f}", "avg_loss": f"{total_loss/num_batches:.4f}"})

    return total_loss / max(num_batches, 1)


def save_checkpoint(model, tokenizer, save_dir, rank, strategy):
    """Save model checkpoint from rank 0."""
    if rank != 0:
        return

    os.makedirs(save_dir, exist_ok=True)

    if strategy == "fsdp":
        # FSDP requires special saving
        with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT):
            state = model.state_dict()
            torch.save(state, os.path.join(save_dir, "model.pt"))
    elif strategy == "ddp":
        model.module.save_pretrained(save_dir)
    else:
        model.save_pretrained(save_dir)

    tokenizer.save_pretrained(save_dir)
    log(rank, f"Checkpoint saved: {save_dir}")


def main(args):
    rank, world_size, device = setup_distributed()

    log(rank, f"{'='*60}")
    log(rank, f"SafeLLM Distributed Training")
    log(rank, f"{'='*60}")
    log(rank, f"Model:    {args.model_name}")
    log(rank, f"Strategy: {args.strategy} ({world_size} GPU{'s' if world_size > 1 else ''})")
    log(rank, f"LoRA:     {'enabled' if args.use_lora else 'disabled'}")
    log(rank, f"AMP:      {'enabled' if args.use_amp else 'disabled'}")
    log(rank, f"Epochs:   {args.epochs}")
    log(rank, f"Batch:    {args.batch_size} x {world_size} = {args.batch_size * world_size} effective")
    log(rank, f"{'='*60}")

    # Load model and tokenizer
    model, tokenizer = load_model(args.model_name, device, args.use_lora)

    # Load dataset
    dataset = InstructionDataset(
        data_path=args.data_path,
        tokenizer=tokenizer,
        max_length=args.max_seq_length,
    )
    log(rank, f"Dataset: {len(dataset)} samples")

    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank) if world_size > 1 else None
    loader = DataLoader(
        dataset, batch_size=args.batch_size, sampler=sampler,
        shuffle=(sampler is None), num_workers=2, pin_memory=True,
    )

    # Wrap model
    model = wrap_model(model, device, args.strategy if world_size > 1 else "none", rank)

    # Optimizer and scaler
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr, weight_decay=0.01,
    )
    scaler = GradScaler(enabled=args.use_amp) if HAS_AMP else None

    # Training loop
    best_loss = float("inf")
    history = []
    t_start = time.time()

    for epoch in range(1, args.epochs + 1):
        if sampler:
            sampler.set_epoch(epoch)

        t0 = time.time()
        avg_loss = train_epoch(model, loader, optimizer, scaler, device, args.use_amp, rank, epoch)
        elapsed = time.time() - t0

        throughput = len(dataset) / elapsed
        history.append({"epoch": epoch, "loss": avg_loss, "time_s": elapsed, "throughput": throughput})

        log(rank, f"Epoch {epoch}/{args.epochs} | Loss: {avg_loss:.4f} | "
            f"Time: {elapsed:.1f}s | Throughput: {throughput:.1f} samples/s")

        # Save best model
        if avg_loss < best_loss:
            best_loss = avg_loss
            save_checkpoint(model, tokenizer, os.path.join(args.save_dir, "best_model"), rank,
                          args.strategy if world_size > 1 else "none")

    total_time = time.time() - t_start
    log(rank, f"\nTraining complete in {total_time:.1f}s")
    log(rank, f"Best loss: {best_loss:.4f}")

    # Save training history
    if rank == 0:
        with open(os.path.join(args.save_dir, "training_history.json"), "w") as f:
            json.dump(history, f, indent=2)

    cleanup()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SafeLLM Distributed Fine-Tuning")
    parser.add_argument("--model_name", type=str, default="meta-llama/Llama-3.2-1B")
    parser.add_argument("--data_path", type=str, default="data/training_data.jsonl")
    parser.add_argument("--save_dir", type=str, default="checkpoints")
    parser.add_argument("--strategy", type=str, default="ddp", choices=["ddp", "fsdp"])
    parser.add_argument("--use_lora", action="store_true")
    parser.add_argument("--use_amp", action="store_true")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--max_seq_length", type=int, default=2048)
    args = parser.parse_args()
    main(args)
