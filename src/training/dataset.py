"""PyTorch Dataset for instruction-tuning from JSONL files."""
import json
import torch
from torch.utils.data import Dataset

ALPACA_TEMPLATE = """Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Input:
{input}

### Response:
{output}"""

ALPACA_TEMPLATE_NO_INPUT = """Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Response:
{output}"""


class InstructionDataset(Dataset):
    """Dataset for instruction-tuning in Alpaca format.

    Expected JSONL format:
        {"instruction": "...", "input": "...", "output": "..."}
    """

    def __init__(self, data_path: str, tokenizer, max_length: int = 2048):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.samples = []

        with open(data_path, "r") as f:
            for line in f:
                line = line.strip()
                if line:
                    self.samples.append(json.loads(line))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        # Format as Alpaca template
        if sample.get("input", "").strip():
            text = ALPACA_TEMPLATE.format(
                instruction=sample["instruction"],
                input=sample["input"],
                output=sample["output"],
            )
        else:
            text = ALPACA_TEMPLATE_NO_INPUT.format(
                instruction=sample["instruction"],
                output=sample["output"],
            )

        # Tokenize
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        input_ids = encoding["input_ids"].squeeze()
        attention_mask = encoding["attention_mask"].squeeze()

        # Labels = input_ids (causal LM), with padding tokens set to -100
        labels = input_ids.clone()
        labels[attention_mask == 0] = -100

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }
