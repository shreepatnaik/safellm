"""Convert raw documents (MD/TXT) into instruction-tuning JSONL format.

Usage:
    python src/training/prepare_data.py --input_dir data/raw_docs/ --output data/training_data.jsonl
"""
import os
import json
import argparse
import re

# Templates for generating Q&A pairs from document chunks
QA_TEMPLATES = [
    ("What is {topic}?", "Based on our documentation, {chunk}"),
    ("How do I {action}?", "{chunk}"),
    ("Explain the process for {topic}.", "{chunk}"),
    ("What are the steps to {action}?", "{chunk}"),
    ("Can you help me with {topic}?", "Sure! {chunk}"),
]


def read_file(filepath: str) -> str:
    """Read a text/markdown file."""
    with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()


def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> list[str]:
    """Split text into overlapping chunks."""
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunk = " ".join(words[i:i + chunk_size])
        if len(chunk.split()) > 20:
            chunks.append(chunk)
    return chunks


def extract_topic(chunk: str) -> str:
    """Extract a rough topic from a text chunk."""
    first_sentence = chunk.split(".")[0].strip()
    words = first_sentence.split()[:6]
    return " ".join(words).lower().rstrip(",.:;")


def generate_qa_pairs(chunk: str, source_file: str) -> list[dict]:
    """Generate instruction-tuning pairs from a document chunk."""
    pairs = []
    topic = extract_topic(chunk)

    # Direct Q&A
    pairs.append({
        "instruction": f"What does the documentation say about {topic}?",
        "input": "",
        "output": chunk.strip(),
        "source": source_file,
    })

    # Summarization
    if len(chunk.split()) > 50:
        pairs.append({
            "instruction": f"Summarize the key points about {topic}.",
            "input": chunk.strip(),
            "output": ". ".join(chunk.split(". ")[:2]).strip() + ".",
            "source": source_file,
        })

    return pairs


def process_directory(input_dir: str, output_path: str, chunk_size: int = 500):
    """Process all documents in a directory into training data."""
    all_pairs = []
    supported_extensions = {".md", ".txt", ".text"}

    for root, _, files in os.walk(input_dir):
        for filename in sorted(files):
            ext = os.path.splitext(filename)[1].lower()
            if ext not in supported_extensions:
                continue

            filepath = os.path.join(root, filename)
            print(f"Processing: {filepath}")

            text = read_file(filepath)
            # Clean up markdown
            text = re.sub(r"#+ ", "", text)
            text = re.sub(r"\[([^\]]+)\]\([^\)]+\)", r"\1", text)
            text = re.sub(r"\*\*([^*]+)\*\*", r"\1", text)
            text = re.sub(r"\n{3,}", "\n\n", text)

            chunks = chunk_text(text, chunk_size=chunk_size)
            for chunk in chunks:
                pairs = generate_qa_pairs(chunk, filename)
                all_pairs.extend(pairs)

    # Write JSONL
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w") as f:
        for pair in all_pairs:
            f.write(json.dumps(pair) + "\n")

    print(f"\nGenerated {len(all_pairs)} training pairs from {len(os.listdir(input_dir))} files")
    print(f"Saved to: {output_path}")


def create_sample_data(output_dir: str = "data"):
    """Create sample IT helpdesk documents for demo purposes."""
    os.makedirs(os.path.join(output_dir, "raw_docs"), exist_ok=True)

    docs = {
        "password_reset.md": """# Password Reset Procedure

To reset your corporate password, follow these steps:

1. Go to the IT Self-Service Portal at https://portal.internal
2. Click "Forgot Password" on the login page
3. Enter your employee badge ID (format: EMP-XXXXX)
4. You will receive a verification code on your registered phone
5. Enter the code and create a new password

Password requirements:
- Minimum 12 characters
- At least one uppercase, one lowercase, one number, one special character
- Cannot reuse the last 5 passwords
- Expires every 90 days

If you are locked out after 5 failed attempts, contact the IT Help Desk at extension 5555 or email helpdesk@company.internal. Include your name and badge ID in the request.
""",
        "vpn_setup.md": """# VPN Setup Guide

The company VPN is required for remote access to internal systems.

## Windows Setup
1. Download GlobalProtect from the Software Center
2. Open GlobalProtect and enter the portal address: vpn.company.internal
3. Sign in with your corporate credentials
4. Select the "Corporate" gateway for full access or "Split-Tunnel" for partial access

## macOS Setup
1. Download GlobalProtect from the Self-Service app
2. Follow the same portal configuration as Windows

## Troubleshooting
- "Authentication Failed": Ensure your password has not expired
- "Gateway Timeout": Try switching between Corporate and Split-Tunnel gateways
- "Certificate Error": Update your system clock and try again

For persistent issues, submit a ticket through ServiceNow or call IT at extension 5555.
""",
        "software_request.md": """# Software Installation Request

To request new software installation:

1. Log into ServiceNow at https://servicenow.internal
2. Navigate to Service Catalog > Software Request
3. Search for the software in the approved catalog
4. If the software is not listed, submit a "New Software Evaluation" request
5. Manager approval is required for all software requests
6. Standard software is provisioned within 2 business days
7. Non-standard software requires security review (5-10 business days)

Approved software includes: Microsoft Office, Adobe Creative Cloud, Slack, Zoom, Visual Studio Code, Python, Docker Desktop, and Postman.

Restricted software (requires security review): Any open-source database, custom browser extensions, remote desktop tools, and VPN clients other than GlobalProtect.
""",
    }

    for filename, content in docs.items():
        path = os.path.join(output_dir, "raw_docs", filename)
        with open(path, "w") as f:
            f.write(content)
        print(f"Created: {path}")

    # Generate training data from sample docs
    process_directory(
        os.path.join(output_dir, "raw_docs"),
        os.path.join(output_dir, "training_data.jsonl"),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, default="data/raw_docs/")
    parser.add_argument("--output", type=str, default="data/training_data.jsonl")
    parser.add_argument("--chunk_size", type=int, default=500)
    parser.add_argument("--create_sample", action="store_true", help="Create sample IT docs for demo")
    args = parser.parse_args()

    if args.create_sample:
        create_sample_data()
    else:
        process_directory(args.input_dir, args.output, args.chunk_size)
