"""Interactive CLI chatbot with guardrails pipeline.

Usage:
    python src/inference/chat.py --model_path checkpoints/best_model/
    python src/inference/chat.py --use_openai  # Use OpenAI API instead of local model
"""
import argparse
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from guardrails import GuardrailsPipeline, GuardResult

# Optional imports
try:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

try:
    import openai
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False


class LocalLLM:
    """Inference with a locally fine-tuned model."""

    def __init__(self, model_path: str):
        if not HAS_TORCH:
            raise ImportError("PyTorch and transformers required. pip install torch transformers")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path, torch_dtype=torch.bfloat16, device_map="auto"
        )
        self.pipe = pipeline("text-generation", model=self.model, tokenizer=self.tokenizer)

    def generate(self, prompt: str, max_tokens: int = 512) -> str:
        result = self.pipe(prompt, max_new_tokens=max_tokens, do_sample=True,
                          temperature=0.7, top_p=0.9)
        return result[0]["generated_text"][len(prompt):].strip()


class OpenAILLM:
    """Inference with OpenAI API."""

    def __init__(self, model: str = "gpt-4o-mini"):
        if not HAS_OPENAI:
            raise ImportError("openai required. pip install openai")
        self.client = openai.OpenAI()
        self.model = model

    def generate(self, prompt: str, max_tokens: int = 512) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You are a helpful IT helpdesk assistant. "
                 "Answer based on company documentation. If unsure, say you don't know."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=max_tokens,
            temperature=0.7,
        )
        return response.choices[0].message.content


def print_guard_status(result: GuardResult, label: str):
    """Print guardrail check results."""
    for check in result.checks:
        icon = {"pass": "✅", "masked": "🔒", "scrubbed": "🔒",
                "warning": "⚠️", "blocked": "🚫"}.get(check.status, "❓")
        detail = f" — {check.details}" if check.details else ""
        print(f"  {icon} {check.name}: {check.status}{detail}")


def chat_loop(llm, guards: GuardrailsPipeline):
    """Interactive chat loop with guardrails."""
    print("\n" + "=" * 60)
    print("🛡️  SafeLLM — IT Helpdesk Chat (with Guardrails)")
    print("=" * 60)
    print("Type your question. Type 'quit' to exit.\n")

    conversation_history = []

    while True:
        try:
            query = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            break

        if not query or query.lower() in ("quit", "exit", "q"):
            print("\nGoodbye!")
            break

        # INPUT GUARDRAILS
        input_result = guards.check_input(query)

        if input_result.blocked:
            print(f"\n🚫 Blocked: {input_result.reason}")
            print("  Input guardrails:")
            print_guard_status(input_result, "input")
            print()
            continue

        safe_query = input_result.text
        if safe_query != query:
            print(f"  🔒 Query sanitized (PII masked)")

        # GENERATE RESPONSE
        try:
            raw_response = llm.generate(safe_query)
        except Exception as e:
            print(f"\n❌ Model error: {e}\n")
            continue

        # OUTPUT GUARDRAILS
        output_result = guards.check_output(raw_response)

        if output_result.blocked:
            print(f"\nBot: I'm sorry, I can't provide that response.")
            print(f"  🚫 Output blocked: {output_result.reason}")
            print()
            continue

        # Display response
        print(f"\nBot: {output_result.text}")

        # Show guardrail status
        print("\n  Guardrails:")
        print_guard_status(input_result, "input")
        print_guard_status(output_result, "output")
        print()


def main():
    parser = argparse.ArgumentParser(description="SafeLLM Chat")
    parser.add_argument("--model_path", type=str, default=None, help="Path to fine-tuned model")
    parser.add_argument("--use_openai", action="store_true", help="Use OpenAI API instead")
    parser.add_argument("--openai_model", type=str, default="gpt-4o-mini")
    args = parser.parse_args()

    guards = GuardrailsPipeline()

    if args.use_openai:
        llm = OpenAILLM(model=args.openai_model)
    elif args.model_path:
        llm = LocalLLM(model_path=args.model_path)
    else:
        print("Specify --model_path or --use_openai")
        print("Example: python src/inference/chat.py --use_openai")
        sys.exit(1)

    chat_loop(llm, guards)


if __name__ == "__main__":
    main()
