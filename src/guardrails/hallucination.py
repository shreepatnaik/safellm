"""Source-grounded hallucination detection for RAG responses."""
from dataclasses import dataclass, field

@dataclass
class HallucinationResult:
    score: float  # 1.0 = fully grounded, 0.0 = fully hallucinated
    flagged: bool
    total_claims: int
    supported_claims: int
    unsupported_claims: list = field(default_factory=list)


class HallucinationChecker:
    """Check if LLM response claims are grounded in retrieved source documents.

    Uses keyword overlap as a lightweight proxy. For production,
    use NLI-based entailment (e.g., cross-encoder/nli-deberta-v3-base).
    """

    def __init__(self, threshold: float = 0.7):
        self.threshold = threshold

    def check(self, response: str, sources: list[str]) -> HallucinationResult:
        """Score how well the response is grounded in sources."""
        claims = self._extract_claims(response)
        if not claims:
            return HallucinationResult(1.0, False, 0, 0)

        combined_sources = " ".join(sources).lower()
        source_words = set(combined_sources.split())

        supported, unsupported = [], []
        for claim in claims:
            if self._is_supported(claim, source_words):
                supported.append(claim)
            else:
                unsupported.append(claim)

        score = len(supported) / len(claims) if claims else 1.0

        return HallucinationResult(
            score=score,
            flagged=score < self.threshold,
            total_claims=len(claims),
            supported_claims=len(supported),
            unsupported_claims=unsupported,
        )

    def _extract_claims(self, text: str) -> list[str]:
        """Split response into verifiable factual sentences."""
        sentences = [s.strip() for s in text.replace("\n", ". ").split(". ") if len(s.strip()) > 10]
        factual_words = {"is", "are", "was", "were", "has", "have", "requires", "takes",
                        "costs", "includes", "provides", "supports", "runs", "uses"}
        return [s for s in sentences if any(w in s.lower().split() for w in factual_words)]

    def _is_supported(self, claim: str, source_words: set) -> bool:
        """Check if claim has sufficient keyword overlap with sources."""
        stop_words = {"the", "a", "an", "is", "are", "was", "were", "it", "to", "for",
                     "and", "or", "in", "on", "at", "of", "by", "with", "that", "this"}
        claim_words = set(claim.lower().split()) - stop_words
        if not claim_words:
            return True
        overlap = len(claim_words & source_words)
        return (overlap / len(claim_words)) > 0.35
