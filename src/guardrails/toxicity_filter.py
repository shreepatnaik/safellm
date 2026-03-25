"""Lightweight toxicity detection for input and output content."""
import re
from dataclasses import dataclass, field

@dataclass
class ToxicityResult:
    is_toxic: bool
    score: float = 0.0
    flagged_categories: list = field(default_factory=list)


class ToxicityFilter:
    """Keyword and pattern-based toxicity filtering.

    For production use, replace with a model-based classifier
    (e.g., HuggingFace toxicity model or Perspective API).
    """

    CATEGORIES = {
        "threat": [
            r"\bi['']?ll\s+kill\b", r"\bthreat\b", r"\bhurt\s+you\b",
            r"\bbomb\b", r"\bshoot\b", r"\bweapon\b",
        ],
        "harassment": [
            r"\bstupid\b", r"\bidiot\b", r"\bdumb\b",
            r"\bushless\b", r"\bincompetent\b",
        ],
        "self_harm": [
            r"\bkill\s+myself\b", r"\bsuicid\b", r"\bself[- ]harm\b",
            r"\bdon['']?t\s+want\s+to\s+live\b",
        ],
        "manipulation": [
            r"\bignore\s+(all\s+)?previous\b", r"\byou\s+are\s+now\b",
            r"\bpretend\s+(to\s+be|you\s+are)\b", r"\bbypass\b",
            r"\bjailbreak\b", r"\bDAN\b", r"\bsystem\s+prompt\b",
        ],
    }

    def __init__(self, threshold: float = 0.5):
        self.threshold = threshold
        self._compiled = {
            cat: [re.compile(p, re.IGNORECASE) for p in patterns]
            for cat, patterns in self.CATEGORIES.items()
        }

    def check(self, text: str) -> ToxicityResult:
        """Check text for toxic content across categories."""
        flagged = []
        total_matches = 0

        for category, patterns in self._compiled.items():
            matches = sum(1 for p in patterns if p.search(text))
            if matches > 0:
                score = min(matches / len(patterns), 1.0)
                flagged.append({"category": category, "score": score, "matches": matches})
                total_matches += matches

        overall_score = min(total_matches / 5, 1.0)

        return ToxicityResult(
            is_toxic=overall_score >= self.threshold or any(
                f["category"] in ("threat", "self_harm", "manipulation") and f["matches"] > 0
                for f in flagged
            ),
            score=overall_score,
            flagged_categories=flagged,
        )
