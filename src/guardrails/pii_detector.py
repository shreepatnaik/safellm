"""PII detection and masking for enterprise chatbot safety."""
import re
from dataclasses import dataclass, field

@dataclass
class PIIEntity:
    type: str
    value: str
    start: int
    end: int

@dataclass
class PIIResult:
    has_pii: bool
    entities: list = field(default_factory=list)
    masked_text: str = ""
    original_text: str = ""


class PIIDetector:
    """Detect and mask personally identifiable information using regex patterns."""

    PATTERNS = {
        "email": r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
        "phone": r'\b(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b',
        "ssn": r'\b\d{3}-\d{2}-\d{4}\b',
        "credit_card": r'\b(?:\d{4}[-\s]?){3}\d{4}\b',
        "ip_address": r'\b(?:25[0-5]|2[0-4]\d|[01]?\d\d?)\.(?:25[0-5]|2[0-4]\d|[01]?\d\d?)\.(?:25[0-5]|2[0-4]\d|[01]?\d\d?)\.(?:25[0-5]|2[0-4]\d|[01]?\d\d?)\b',
        "badge_id": r'\b(?:EMP|BADGE|ID)[-#]?\d{4,8}\b',
        "dob": r'\b(?:DOB|born|birthday)[:\s]*\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b',
        "api_key": r'\b(?:sk-|ghp_|AKIA)[A-Za-z0-9]{16,}\b',
    }

    def scan(self, text: str) -> PIIResult:
        """Scan text for PII entities and return masked version."""
        entities = []
        masked_text = text

        for pii_type, pattern in self.PATTERNS.items():
            for match in re.finditer(pattern, text, re.IGNORECASE):
                entities.append(PIIEntity(
                    type=pii_type,
                    value=match.group(),
                    start=match.start(),
                    end=match.end(),
                ))
                masked_text = masked_text.replace(match.group(), f"[{pii_type.upper()}]")

        return PIIResult(
            has_pii=len(entities) > 0,
            entities=entities,
            masked_text=masked_text,
            original_text=text,
        )
