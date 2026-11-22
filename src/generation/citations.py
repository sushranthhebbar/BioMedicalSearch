"""Deterministic inline citation extraction, validation, and sanitization."""

from __future__ import annotations

import re
from typing import Iterable, List, Optional, Sequence, Tuple

from pydantic import BaseModel, Field

# Match bracketed groups `[...]` that are NOT markdown links `[label](url)`.
_BRACKET_RE = re.compile(r"\[([^\[\]]+)\](?!\()")

# Individual item inside a citation bracket: optional prefix + ID token.
_CITATION_ITEM_RE = re.compile(
    r"^(?:(?P<prefix>Passage\s*ID|Passage|PMID|ID)\s*:?\s*)?(?P<id>[A-Za-z0-9_-]+)$",
    re.IGNORECASE,
)

# Cleanup regexes after stripping invalid citation brackets.
_SPACE_BEFORE_PUNCT_RE = re.compile(r"\s+([.,;:!?])")
_MULTI_SPACE_RE = re.compile(r"[ \t]{2,}")


class CitationValidationResult(BaseModel):
    """Structured audit result from deterministic inline citation validation."""

    sanitized_answer: str
    allowed_ids: List[str] = Field(default_factory=list)
    valid_citations: List[str] = Field(default_factory=list)
    invalid_citations: List[str] = Field(default_factory=list)
    total_citation_occurrences: int = 0
    valid_citation_occurrences: int = 0
    invalid_citation_occurrences: int = 0
    citation_validity_rate: float = 1.0
    zero_hallucinated_ids: bool = True
    missing_citations: bool = False
    is_valid: bool = True


def _parse_citation_bracket(
    inner: str,
    allowed_set: set[str],
) -> Optional[List[str]]:
    """Parse a bracket's inner text into citation IDs if it is a citation block.

    Returns `None` when the bracket is ordinary biomedical prose (e.g. `[Ca2+]`,
    `[18F]`, `[sic]`, `[p < 0.05]`) rather than a citation block.
    """
    parts = [p.strip() for p in re.split(r"[,;]", inner) if p.strip()]
    if not parts:
        return None

    parsed_ids: List[str] = []
    for part in parts:
        match = _CITATION_ITEM_RE.match(part)
        if not match:
            return None
        has_prefix = bool(match.group("prefix"))
        token = match.group("id").strip()

        # Treat as a citation token if it matches an allowed passage ID, carries an
        # explicit Passage/PMID/ID prefix, or is a bare integer (catches both
        # hallucinated PMIDs like `[99999999]` and 1-based refs like `[1]`).
        if token in allowed_set or has_prefix or token.isdigit():
            parsed_ids.append(token)
        else:
            return None

    return parsed_ids


def extract_citation_ids(
    text: str,
    allowed_ids: Optional[Iterable[str]] = None,
) -> List[str]:
    """Deterministically extract all inline citation IDs from text in order of appearance."""
    if not text:
        return []
    allowed_set = {str(pid).strip() for pid in (allowed_ids or []) if str(pid).strip()}
    extracted: List[str] = []
    for match in _BRACKET_RE.finditer(text):
        ids = _parse_citation_bracket(match.group(1), allowed_set)
        if ids:
            extracted.extend(ids)
    return extracted


def _normalize_whitespace(text: str) -> str:
    """Clean up minor spacing artifacts introduced when invalid brackets are removed."""
    lines = text.splitlines()
    cleaned_lines: List[str] = []
    for line in lines:
        s = _MULTI_SPACE_RE.sub(" ", line)
        s = _SPACE_BEFORE_PUNCT_RE.sub(r"\1", s)
        cleaned_lines.append(s.strip())
    return "\n".join(cleaned_lines).strip()


def validate_citations(
    answer_text: str,
    allowed_ids: Sequence[str],
    sufficient_evidence: bool = True,
) -> CitationValidationResult:
    """Validate every inline `[ID]` citation against `allowed_ids` and strip invalid IDs.

    Rules enforced deterministically in Python:
      1. Every inline citation ID must belong to `allowed_ids` (the selected context
         passages passed to the LLM for this query).
      2. Multi-ID brackets like `[12345, 67890]` or `[PMID: 12345]` are normalized
         to canonical `[12345] [67890]` tokens for consistent UI rendering.
      3. Any invalid/hallucinated citation ID is recorded in `invalid_citations` and
         stripped from `sanitized_answer`.
      4. A non-abstaining answer (`sufficient_evidence=True`) with zero valid inline
         citations is flagged with `missing_citations=True`, `citation_validity_rate=0.0`,
         and `is_valid=False`.
    """
    ordered_allowed: List[str] = []
    allowed_set: set[str] = set()
    for pid in allowed_ids:
        norm = str(pid).strip()
        if norm and norm not in allowed_set:
            allowed_set.add(norm)
            ordered_allowed.append(norm)

    raw_text = (answer_text or "").strip()
    valid_unique: List[str] = []
    valid_seen: set[str] = set()
    invalid_unique: List[str] = []
    invalid_seen: set[str] = set()

    total_occ = 0
    valid_occ = 0
    invalid_occ = 0

    def _replace_bracket(match: re.Match[str]) -> str:
        nonlocal total_occ, valid_occ, invalid_occ
        inner = match.group(1)
        ids = _parse_citation_bracket(inner, allowed_set)
        if not ids:
            return match.group(0)

        kept_tokens: List[str] = []
        for cid in ids:
            total_occ += 1
            if cid in allowed_set:
                valid_occ += 1
                if cid not in valid_seen:
                    valid_seen.add(cid)
                    valid_unique.append(cid)
                kept_tokens.append(f"[{cid}]")
            else:
                invalid_occ += 1
                if cid not in invalid_seen:
                    invalid_seen.add(cid)
                    invalid_unique.append(cid)

        return " ".join(kept_tokens)

    sanitized = _BRACKET_RE.sub(_replace_bracket, raw_text)
    if invalid_occ > 0:
        sanitized = _normalize_whitespace(sanitized)

    zero_hallucinated = invalid_occ == 0
    missing_citations = sufficient_evidence and valid_occ == 0

    if total_occ > 0:
        validity_rate = round(valid_occ / float(total_occ), 4)
    elif sufficient_evidence:
        validity_rate = 0.0
    else:
        # Abstaining answer with no hallucinated citations is citation-compliant.
        validity_rate = 1.0

    is_valid = zero_hallucinated and not missing_citations

    return CitationValidationResult(
        sanitized_answer=sanitized,
        allowed_ids=ordered_allowed,
        valid_citations=valid_unique,
        invalid_citations=invalid_unique,
        total_citation_occurrences=total_occ,
        valid_citation_occurrences=valid_occ,
        invalid_citation_occurrences=invalid_occ,
        citation_validity_rate=validity_rate,
        zero_hallucinated_ids=zero_hallucinated,
        missing_citations=missing_citations,
        is_valid=is_valid,
    )
