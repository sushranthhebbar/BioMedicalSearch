"""Grounded answer generation and deterministic inline citation enforcement."""

from src.generation.answerer import (
    GenerationResult,
    GroundedAnswerer,
    format_passages_for_prompt,
)
from src.generation.citations import (
    CitationValidationResult,
    extract_citation_ids,
    validate_citations,
)

__all__ = [
    "CitationValidationResult",
    "GenerationResult",
    "GroundedAnswerer",
    "extract_citation_ids",
    "format_passages_for_prompt",
    "validate_citations",
]
