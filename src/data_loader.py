"""Dataset loading, PMID normalization, corpus cleaning, and persistence."""

from __future__ import annotations

import ast
import json
import math
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from pydantic import BaseModel, Field

from src.config import Settings, get_settings

INVALID_TEXT_VALUES = {"", "nan", "none", "null", "n/a"}
NUMERIC_PMID_RE = re.compile(r"^\d+$")
WHITESPACE_RE = re.compile(r"\s+")


class Passage(BaseModel):
    """Canonical biomedical passage stored identically across BM25 and FAISS."""

    id: str
    text: str
    pubmed_url: Optional[str] = None


class BioASQQuestion(BaseModel):
    """Canonical BioASQ benchmark question with normalized relevant passage IDs."""

    id: str
    question: str
    answer: str
    relevant_passage_ids: List[str] = Field(default_factory=list)


def normalize_id(raw_id: Any) -> Optional[str]:
    """Normalize a raw dataset ID (int, float, numeric string, etc.) into a canonical string ID."""
    if raw_id is None:
        return None
    if isinstance(raw_id, bool):
        return None
    if isinstance(raw_id, int):
        return str(raw_id)
    if isinstance(raw_id, float):
        if math.isnan(raw_id) or math.isinf(raw_id):
            return None
        if raw_id.is_integer():
            return str(int(raw_id))
        return str(raw_id).strip()

    text = str(raw_id).strip()
    if text.lower() in INVALID_TEXT_VALUES:
        return None

    # Handle float-formatted integer strings like "12345678.0"
    if text.endswith(".0") and NUMERIC_PMID_RE.match(text[:-2]):
        return text[:-2]

    return text


def build_pubmed_url(passage_id: str) -> Optional[str]:
    """Build a canonical PubMed URL when the passage ID is a valid numeric PMID."""
    if passage_id and NUMERIC_PMID_RE.match(passage_id):
        return f"https://pubmed.ncbi.nlm.nih.gov/{passage_id}/"
    return None


def clean_passage_text(raw_text: Any) -> Optional[str]:
    """Clean passage text and return None if empty, whitespace-only, or NaN-like."""
    if raw_text is None:
        return None
    if isinstance(raw_text, float) and math.isnan(raw_text):
        return None

    text = WHITESPACE_RE.sub(" ", str(raw_text)).strip()
    if text.lower() in INVALID_TEXT_VALUES:
        return None
    return text


def parse_relevant_passage_ids(raw_value: Any) -> List[str]:
    """Safely parse stringified lists (or native lists/scalars) of relevant passage IDs."""
    if raw_value is None:
        return []

    parsed: Any = raw_value
    if isinstance(raw_value, str):
        stripped = raw_value.strip()
        if stripped.lower() in INVALID_TEXT_VALUES:
            return []
        try:
            parsed = ast.literal_eval(stripped)
        except (ValueError, SyntaxError):
            # Fallback: extract comma/whitespace separated tokens or digits
            parsed = [tok.strip(" []'\"") for tok in stripped.split(",") if tok.strip(" []'\"")]

    if not isinstance(parsed, (list, tuple, set)):
        parsed = [parsed]

    seen: set[str] = set()
    normalized_ids: List[str] = []
    for item in parsed:
        norm = normalize_id(item)
        if norm and norm not in seen:
            seen.add(norm)
            normalized_ids.append(norm)
    return normalized_ids


def clean_corpus_records(
    records: Iterable[Dict[str, Any]],
) -> Tuple[List[Passage], Dict[str, int]]:
    """Filter invalid/empty passages and deduplicate IDs while preserving deterministic order."""
    passages: List[Passage] = []
    seen_ids: set[str] = set()
    stats = {
        "total_raw": 0,
        "dropped_invalid_id": 0,
        "dropped_empty_text": 0,
        "dropped_duplicate_id": 0,
        "kept": 0,
    }

    for row in records:
        stats["total_raw"] += 1
        pid = normalize_id(row.get("id"))
        if not pid:
            stats["dropped_invalid_id"] += 1
            continue

        text = clean_passage_text(row.get("passage", row.get("text")))
        if not text:
            stats["dropped_empty_text"] += 1
            continue

        if pid in seen_ids:
            stats["dropped_duplicate_id"] += 1
            continue

        seen_ids.add(pid)
        passages.append(
            Passage(
                id=pid,
                text=text,
                pubmed_url=build_pubmed_url(pid),
            )
        )

    stats["kept"] = len(passages)
    return passages, stats


def clean_question_records(
    records: Iterable[Dict[str, Any]],
    valid_passage_ids: Optional[set[str]] = None,
) -> Tuple[List[BioASQQuestion], Dict[str, int]]:
    """Normalize BioASQ question records and optionally filter to those with >=1 indexed gold passage."""
    questions: List[BioASQQuestion] = []
    seen_ids: set[str] = set()
    stats = {
        "total_raw": 0,
        "dropped_invalid": 0,
        "dropped_no_indexed_gold": 0,
        "kept": 0,
    }

    for row in records:
        stats["total_raw"] += 1
        qid = normalize_id(row.get("id"))
        qtext = clean_passage_text(row.get("question"))
        atext = clean_passage_text(row.get("answer")) or ""
        rel_ids = parse_relevant_passage_ids(row.get("relevant_passage_ids"))

        if not qid or not qtext or qid in seen_ids:
            stats["dropped_invalid"] += 1
            continue

        if valid_passage_ids is not None:
            rel_ids = [pid for pid in rel_ids if pid in valid_passage_ids]
            if not rel_ids:
                stats["dropped_no_indexed_gold"] += 1
                continue

        seen_ids.add(qid)
        questions.append(
            BioASQQuestion(
                id=qid,
                question=qtext,
                answer=atext,
                relevant_passage_ids=rel_ids,
            )
        )

    stats["kept"] = len(questions)
    return questions, stats


def load_hf_corpus(settings: Optional[Settings] = None) -> Tuple[List[Passage], Dict[str, int]]:
    """Load and clean the `text-corpus` subset of `rag-datasets/rag-mini-bioasq`."""
    from datasets import load_dataset

    cfg = settings or get_settings()
    ds = load_dataset(
        cfg.hf_dataset_name,
        cfg.hf_corpus_config,
        split=cfg.hf_corpus_split,
    )
    return clean_corpus_records(ds)


def load_hf_questions(
    valid_passage_ids: Optional[set[str]] = None,
    settings: Optional[Settings] = None,
) -> Tuple[List[BioASQQuestion], Dict[str, int]]:
    """Load and normalize the `question-answer-passages` subset of `rag-datasets/rag-mini-bioasq`."""
    from datasets import load_dataset

    cfg = settings or get_settings()
    ds = load_dataset(
        cfg.hf_dataset_name,
        cfg.hf_qa_config,
        split=cfg.hf_qa_split,
    )
    return clean_question_records(ds, valid_passage_ids=valid_passage_ids)


def save_passages_jsonl(passages: Sequence[Passage], path: Path) -> None:
    """Persist canonical passages to JSONL (one JSON object per line)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for p in passages:
            f.write(p.model_dump_json() + "\n")


def load_passages_jsonl(path: Path) -> List[Passage]:
    """Load canonical passages from JSONL preserving exact indexed order."""
    if not path.exists():
        raise FileNotFoundError(f"Canonical passages file not found: {path}")

    passages: List[Passage] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            passages.append(Passage.model_validate_json(line))
    return passages


def load_passage_map(path: Path) -> Dict[str, Passage]:
    """Load canonical passages as an ID -> Passage dictionary."""
    return {p.id: p for p in load_passages_jsonl(path)}
