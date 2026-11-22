"""Generate a fixed, stratified 100-query biomedical evaluation benchmark from rag-mini-bioasq.

Filters test questions to those with >=1 valid `relevant_passage_id` in the indexed corpus,
classifies questions into the 6 required categories (`yes_no`, `definition`, `list`,
`treatment`, `mechanism`, `effect`), and samples deterministically with `seed=42`.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import Settings, get_settings
from src.data_loader import BioASQQuestion, load_hf_questions, load_passages_jsonl
from src.indexing import compute_id_fingerprint

# Target balanced counts across the 6 assignment categories (total = 100)
CATEGORY_QUOTAS: Dict[str, int] = {
    "yes_no": 18,
    "definition": 16,
    "list": 16,
    "treatment": 17,
    "mechanism": 17,
    "effect": 16,
}

# ---------------------------------------------------------------------------
# Deterministic linguistic & biomedical category heuristics
# ---------------------------------------------------------------------------

_YES_NO_START_RE = re.compile(
    r"^(is|are|does|do|can|could|has|have|was|were|did|will|should)\b", re.IGNORECASE
)
_YES_NO_ANS_RE = re.compile(r"^(yes|no)\b", re.IGNORECASE)

_EXPLICIT_DEF_RE = re.compile(
    r"^(?:"
    r"define\b|"
    r"what\s+does\s+(?:the\s+)?(?:acronym\s+|term\s+|abbreviation\s+)?[\w\"'\-]+\s+stand\s+for\b|"
    r"what\s+is\s+the\s+(?:definition|meaning)\s+of\b"
    r")",
    re.IGNORECASE,
)

_SIMPLE_WHAT_IS_RE = re.compile(
    r"^what\s+(?:is|are)\s+(?:a\s+|an\s+|the\s+)?[A-Za-z0-9][A-Za-z0-9\-\(\)'\"/\s]{1,65}\??$",
    re.IGNORECASE,
)
_NON_DEF_PREPOSITIONS_RE = re.compile(
    r"\b(of|in|for|between|during|after|before|caused|used|associated|found|located|encoded|"
    r"involved|required|responsible|known|called|named|considered|reported|linked|related|"
    r"compared|versus|vs|difference|relationship|effect|effects|mechanism|role|function|"
    r"treatment|therapy|symptom|symptoms|cause|causes|type|types|subtype|subtypes|"
    r"gene|genes|protein|proteins|drug|drugs|mutation|mutations|biomarker|biomarkers|"
    r"main|major|most|common|primary|first|best|standard|normal|average|incidence|prevalence)\b",
    re.IGNORECASE,
)

_LIST_RE = re.compile(
    r"^(?:"
    r"list\b|"
    r"name\s+(?:the|all|some|two|three|four|five|six|\d+)\b|"
    r"which\s+(?:genes|proteins|diseases|disorders|mutations|mirnas|enzymes|receptors|"
    r"symptoms|pathways|biomarkers|markers|antibodies|compounds|factors|cells|viruses|"
    r"bacteria|syndromes|tumors|cancers|conditions|ligands|isoforms|subunits|chromosomes|"
    r"tissues|organs|techniques|methods|tools|databases)\b|"
    r"what\s+are\s+the\s+(?:main\s+|major\s+|common\s+|primary\s+|most\s+common\s+|key\s+)?"
    r"(?:symptoms|signs|causes|types|subtypes|genes|proteins|features|components|hallmarks|"
    r"stages|phases|classes|forms|variants|biomarkers|markers|risk\s+factors|"
    r"clinical\s+features|characteristics|manifestations|indications|criteria)\b"
    r")",
    re.IGNORECASE,
)

_TREATMENT_RE = re.compile(
    r"\b("
    r"treat|treated|treating|treatment|treatments|therapy|therapies|therapeutic|"
    r"drug|drugs|medication|medications|used\s+for\s+(?:the\s+)?treatment|"
    r"used\s+to\s+treat|first[- ]line|approved\s+(?:by\s+(?:the\s+)?fda\s+)?for|"
    r"indicated\s+for|chemotherapy|immunotherapy|radiotherapy|antibiotic|antibiotics|"
    r"vaccine|vaccines|antiviral|prophylaxis|pharmacotherapy|clinical\s+management|"
    r"cure|regimen|standard\s+of\s+care"
    r")\b",
    re.IGNORECASE,
)

_EFFECT_RE = re.compile(
    r"\b("
    r"effect|effects|side\s+effect|side\s+effects|adverse\s+effect|adverse\s+effects|"
    r"adverse\s+event|adverse\s+events|impact\s+of|affect|affects|affecting|"
    r"consequence|consequences|influence\s+of|efficacy|effectiveness|toxicity|"
    r"outcome|outcomes"
    r")\b",
    re.IGNORECASE,
)

_MECHANISM_RE = re.compile(
    r"\b("
    r"mechanism|mechanisms|mode\s+of\s+action|how\s+does|how\s+do|"
    r"signaling\s+pathway|pathway|pathogenesis|molecular\s+basis|molecular\s+mechanism|"
    r"how\s+is\s+.+\s+regulated|biological\s+function|role\s+of\s+.+\s+in|function\s+of"
    r")\b",
    re.IGNORECASE,
)


def classify_question(question: str, answer: str = "") -> str:
    """Deterministically classify a BioASQ question into one of the 6 target categories or `other`."""
    q = " ".join((question or "").strip().split())
    a = " ".join((answer or "").strip().split())
    if not q:
        return "other"

    ql = q.lower()
    starts_aux = bool(_YES_NO_START_RE.search(q))

    # 1. Pure yes/no questions (avoid "X or Y?" choice questions)
    if starts_aux and _YES_NO_ANS_RE.search(a) and " or " not in ql:
        return "yes_no"

    # 2. Strict definition / acronym questions
    if _EXPLICIT_DEF_RE.search(q):
        return "definition"
    if (
        _SIMPLE_WHAT_IS_RE.match(q)
        and len(q.split()) <= 8
        and not _NON_DEF_PREPOSITIONS_RE.search(q)
    ):
        return "definition"

    # 3. Effect / outcome / toxicity / adverse-event questions (non yes/no)
    if not starts_aux and _EFFECT_RE.search(q):
        return "effect"

    # 4. Treatment / clinical management / therapy questions (non yes/no)
    if not starts_aux and _TREATMENT_RE.search(q):
        return "treatment"

    # 5. Biological / molecular mechanism questions (non yes/no)
    if not starts_aux and _MECHANISM_RE.search(q):
        return "mechanism"

    # 6. List / enumeration questions
    if _LIST_RE.search(q):
        return "list"

    return "other"


def _canonical_sort_key(q: BioASQQuestion) -> Tuple[int, str]:
    """Stable numeric-first sort key independent of hash randomization."""
    if q.id.isdigit():
        return (int(q.id), q.id)
    return (10**12, q.id)


def create_stratified_eval_set(
    questions: Sequence[BioASQQuestion],
    quotas: Dict[str, int] = CATEGORY_QUOTAS,
    seed: int = 42,
) -> Tuple[List[Dict[str, object]], Dict[str, int]]:
    """Deterministically sample a balanced 100-query benchmark set across the 6 categories."""
    buckets: Dict[str, List[BioASQQuestion]] = {cat: [] for cat in quotas}
    pool_counts: Dict[str, int] = Counter()

    for q in sorted(questions, key=_canonical_sort_key):
        if not q.question.strip() or not q.answer.strip() or not q.relevant_passage_ids:
            continue
        cat = classify_question(q.question, q.answer)
        pool_counts[cat] += 1
        if cat in buckets:
            buckets[cat].append(q)

    rng = random.Random(seed)
    selected: List[Dict[str, object]] = []
    seen_ids: set[str] = set()

    for cat, target in quotas.items():
        candidates = buckets[cat]
        if len(candidates) < target:
            raise ValueError(
                f"Insufficient candidates for category '{cat}': found {len(candidates)}, need {target}."
            )

        if cat == "yes_no":
            # Keep a natural mix of both 'Yes' and 'No' gold answers when possible
            yes_cands = [q for q in candidates if q.answer.strip().lower().startswith("yes")]
            no_cands = [q for q in candidates if q.answer.strip().lower().startswith("no")]
            n_no = min(len(no_cands), max(1, target // 3))
            n_yes = target - n_no
            if len(yes_cands) < n_yes:
                n_yes = len(yes_cands)
                n_no = target - n_yes
            picked = rng.sample(yes_cands, n_yes) + rng.sample(no_cands, n_no)
            rng.shuffle(picked)
        else:
            picked = rng.sample(candidates, target)

        for q in sorted(picked, key=_canonical_sort_key):
            if q.id in seen_ids:
                raise RuntimeError(f"Duplicate query ID sampled: {q.id}")
            seen_ids.add(q.id)
            selected.append(
                {
                    "id": q.id,
                    "category": cat,
                    "question": q.question,
                    "answer": q.answer,
                    "relevant_passage_ids": list(q.relevant_passage_ids),
                }
            )

    # Stable overall ordering by category order then canonical numeric query ID
    return selected, dict(pool_counts)


def _load_questions_resilient(
    valid_passage_ids: set[str],
    settings: Settings,
    offline: bool = False,
) -> Tuple[List[BioASQQuestion], Dict[str, int]]:
    """Load BioASQ questions from Hugging Face, falling back cleanly to local HF cache if offline."""
    if offline:
        os.environ["HF_HUB_OFFLINE"] = "1"
        return load_hf_questions(valid_passage_ids=valid_passage_ids, settings=settings)

    try:
        return load_hf_questions(valid_passage_ids=valid_passage_ids, settings=settings)
    except Exception:
        os.environ["HF_HUB_OFFLINE"] = "1"
        return load_hf_questions(valid_passage_ids=valid_passage_ids, settings=settings)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create fixed, stratified 100-query benchmark set (eval/eval_queries_100.json)."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output JSON path (default: eval/eval_queries_100.json).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for deterministic stratified sampling (default: 42).",
    )
    parser.add_argument(
        "--offline",
        action="store_true",
        help="Force Hugging Face offline cache mode (HF_HUB_OFFLINE=1).",
    )
    parser.add_argument(
        "--print-all",
        action="store_true",
        help="Print all 100 selected queries grouped by category for manual spot-check.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    settings = get_settings()
    output_path = args.output or (settings.eval_dir / "eval_queries_100.json")

    passages = load_passages_jsonl(settings.passages_path)
    passage_ids = [p.id for p in passages]
    valid_pid_set = set(passage_ids)
    fingerprint = compute_id_fingerprint(passage_ids)

    questions, q_stats = _load_questions_resilient(
        valid_passage_ids=valid_pid_set,
        settings=settings,
        offline=args.offline,
    )

    selected, pool_counts = create_stratified_eval_set(
        questions=questions,
        quotas=CATEGORY_QUOTAS,
        seed=args.seed,
    )

    # Final integrity assertions
    assert len(selected) == sum(CATEGORY_QUOTAS.values()) == 100
    assert len({item["id"] for item in selected}) == 100
    for item in selected:
        rel_ids = item["relevant_passage_ids"]
        assert isinstance(rel_ids, list) and len(rel_ids) >= 1
        assert all(pid in valid_pid_set for pid in rel_ids)

    category_counts = dict(Counter(str(item["category"]) for item in selected))
    yes_no_breakdown = {
        "yes": sum(
            1
            for item in selected
            if item["category"] == "yes_no"
            and str(item["answer"]).strip().lower().startswith("yes")
        ),
        "no": sum(
            1
            for item in selected
            if item["category"] == "yes_no"
            and str(item["answer"]).strip().lower().startswith("no")
        ),
    }

    payload = {
        "metadata": {
            "dataset": settings.hf_dataset_name,
            "qa_config": settings.hf_qa_config,
            "qa_split": settings.hf_qa_split,
            "seed": args.seed,
            "total_queries": len(selected),
            "corpus_size": len(passages),
            "corpus_id_fingerprint": fingerprint,
            "question_filter_stats": q_stats,
            "category_pool_counts": pool_counts,
            "category_counts": category_counts,
            "yes_no_answer_balance": yes_no_breakdown,
        },
        "queries": selected,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    print(f"Saved {len(selected)} fixed evaluation queries to {output_path}")
    print(f"Corpus size: {len(passages)} | Fingerprint: {fingerprint[:16]}...")
    print(f"Category pool counts: {pool_counts}")
    print(f"Sampled category counts: {category_counts} (yes_no balance: {yes_no_breakdown})")

    if args.print_all:
        current_cat: Optional[str] = None
        for idx, item in enumerate(selected, start=1):
            cat = str(item["category"])
            if cat != current_cat:
                current_cat = cat
                print(f"\n--- {cat.upper()} ({category_counts[cat]}) ---")
            print(
                f"  {idx:3d}. [id={item['id']}] (gold={len(item['relevant_passage_ids'])}) "
                f"{item['question']}"
            )


if __name__ == "__main__":
    main()
