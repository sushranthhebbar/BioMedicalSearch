"""Centralized configuration for Biomedical Hybrid Search & RAG."""

from __future__ import annotations

from enum import Enum
from functools import lru_cache
from pathlib import Path
from typing import Dict

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


PROJECT_ROOT = Path(__file__).resolve().parent.parent


class RetrievalMode(str, Enum):
    """Supported retrieval configurations matching the 5 evaluation configs."""

    LEXICAL = "lexical"
    DENSE = "dense"
    HYBRID = "hybrid"
    HYBRID_QE = "hybrid_qe"
    HYBRID_QE_RERANK = "hybrid_qe_rerank"


# Standard Gemini 2.5 Flash pricing (USD per 1M tokens) for reproducible cost estimation.
# Free tier has $0 actual spend, so we report estimated equivalent API cost for trade-off analysis.
MODEL_PRICING_PER_1M_TOKENS: Dict[str, Dict[str, float]] = {
    "gemini-2.5-flash": {"input": 0.15, "output": 0.60},
    "gemini-2.5-flash-lite": {"input": 0.075, "output": 0.30},
    "gemini-2.0-flash": {"input": 0.10, "output": 0.40},
}


def estimate_llm_cost_usd(
    model_name: str,
    prompt_tokens: int,
    completion_tokens: int,
) -> float:
    """Estimate USD cost for a Gemini call from token counts."""
    pricing = MODEL_PRICING_PER_1M_TOKENS.get(
        model_name,
        MODEL_PRICING_PER_1M_TOKENS["gemini-2.5-flash"],
    )
    cost = (prompt_tokens / 1_000_000.0) * pricing["input"] + (
        completion_tokens / 1_000_000.0
    ) * pricing["output"]
    return round(cost, 8)


class Settings(BaseSettings):
    """Application settings loaded from environment variables and `.env`."""

    model_config = SettingsConfigDict(
        env_file=str(PROJECT_ROOT / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # API credentials
    gemini_api_key: str = Field(default="", alias="GEMINI_API_KEY")

    # Dataset
    hf_dataset_name: str = Field(default="rag-datasets/rag-mini-bioasq")
    hf_corpus_config: str = Field(default="text-corpus")
    hf_corpus_split: str = Field(default="passages")
    hf_qa_config: str = Field(default="question-answer-passages")
    hf_qa_split: str = Field(default="test")

    # Models
    embedding_model: str = Field(
        default="BAAI/bge-small-en-v1.5", alias="EMBEDDING_MODEL"
    )
    reranker_model: str = Field(
        default="ms-marco-MiniLM-L-12-v2", alias="RERANKER_MODEL"
    )
    gemini_generation_model: str = Field(
        default="gemini-2.5-flash", alias="GEMINI_GENERATION_MODEL"
    )
    gemini_expansion_model: str = Field(
        default="gemini-2.5-flash", alias="GEMINI_EXPANSION_MODEL"
    )
    gemini_judge_model: str = Field(
        default="gemini-2.5-flash", alias="GEMINI_JUDGE_MODEL"
    )

    # Rate-limit pacing (0 disables pacing)
    gemini_rpm_limit: int = Field(default=10, alias="GEMINI_RPM_LIMIT")

    # Paths
    data_dir: Path = Field(default=PROJECT_ROOT / "data")
    eval_dir: Path = Field(default=PROJECT_ROOT / "eval")

    # Retrieval defaults
    default_retrieval_mode: RetrievalMode = Field(
        default=RetrievalMode.HYBRID_QE_RERANK
    )
    default_top_k: int = Field(default=10)
    default_answer_context_k: int = Field(default=5)
    candidate_pool_k: int = Field(default=50)
    rerank_candidate_k: int = Field(default=30)

    # RRF & Query Expansion parameters
    rrf_k: int = Field(default=60)
    rrf_lexical_weight: float = Field(default=1.0)
    rrf_dense_weight: float = Field(default=1.0)
    qe_num_expansions: int = Field(default=3)
    qe_original_query_weight: float = Field(default=2.0)
    qe_expanded_query_weight: float = Field(default=1.0)
    qe_temperature: float = Field(default=0.2)

    # Embedding indexing batch / checkpointing
    embedding_batch_size: int = Field(default=128)
    embedding_checkpoint_interval: int = Field(default=2048)

    # Minimum relevance score thresholds per mode for insufficient-evidence gating (Phase 2)
    min_score_thresholds: Dict[str, float] = Field(
        default_factory=lambda: {
            RetrievalMode.LEXICAL.value: 2.0,
            RetrievalMode.DENSE.value: 0.45,
            RetrievalMode.HYBRID.value: 0.012,
            RetrievalMode.HYBRID_QE.value: 0.018,
            RetrievalMode.HYBRID_QE_RERANK.value: 0.15,
        }
    )

    # API server
    api_host: str = Field(default="127.0.0.1", alias="API_HOST")
    api_port: int = Field(default=8000, alias="API_PORT")

    @property
    def indexes_dir(self) -> Path:
        return self.data_dir / "indexes"

    @property
    def bm25_dir(self) -> Path:
        return self.indexes_dir / "bm25"

    @property
    def faiss_dir(self) -> Path:
        return self.indexes_dir / "faiss"

    @property
    def passages_path(self) -> Path:
        return self.indexes_dir / "passages.jsonl"

    @property
    def cache_dir(self) -> Path:
        return self.data_dir / "cache"

    @property
    def logs_dir(self) -> Path:
        return self.data_dir / "logs"

    @property
    def search_log_path(self) -> Path:
        return self.logs_dir / "searches.jsonl"

    @property
    def embeddings_checkpoint_path(self) -> Path:
        return self.cache_dir / "embeddings_checkpoint.npz"

    @property
    def query_expansion_cache_path(self) -> Path:
        return self.cache_dir / "query_expansions.json"

    @property
    def fastembed_cache_dir(self) -> Path:
        return self.cache_dir / "fastembed"

    @property
    def flashrank_cache_dir(self) -> Path:
        return self.cache_dir / "flashrank"

    def ensure_directories(self) -> None:
        """Create required local data directories if they do not exist."""
        for path in (
            self.indexes_dir,
            self.bm25_dir,
            self.faiss_dir,
            self.cache_dir,
            self.logs_dir,
            self.fastembed_cache_dir,
            self.flashrank_cache_dir,
        ):
            path.mkdir(parents=True, exist_ok=True)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return a cached singleton Settings instance."""
    return Settings()
