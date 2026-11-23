"""Configuration for the accuracy-first BM25 baseline."""
from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


BASE_DIR = Path(__file__).resolve().parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=BASE_DIR / ".env", env_file_encoding="utf-8", extra="ignore"
    )

    llm_api_key: str = ""
    llm_base_url: str = "https://api.deepseek.com"
    llm_model: str = "deepseek-v4-flash"
    llm_temperature: float = 0
    llm_max_tokens: int = Field(default=0, ge=0)
    llm_timeout: int = Field(default=1800, ge=1)
    llm_max_retries: int = Field(default=2, ge=0, le=10)
    llm_enable_thinking: bool = True
    llm_reasoning_effort: str = Field(default="high", pattern="^(low|high|max)$")
    llm_thinking_rounds: int = Field(default=10, ge=0, le=10)
    llm_enable_query_planner: bool = True
    llm_planner_max_tokens: int = Field(default=0, ge=0)

    corpus_path: Path = Path("data/corpus")
    bm25_db_path: Path = Path("indexes/bm25_full.sqlite")
    retrieval_method: Literal["bm25"] = "bm25"
    retrieval_mode: Literal["bm25", "bm25_bge", "bm25_qwen", "bm25_bge_qwen"] = "bm25"
    qwen_index_path: Path = Path("indexes/experimental_qwen")
    qwen_model_path: Path = Path("models/qwen3-embedding-0.6b")
    qwen_device: str = "cpu"
    bge_index_path: Path = Path("indexes/experimental_bge")
    bge_model_path: Path = Path("models/bge-small-en-v1.5")
    bge_device: str = "cpu"
    hybrid_candidate_k: int = Field(default=50, ge=1, le=1000)
    hybrid_rrf_k: int = Field(default=60, ge=1)
    hybrid_bm25_weight: float = Field(default=1.0, gt=0)
    hybrid_bge_weight: float = Field(default=1.0, gt=0)
    retrieval_top_k: int = Field(default=8, ge=1, le=20)
    retrieval_candidate_k: int = Field(default=50, ge=1, le=1000)
    retrieval_rrf_k: int = Field(default=60, ge=1, le=1000)

    max_read_chars: int = Field(default=4000, ge=1)
    max_matches: int = Field(default=10, ge=1)
    max_context_chars: int = Field(default=500, ge=0)

    max_agent_steps: int = Field(default=10, ge=1)
    max_tool_calls: int = Field(default=15, ge=1)
    # A soft check before a request: an in-flight response may exceed this value.
    # Set 0 to disable the per-question check; step/call limits still apply.
    max_total_tokens_per_run: int = Field(default=95_000, ge=0)
    min_remaining_tokens_for_llm_call: int = Field(default=10_000, ge=1)
    max_stalled_searches: int = Field(default=3, ge=1, le=10)
    max_repeated_candidates: int = Field(default=2, ge=1, le=10)

    skills_path: Path = Path("skills")
    log_level: str = "INFO"

    def resolve_path(self, path: Path) -> Path:
        return path if path.is_absolute() else BASE_DIR / path


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
