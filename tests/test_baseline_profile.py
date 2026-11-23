from pathlib import Path

import pytest
from pydantic import ValidationError

from config import BASE_DIR, Settings


def test_example_and_defaults_agree_on_published_profile():
    defaults = Settings(_env_file=None)
    example = Settings(_env_file=BASE_DIR / ".env.example")
    for name in (
        "retrieval_method", "retrieval_top_k", "retrieval_candidate_k", "retrieval_rrf_k",
        "llm_model", "llm_base_url", "llm_temperature", "llm_enable_thinking",
        "llm_reasoning_effort", "llm_thinking_rounds", "llm_enable_query_planner",
        "max_agent_steps", "max_tool_calls", "max_total_tokens_per_run",
        "max_stalled_searches",
    ):
        assert getattr(defaults, name) == getattr(example, name), name


def test_non_baseline_retrieval_is_rejected():
    with pytest.raises(ValidationError):
        Settings(_env_file=None, retrieval_method="unsupported-backend")


def test_relative_paths_resolve_from_the_project_root():
    settings = Settings(_env_file=None)
    assert settings.resolve_path(Path("indexes/example.sqlite")) == BASE_DIR / "indexes/example.sqlite"
