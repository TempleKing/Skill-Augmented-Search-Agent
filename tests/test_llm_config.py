from agent.llm import (
    configure_llm_for_required_tool,
    configure_llm_for_round,
    get_llm,
)
from config import settings


def test_deepseek_official_requests_disable_thinking(monkeypatch):
    monkeypatch.setattr(settings, "llm_api_key", "test-key")
    monkeypatch.setattr(settings, "llm_base_url", "https://api.deepseek.com")
    monkeypatch.setattr(settings, "llm_model", "deepseek-v4-flash")
    monkeypatch.setattr(settings, "llm_enable_thinking", False)
    get_llm.cache_clear()
    model = get_llm()

    assert model.extra_body == {"thinking": {"type": "disabled"}}
    get_llm.cache_clear()


def test_deepseek_official_can_enable_thinking(monkeypatch):
    monkeypatch.setattr(settings, "llm_api_key", "test-key")
    monkeypatch.setattr(settings, "llm_base_url", "https://api.deepseek.com")
    monkeypatch.setattr(settings, "llm_model", "deepseek-v4-flash")
    monkeypatch.setattr(settings, "llm_enable_thinking", True)
    monkeypatch.setattr(settings, "llm_reasoning_effort", "low")
    get_llm.cache_clear()
    model = get_llm()

    assert model.extra_body == {"thinking": {"type": "enabled"}}
    assert model.reasoning_effort == "low"
    get_llm.cache_clear()


def test_deepseek_thinking_can_be_limited_to_early_rounds(monkeypatch):
    monkeypatch.setattr(settings, "llm_api_key", "test-key")
    monkeypatch.setattr(settings, "llm_base_url", "https://api.deepseek.com")
    monkeypatch.setattr(settings, "llm_model", "deepseek-v4-flash")
    monkeypatch.setattr(settings, "llm_enable_thinking", True)
    monkeypatch.setattr(settings, "llm_reasoning_effort", "high")
    monkeypatch.setattr(settings, "llm_thinking_rounds", 1)
    get_llm.cache_clear()
    model = get_llm()

    first_round = configure_llm_for_round(model, 0)
    later_round = configure_llm_for_round(model, 1)

    assert first_round.extra_body == {"thinking": {"type": "enabled"}}
    assert first_round.reasoning_effort == "high"
    assert later_round.extra_body == {"thinking": {"type": "disabled"}}
    assert later_round.reasoning_effort is None
    get_llm.cache_clear()


def test_deepseek_required_tool_disables_thinking(monkeypatch):
    monkeypatch.setattr(settings, "llm_api_key", "test-key")
    monkeypatch.setattr(settings, "llm_base_url", "https://api.deepseek.com")
    monkeypatch.setattr(settings, "llm_model", "deepseek-v4-flash")
    monkeypatch.setattr(settings, "llm_enable_thinking", True)
    monkeypatch.setattr(settings, "llm_reasoning_effort", "high")
    get_llm.cache_clear()
    model = configure_llm_for_required_tool(get_llm())

    assert model.extra_body == {"thinking": {"type": "disabled"}}
    assert model.reasoning_effort is None
    get_llm.cache_clear()
