from langchain_core.messages import AIMessage

from agent import query_planner
from agent.query_planner import normalize_planned_queries


def test_normalize_planned_queries_deduplicates_and_bounds_shape():
    queries = normalize_planned_queries([
        "Olympic scientist plant hormone",
        "Olympic scientist plant hormone",
        "National Medal Science regulatory molecule",
        "too short",
        "one two three four five six seven eight nine ten eleven twelve thirteen",
    ])
    assert queries == [
        "Olympic scientist plant hormone",
        "National Medal Science regulatory molecule",
    ]


def test_deepseek_planner_disables_thinking_without_a_token_cap(monkeypatch):
    class Model:
        def __init__(self, options=None):
            self.options = options or {}

        def model_copy(self, *, update):
            return Model({**self.options, **update})

        def invoke(self, messages):
            assert self.options == {
                "extra_body": {"thinking": {"type": "disabled"}},
                "reasoning_effort": None,
                "max_tokens": None,
            }
            return AIMessage(content='{"queries": ["test query terms"]}')

    monkeypatch.setattr(query_planner.settings, "llm_base_url", "https://api.deepseek.com")
    monkeypatch.setattr(query_planner.settings, "llm_enable_query_planner", True)
    monkeypatch.setattr(query_planner.settings, "llm_planner_max_tokens", 0)
    monkeypatch.setattr(query_planner, "get_llm", Model)

    queries, diagnostics = query_planner.plan_queries("Which entity matches the clues?")
    assert queries == ["test query terms"]
    assert diagnostics["status"] == "success"
