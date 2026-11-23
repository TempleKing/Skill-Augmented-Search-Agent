from langchain_core.messages import AIMessage, HumanMessage

import agent.graph as graph_module
from agent.answer_validator import extract_required_facts


def make_state(**updates):
    state = {
        "question": "What is the answer?",
        "messages": [HumanMessage(content="What is the answer?")],
        "search_round": 0,
        "tool_calls": 0,
        "total_tokens": 0,
        "token_usage_complete": True,
        "active_queries": [],
        "visited_docs": [],
        "evidence": [],
        "trajectory": [],
        "final_answer": "",
        "selected_skills": [],
        "pending_skill": None,
        "active_skill": None,
        "skill_instructions": "",
        "skill_reason": "",
        "empty_response_count": 0,
        "termination_reason": "",
        "force_final_answer": False,
    }
    state.update(updates)
    return state


class FakeModel:
    def __init__(self, response):
        self.response = response

    def bind_tools(self, _tools):
        return self

    def invoke(self, _messages):
        return self.response


class SequenceModel:
    def __init__(self, responses):
        self.responses = iter(responses)

    def bind_tools(self, _tools):
        return self

    def invoke(self, _messages):
        return next(self.responses)


def test_extract_response_text_handles_provider_block_content():
    content = [
        {"type": "text", "text": " Queen Arwa University "},
        {"type": "metadata", "value": "ignored"},
    ]

    assert graph_module.extract_response_text(content) == "Queen Arwa University"
    assert graph_module.extract_response_text([]) == ""


def test_first_empty_response_is_retried(monkeypatch):
    response = AIMessage(content=[])
    monkeypatch.setattr(
        graph_module,
        "get_llm",
        lambda: FakeModel(response),
    )

    update = graph_module.agent_node(make_state())
    routed_state = make_state(
        messages=[response],
        empty_response_count=update["empty_response_count"],
    )

    assert update["final_answer"] == ""
    assert update["termination_reason"] == "retrying_empty_response"
    assert graph_module.route_after_agent(routed_state) == "retry_empty"


def test_repeated_empty_response_falls_back_to_unknown(monkeypatch):
    response = AIMessage(content="")
    monkeypatch.setattr(
        graph_module,
        "get_llm",
        lambda: FakeModel(response),
    )

    update = graph_module.agent_node(
        make_state(empty_response_count=1)
    )
    routed_state = make_state(
        messages=[response],
        empty_response_count=update["empty_response_count"],
    )

    assert update["final_answer"] == "Unknown"
    assert update["termination_reason"] == "llm_empty_response"
    assert graph_module.route_after_agent(routed_state) == "validate_answer"


def test_non_empty_response_completes_normally(monkeypatch):
    response = AIMessage(content="1996")
    monkeypatch.setattr(
        graph_module,
        "get_llm",
        lambda: FakeModel(response),
    )

    update = graph_module.agent_node(make_state())

    assert update["final_answer"] == "1996"
    assert update["empty_response_count"] == 0
    assert update["termination_reason"] == "completed"


def test_token_budget_stops_before_another_agent_round(monkeypatch):
    response = AIMessage(
        content="",
        tool_calls=[{"id": "pending-search", "name": "search_documents", "args": {"query": "candidate"}}],
        usage_metadata={
            "input_tokens": 4,
            "output_tokens": 2,
            "total_tokens": 6,
        },
    )
    monkeypatch.setattr(graph_module, "get_llm", lambda: FakeModel(response))
    monkeypatch.setattr(
        graph_module.settings,
        "max_total_tokens_per_run",
        5,
    )

    update = graph_module.agent_node(make_state(tool_calls=3))
    routed_state = make_state(
        messages=[response],
        termination_reason=update["termination_reason"],
    )

    assert update["final_answer"] == "Unknown"
    assert update["termination_reason"] == "token_budget_exhausted"
    assert update["tool_calls"] == 4
    assert update["trajectory"][-3]["type"] == "tool_call"
    blocked = update["trajectory"][-2]
    assert blocked["type"] == "tool_result"
    assert blocked["tool_call_id"] == "pending-search"
    assert blocked["executed"] is False
    assert update["trajectory"][-1]["type"] == "budget_stop"
    assert graph_module.route_after_agent(routed_state) == graph_module.END


def test_low_remaining_budget_stops_before_calling_model(monkeypatch):
    def fail_if_called():
        raise AssertionError("LLM must not be created when reserve is too low")

    monkeypatch.setattr(graph_module, "get_llm", fail_if_called)
    monkeypatch.setattr(
        graph_module.settings,
        "max_total_tokens_per_run",
        100,
    )
    state = make_state(
        search_round=2,
        total_tokens=99,
    )
    update = graph_module.agent_node(state)

    assert update["termination_reason"] == "token_budget_exhausted"
    assert update["trajectory"][-1]["phase"] == "before_llm_call"


def test_zero_token_budget_disables_token_stops(monkeypatch):
    response = AIMessage(
        content="supported answer",
        usage_metadata={
            "input_tokens": 4,
            "output_tokens": 2,
            "total_tokens": 6,
        },
    )
    monkeypatch.setattr(graph_module, "get_llm", lambda: FakeModel(response))
    monkeypatch.setattr(
        graph_module.settings,
        "max_total_tokens_per_run",
        0,
    )

    update = graph_module.agent_node(make_state(search_round=2, total_tokens=999_999))

    assert update["final_answer"] == "supported answer"
    assert update["termination_reason"] == "completed"


def test_graph_retries_once_then_returns_unknown(monkeypatch):
    model = SequenceModel([
        AIMessage(content=[]),
        AIMessage(content=""),
    ])
    monkeypatch.setattr(graph_module, "get_llm", lambda: model)

    result = graph_module.run_agent("Give me an unsupported fact")

    assert result["search_round"] == 2
    assert result["final_answer"] == "Unknown"
    assert result["termination_reason"] == "llm_empty_response"
