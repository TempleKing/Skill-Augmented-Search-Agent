from types import SimpleNamespace

from agent.question_analysis import (
    build_bootstrap_queries,
    build_constraint_queries,
    extract_search_constraints,
    is_multi_hop_question,
)
from agent import graph as graph_module


LONG_CLUE_QUESTION = (
    "A scientist was born under the zodiac sign Cancer. "
    "They earned a bachelor's degree and a PhD in different fields. "
    "In the same year they participated in a renowned competition. "
    "They discovered a regulatory molecule and received a national award. "
    "What is the full name of said scientist?"
)


def test_long_clue_question_is_multi_hop_and_excludes_final_request():
    constraints = extract_search_constraints(LONG_CLUE_QUESTION)
    assert len(constraints) == 4
    assert all("full name" not in item.lower() for item in constraints)
    assert is_multi_hop_question(LONG_CLUE_QUESTION) is True


def test_bootstrap_queries_are_complementary_and_compact():
    queries = build_bootstrap_queries(LONG_CLUE_QUESTION)
    assert len(queries) == 2
    assert queries[0] != queries[1]
    assert all(len(query.split()) <= 18 for query in queries)
    assert "Cancer" in queries[0]
    assert "regulatory" in queries[1]


def test_constraint_queries_keep_each_clue_independent():
    queries = build_constraint_queries(LONG_CLUE_QUESTION)
    assert len(queries) == 4
    assert "Cancer" in queries[0]
    assert "competition" in queries[2]
    assert "regulatory" in queries[3]


def test_simple_direct_question_is_not_multi_hop():
    assert is_multi_hop_question("What year was the university founded?") is False


def test_bootstrap_search_records_queries_and_evidence(monkeypatch):
    calls = []

    def fake_invoke(args):
        calls.append(args)
        return [{
            "rank": 1,
            "docid": f"doc-{len(calls)}",
            "score": 1.0,
            "snippet": "A useful clue passage.",
            "url": "https://example.com",
        }]

    monkeypatch.setattr(
        graph_module,
        "search_documents",
        SimpleNamespace(invoke=fake_invoke),
    )
    update = graph_module.bootstrap_search_node({
        "question": LONG_CLUE_QUESTION,
        "bootstrap_complete": False,
        "evidence": [],
        "visited_docs": [],
        "active_queries": [],
        "trajectory": [],
        "tool_calls": 0,
    })

    assert update["bootstrap_complete"] is True
    assert len(calls) == 3
    assert calls[0]["query"] == LONG_CLUE_QUESTION
    assert calls[0]["top_k"] == 20
    assert update["tool_calls"] == 3
    assert len(update["evidence"]) == 3
    assert update["trajectory"][0]["source"] == "deterministic_bootstrap"
    assert set(update["search_progress"]["seen_evidence_ids"]) == {
        item["evidence_id"] for item in update["evidence"]
    }


def test_bootstrap_saves_late_clues_until_search_stalls(monkeypatch):
    calls = []
    monkeypatch.setattr(
        graph_module, "search_documents",
        SimpleNamespace(invoke=lambda args: calls.append(args) or []),
    )
    planned = [f"distinct query about clue number {index}" for index in range(6)]
    state = {
        "question": LONG_CLUE_QUESTION, "planned_queries": planned,
        "bootstrap_complete": False, "evidence": [], "visited_docs": [],
        "active_queries": [], "trajectory": [], "tool_calls": 0,
    }
    state.update(graph_module.bootstrap_search_node(state))
    assert [call["query"] for call in calls] == [LONG_CLUE_QUESTION, *planned[:2]]
    assert state["deferred_queries"] == planned[2:]

    state["search_progress"] = {"made_progress": False}
    assert graph_module.route_after_search_progress(state) == "deferred_search"
    state.update(graph_module.bootstrap_search_node(state))
    assert calls[-1]["query"] == planned[2]

    state["force_final_answer"] = True
    assert graph_module.route_after_search_progress(state) == "deferred_search"
    state.update(graph_module.bootstrap_search_node(state))
    assert [call["query"] for call in calls] == [LONG_CLUE_QUESTION, *planned]
    assert state["deferred_queries"] == []
    assert graph_module.route_after_search_progress(state) == "agent"
