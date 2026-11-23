import json

from langchain_core.messages import HumanMessage, SystemMessage

from agent.graph import build_messages, settings
from agent.search_progress import initial_search_progress


def make_state(trajectory):
    snippet = "Queen Arwa University was established in 1996."
    return {
        "question": "Which university was established in which year?",
        "search_round": 1,
        "tool_calls": 1,
        "active_queries": ["Queen Arwa University established"],
        "visited_docs": [],
        "evidence": [{
            "evidence_id": "ev-1",
            "docid": "86190",
            "url": "https://example.com",
            "text": snippet,
            "source_tool": "search_documents",
            "query": "Queen Arwa University established",
            "score": -20.0,
            "rank": 1,
            "start_offset": None,
            "end_offset": None,
        }],
        "trajectory": trajectory,
        "required_facts": [],
        "skill_instructions": "",
        "empty_response_count": 0,
        "answer_validation": {},
        "search_progress": initial_search_progress(),
    }


def test_follow_up_context_keeps_system_message_before_user_message():
    messages = build_messages(make_state([]))

    assert len(messages) == 2
    assert isinstance(messages[0], SystemMessage)
    assert isinstance(messages[1], HumanMessage)
    assert messages[1].content.startswith("Which university was established in which year?")
    assert "RECORDED SEARCH STATE" in messages[1].content


def test_successful_tool_snippet_is_not_duplicated_in_context():
    snippet = "Queen Arwa University was established in 1996."
    trajectory = [{
        "type": "tool_result",
        "tool": "search_documents",
        "status": "success",
        "output": [{
            "docid": "86190",
            "snippet": snippet,
            "rank": 1,
            "score": -20.0,
        }],
    }]

    messages = build_messages(make_state(trajectory))

    assert messages[1].content.count(snippet) == 1


def test_tool_errors_remain_available_in_recent_context():
    trajectory = [{
        "type": "tool_result",
        "tool": "open_document",
        "status": "error",
        "output": {"error": "document_not_found"},
    }]

    messages = build_messages(make_state(trajectory))

    assert "document_not_found" in messages[1].content


def test_context_carries_one_old_search_passage_and_keeps_supporting_evidence():
    state = make_state([])
    original = state["evidence"][0]
    state["evidence"] = [
        {**original, "evidence_id": eid, "docid": docid, "rank": rank, "text": f"Passage {eid}"}
        for rank, (eid, docid) in enumerate((("e1", "11"), ("e2", "22"), ("e3", "33")), 1)
    ]
    state["required_facts"] = [{"fact_id": "fact-1", "evidence_ids": ["e3"]}]
    state["trajectory"] = [{"type": "llm_response", "context": {"evidence_ids": ["e1", "e2", "e3"]}}]
    metrics = {}

    messages = build_messages(state, metrics)
    payload = json.loads(messages[1].content.split("\n", 4)[-1].strip())
    evidence = payload["selected_evidence"]
    assert {item["evidence_id"] for item in evidence} == {"e1", "e3"}
    assert all("url" not in item and "score" not in item and "query" not in item for item in evidence)
    assert list(payload["queries"].values()) == [original["query"]]
    assert metrics["repeated_evidence_items"] == 2
