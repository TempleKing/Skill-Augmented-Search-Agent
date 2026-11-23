import json
from types import SimpleNamespace

from langchain_core.messages import AIMessage
from langchain_core.tools import tool
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode

import agent.efficiency as efficiency
import agent.graph as graph_module
from agent.state import SearchAgentState


def test_tool_node_skips_redundant_search_but_keeps_new_clue_and_depth(monkeypatch):
    executed = []

    @tool("search_documents")
    def search(query: str, top_k: int = 8) -> list[dict]:
        """Search test documents."""
        executed.append((query, top_k))
        return [{"docid": query, "snippet": "Matching evidence", "rank": 1}]

    monkeypatch.setattr(graph_module, "tool_node", ToolNode([search]))
    builder = StateGraph(SearchAgentState)
    builder.add_node("tools", graph_module.execute_tools_node)
    builder.add_node("record", graph_module.record_tools_node)
    builder.add_edge(START, "tools")
    builder.add_edge("tools", "record")
    builder.add_edge("record", END)

    prior = "Boston festival award 2010"
    proposed = [("skip", "Boston festival award", 8), ("year", prior.replace("2010", "2011"), 8),
                ("depth", prior, 12)]
    requests = [
        {"type": "tool_call", "tool_call_id": cid, "tool": "search_documents",
         "input": {"query": query, "top_k": top_k}}
        for cid, query, top_k in proposed
    ]
    result = builder.compile().invoke({
        "question": "When did the festival happen?",
        "messages": [AIMessage(content="", tool_calls=[
            {"id": request["tool_call_id"], "name": request["tool"], "args": request["input"]}
            for request in requests
        ])],
        "search_round": 1,
        "tool_calls": 1,
        "evidence": [],
        "active_queries": [prior],
        "visited_docs": [],
        "trajectory": [
            {"type": "tool_call", "tool_call_id": "prior", "tool": "search_documents",
             "input": {"query": prior, "top_k": 8}},
            {"type": "tool_result", "tool_call_id": "prior", "tool": "search_documents",
             "status": "success", "output": [{"docid": "first", "snippet": "Evidence"}]},
            *requests,
        ],
    })
    assert sorted(executed) == sorted((query, top_k) for _, query, top_k in proposed[1:])
    tool_results = [event for event in result["trajectory"] if event["type"] == "tool_result"]
    assert [event["tool_call_id"] for event in tool_results] == ["prior", "skip", "year", "depth"]
    assert tool_results[1]["output"]["error"] == "redundant_search"
    assert tool_results[1]["executed"] is False
    assert result["tool_calls"] == 4
    assert efficiency.summarize_efficiency(result["trajectory"])["tool_attempts"] == 3


def test_efficiency_reports_usage_and_writes_run_history(tmp_path, monkeypatch):
    monkeypatch.setattr(efficiency, "settings", SimpleNamespace(
        resolve_path=lambda path: tmp_path / path,
        llm_model="test-model", retrieval_method="bm25",
    ))
    trajectory = [
        {"type": "llm_response", "round": 1, "total_tokens": 120,
         "usage_metadata": {"input_tokens": 100, "output_tokens": 20,
                            "input_token_details": {"cache_read": 8}},
         "context": {"message_chars": 220, "evidence_items": 3, "repeated_evidence_items": 2}},
        {"type": "search_progress", "round": 1, "made_progress": False},
        {"type": "search_progress", "round": 1, "made_progress": False},
        {"type": "tool_result", "executed": False, "output": {"error": "redundant_search"}},
    ]
    metrics = efficiency.record_run_efficiency(
        {"trajectory": trajectory, "termination_reason": "completed"}, "run-id", 2.5, False,
    )
    assert metrics["reported_api_tokens"] == 120
    assert metrics["cached_input_tokens"] == 8
    assert metrics["repeated_evidence_items"] == 2
    assert metrics["redundant_searches_skipped"] == 1
    assert metrics["no_progress_round_tokens"] == 120
    assert metrics["no_progress_rounds"] == 1
    report = json.loads((tmp_path / "output/trajectories/efficiency/run-id.json").read_text("utf-8"))
    assert report["no_progress_round_input_tokens"] == 100
    assert report["termination_reason"] == "completed"
