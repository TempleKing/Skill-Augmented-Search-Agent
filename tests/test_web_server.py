import json
import sqlite3
from contextlib import closing
from types import SimpleNamespace

import pytest

import web_server
from config import settings
from web_server import RequestHandler, health_payload, public_run_result


def test_public_result_exposes_required_ui_fields_without_llm_messages():
    result = public_run_result({
        "final_answer": "Answer",
        "termination_reason": "completed",
        "selected_skills": ["multi-hop-search"],
        "evidence": [{
            "docid": "42", "url": "https://example.test",
            "text": "support", "source_tool": "search_documents",
        }],
        "trajectory": [
            {"type": "llm_response", "content": "private model message"},
            {"type": "tool_call", "tool": "search_documents"},
            {"type": "tool_result", "tool": "search_documents"},
        ],
        "tool_calls": 1,
        "total_tokens": 100,
    }, 1.25)

    assert result["answer"] == "Answer"
    assert result["selected_skills"] == ["multi-hop-search"]
    assert result["evidence"][0]["docid"] == "42"
    assert [event["type"] for event in result["trajectory"]] == [
        "tool_call", "tool_result",
    ]
    assert result["tool_calls"] == 1
    assert result["total_tokens"] == 100
    assert result["elapsed_seconds"] == 1.25


def test_health_reports_actual_index_count_and_never_credentials(tmp_path, monkeypatch):
    database = tmp_path / "corpus.sqlite"
    with closing(sqlite3.connect(database)) as connection, connection:
        connection.executescript(
            "CREATE VIRTUAL TABLE documents USING fts5(docid UNINDEXED, text, url UNINDEXED);"
            "CREATE TABLE doc_locator (docid TEXT PRIMARY KEY, document_rowid INTEGER);"
            "INSERT INTO documents VALUES ('one', 'An archive entry.', '');"
            "INSERT INTO doc_locator VALUES ('one', 1);"
        )
    monkeypatch.setattr(settings, "bm25_db_path", database)
    monkeypatch.setattr(settings, "llm_api_key", "private-test-credential")
    health = health_payload()
    assert health["bm25_ready"] is True
    assert health["corpus_documents"] == 1
    assert health["retrieval_top_k"] == settings.retrieval_top_k
    assert health["model"] == settings.llm_model
    assert "private-test-credential" not in json.dumps(health)


def test_health_does_not_treat_an_empty_file_as_a_ready_index(tmp_path, monkeypatch):
    database = tmp_path / "empty.sqlite"
    database.touch()
    monkeypatch.setattr(settings, "bm25_db_path", database)
    assert health_payload()["bm25_ready"] is False
    assert health_payload()["corpus_documents"] is None


def fake_handler(path):
    handler = object.__new__(RequestHandler)
    handler.path = path
    responses = []
    handler._json = lambda payload, status=200: responses.append((status, payload))
    return handler, responses


def test_document_endpoint_passes_requested_offset_to_the_reader(monkeypatch):
    import tools.document_tools

    calls = []
    monkeypatch.setattr(tools.document_tools, "open_document", SimpleNamespace(
        invoke=lambda arguments: calls.append(arguments) or {"docid": "example", "content": "Second page"},
    ))
    handler, responses = fake_handler("/api/documents/example?offset=4000&limit=300")
    handler.do_GET()
    assert calls == [{"docid": "example", "offset": 4000, "limit": 300}]
    assert responses[0][0] == 200


@pytest.mark.parametrize("offset", ["-1", "invalid"])
def test_document_endpoint_rejects_invalid_offsets(offset):
    handler, responses = fake_handler(f"/api/documents/example?offset={offset}")
    handler.do_GET()
    assert responses[0][0] == 400


def test_run_endpoint_requires_confirmation_before_readiness_or_model_calls(monkeypatch):
    monkeypatch.setattr(web_server, "health_payload", lambda: pytest.fail("Readiness must not run before confirmation"))
    handler, responses = fake_handler("/api/runs")
    handler._read_json = lambda: {"question": "Find a researcher."}
    handler.do_POST()
    assert responses[0][0] == 400


def test_run_endpoint_rejects_oversized_questions_before_any_model_call():
    handler, responses = fake_handler("/api/runs")
    handler._read_json = lambda: {"question": "x" * 6001, "confirm_api_use": True}
    handler.do_POST()
    assert responses[0][0] == 400


@pytest.mark.parametrize("question", [None, ["question"], {"text": "question"}])
def test_run_endpoint_rejects_non_string_questions(question):
    handler, responses = fake_handler("/api/runs")
    handler._read_json = lambda: {"question": question, "confirm_api_use": True}
    handler.do_POST()
    assert responses[0][0] == 400


def test_run_endpoint_returns_unavailable_for_missing_index_or_model(monkeypatch):
    monkeypatch.setattr(web_server, "health_payload", lambda: {"bm25_ready": False, "llm_configured": True})
    handler, responses = fake_handler("/api/runs")
    handler._read_json = lambda: {"question": "Find a researcher.", "confirm_api_use": True}
    handler.do_POST()
    assert responses[0][0] == 503


def test_static_server_rejects_traversal_outside_the_frontend_build(tmp_path, monkeypatch):
    frontend = tmp_path / "dist"
    frontend.mkdir()
    (frontend / "index.html").write_text("app", encoding="utf-8")
    (tmp_path / "private.txt").write_text("private", encoding="utf-8")
    monkeypatch.setattr(web_server, "FRONTEND_DIR", frontend)
    handler, _ = fake_handler("/")
    errors = []
    handler.send_error = lambda code, *args: errors.append(code)
    handler._static("../private.txt")
    assert errors == [404]
