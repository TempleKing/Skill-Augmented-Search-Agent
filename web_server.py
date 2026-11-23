from __future__ import annotations

import argparse
import json
import mimetypes
import sqlite3
import threading
import time
from functools import lru_cache
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse

from config import settings


ROOT = Path(__file__).resolve().parent
FRONTEND_DIR = ROOT / "frontend" / "dist"
RUN_LOCK = threading.Lock()
VISIBLE_EVENT_TYPES = {
    "skill_loaded",
    "tool_call",
    "tool_result",
    "search_progress",
    "answer_validation",
    "budget_stop",
    "llm_error",
}


@lru_cache(maxsize=8)
def _inspect_index(path: str, size: int, modified_ns: int) -> tuple[bool, int | None]:
    """Cache read-only readiness by file identity; never report invented counts."""
    try:
        connection = sqlite3.connect(Path(path).as_uri() + "?mode=ro", uri=True, timeout=2)
        try:
            tables = {row[0] for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE name IN ('documents', 'doc_locator')"
            )}
            if tables != {"documents", "doc_locator"}:
                return False, None
            count = connection.execute("SELECT COUNT(*) FROM doc_locator").fetchone()[0]
            return count > 0, count
        finally:
            connection.close()
    except sqlite3.Error:
        return False, None


def health_payload() -> dict:
    path = settings.resolve_path(settings.bm25_db_path).resolve()
    try:
        stat = path.stat()
        index_ready, documents = _inspect_index(str(path), stat.st_size, stat.st_mtime_ns)
    except OSError:
        index_ready, documents = False, None
    return {
        "status": "ok",
        "llm_configured": bool(
            settings.llm_api_key and settings.llm_base_url and settings.llm_model
        ),
        "bm25_ready": index_ready,
        "corpus_documents": documents,
        "retrieval_method": settings.retrieval_method,
        "retrieval_top_k": settings.retrieval_top_k,
        "retrieval_candidate_k": settings.retrieval_candidate_k,
        "retrieval_rrf_k": settings.retrieval_rrf_k,
        "model": settings.llm_model,
        "max_agent_steps": settings.max_agent_steps,
        "max_tool_calls": settings.max_tool_calls,
        "max_stalled_searches": settings.max_stalled_searches,
        "max_total_tokens_per_run": settings.max_total_tokens_per_run,
        "min_remaining_tokens_for_llm_call": (
            settings.min_remaining_tokens_for_llm_call
        ),
    }


def public_run_result(result: dict, elapsed_seconds: float) -> dict:
    evidence = []
    for item in result.get("evidence", []):
        evidence.append({
            "evidence_id": item.get("evidence_id"),
            "docid": item.get("docid"),
            "url": item.get("url"),
            "title": item.get("title"),
            "text": item.get("text", ""),
            "source_tool": item.get("source_tool"),
            "query": item.get("query"),
            "score": item.get("score"),
            "rank": item.get("rank"),
            "start_offset": item.get("start_offset"),
            "end_offset": item.get("end_offset"),
        })
    trajectory = [
        event for event in result.get("trajectory", [])
        if event.get("type") in VISIBLE_EVENT_TYPES
    ]
    return {
        "answer": result.get("final_answer", ""),
        "termination_reason": result.get("termination_reason", ""),
        "selected_skills": result.get("selected_skills", []),
        "evidence": evidence,
        "trajectory": trajectory,
        "tool_calls": int(result.get("tool_calls", 0)),
        "total_tokens": int(result.get("total_tokens", 0)),
        "efficiency": result.get("efficiency", {}),
        "elapsed_seconds": elapsed_seconds,
    }


class RequestHandler(BaseHTTPRequestHandler):
    server_version = "CorpusSearch/1.0"

    def _json(self, payload: dict | list, status: int = 200) -> None:
        body = json.dumps(payload, ensure_ascii=False, default=str).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        self.wfile.write(body)

    def _read_json(self) -> dict:
        length = int(self.headers.get("Content-Length", "0"))
        if length <= 0 or length > 100_000:
            raise ValueError("Invalid request body")
        value = json.loads(self.rfile.read(length).decode("utf-8"))
        if not isinstance(value, dict):
            raise ValueError("Expected a JSON object")
        return value

    def _static(self, relative: str) -> None:
        if not (FRONTEND_DIR / "index.html").is_file():
            self.send_error(HTTPStatus.SERVICE_UNAVAILABLE, "Frontend build missing. Run npm ci and npm run build in frontend/.")
            return
        requested = (FRONTEND_DIR / relative).resolve()
        if not requested.is_relative_to(FRONTEND_DIR.resolve()) or not requested.is_file():
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        body = requested.read_bytes()
        content_type = mimetypes.guess_type(requested.name)[0] or "application/octet-stream"
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", f"{content_type}; charset=utf-8" if content_type.startswith("text/") or content_type in {"application/javascript", "application/json"} else content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header(
            "Content-Security-Policy",
            "default-src 'self'; style-src 'self'; script-src 'self'; "
            "img-src 'self' data:; connect-src 'self'; font-src 'self'; "
            "object-src 'none'; base-uri 'self'; frame-ancestors 'none'",
        )
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/api/health":
            self._json(health_payload())
            return
        if parsed.path.startswith("/api/documents/"):
            docid = unquote(parsed.path.removeprefix("/api/documents/")).strip()
            query = parse_qs(parsed.query)
            try:
                limit = min(4000, max(1, int(query.get("limit", ["4000"])[0])))
                offset = int(query.get("offset", ["0"])[0])
                if offset < 0:
                    raise ValueError("offset must be nonnegative")
            except ValueError:
                self._json({"error": "offset and limit must be valid nonnegative integers."}, 400)
                return
            try:
                from tools.document_tools import open_document
                result = open_document.invoke({"docid": docid, "offset": offset, "limit": limit})
                self._json(result, 404 if result.get("error") else 200)
            except Exception as error:
                self._json({"error": f"{type(error).__name__}: {error}"}, 500)
            return
        if parsed.path in {"/", "/index.html"}:
            self._static("index.html")
            return
        self._static(parsed.path.lstrip("/"))

    def do_POST(self) -> None:
        if urlparse(self.path).path != "/api/runs":
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        try:
            payload = self._read_json()
            raw_question = payload.get("question", "")
            if not isinstance(raw_question, str):
                self._json({"error": "Question must be a string."}, 400)
                return
            question = raw_question.strip()
            if not question:
                self._json({"error": "Question cannot be empty."}, 400)
                return
            if len(question) > 6000:
                self._json({"error": "Question cannot exceed 6000 characters."}, 400)
                return
            if payload.get("confirm_api_use") is not True:
                self._json({"error": "API use must be confirmed."}, 400)
                return
            health = health_payload()
            if not health["bm25_ready"] or not health["llm_configured"]:
                self._json({"error": "BM25 index and model configuration must be ready."}, 503)
                return
            if not RUN_LOCK.acquire(blocking=False):
                self._json({"error": "Another run is already active."}, 409)
                return
            started = time.perf_counter()
            try:
                from agent.graph import run_agent
                result = run_agent(question)
            finally:
                RUN_LOCK.release()
            self._json(public_run_result(result, time.perf_counter() - started))
        except (ValueError, json.JSONDecodeError) as error:
            self._json({"error": str(error)}, 400)
        except Exception as error:
            self._json({"error": f"{type(error).__name__}: {error}"}, 500)

    def log_message(self, format: str, *args) -> None:
        print(f"[web] {self.address_string()} {format % args}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the local course web interface.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=4173)
    args = parser.parse_args()
    server = ThreadingHTTPServer((args.host, args.port), RequestHandler)
    print(f"BM25 Search Workbench: http://{args.host}:{args.port}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
