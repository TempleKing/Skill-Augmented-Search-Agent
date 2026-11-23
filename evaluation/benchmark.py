from __future__ import annotations

import json
import math
import re
import statistics
import unicodedata
from collections import Counter
from datetime import datetime
from pathlib import Path
from collections.abc import Callable, Iterator
from typing import Any, Iterable

from agent.efficiency import METRICS, summarize_efficiency


ARTICLES = re.compile(r"\b(a|an|the)\b", re.IGNORECASE)
NON_WORD = re.compile(r"[^\w\s]", re.UNICODE)


def normalize_answer(value: Any) -> str:
    """BrowseComp-style normalization for inexpensive local regression checks."""
    text = unicodedata.normalize("NFKC", str(value or "")).casefold()
    text = NON_WORD.sub(" ", text)
    text = ARTICLES.sub(" ", text)
    return " ".join(text.split())


def answer_metrics(prediction: str, reference: str) -> dict[str, float | bool]:
    predicted = normalize_answer(prediction)
    expected = normalize_answer(reference)
    exact = bool(expected) and predicted == expected
    contains = bool(expected and predicted) and (
        expected in predicted or predicted in expected
    )

    predicted_tokens = predicted.split()
    expected_tokens = expected.split()
    common = sum((Counter(predicted_tokens) & Counter(expected_tokens)).values())
    if not predicted_tokens or not expected_tokens or not common:
        token_f1 = 0.0
    else:
        precision = common / len(predicted_tokens)
        recall = common / len(expected_tokens)
        token_f1 = 2 * precision * recall / (precision + recall)

    return {"exact_match": exact, "contains_match": contains, "token_f1": token_f1}


def _compact_case(row: dict[str, Any], line_number: int) -> dict[str, Any]:
    question = row.get("query", row.get("question"))
    if question is None:
        raise ValueError(f"Line {line_number} needs a query/question field.")

    # The released validation split has an answer, while the final test split
    # intentionally does not. Keep one loader for both so the exact same
    # implementation can produce the final course submission.
    answer = row.get("answer", "")

    def docids(field: str) -> list[str]:
        return [
            str(item.get("docid") if isinstance(item, dict) else item)
            for item in (row.get(field, []) or [])
        ]

    # The official JSONL embeds full document text. Evaluation only needs IDs;
    # dropping text here keeps 830 cases small even though the source is >2 GB.
    return {
        "query_id": str(row.get("id", row.get("query_id", line_number))),
        "query": str(question),
        "answer": str(answer),
        "evidence_docs": docids("evidence_docs"),
        "gold_docs": docids("gold_docs"),
    }


def iter_cases(
    path: str | Path,
    progress: Callable[[int, int], None] | None = None,
) -> Iterator[dict[str, Any]]:
    source = Path(path)
    if source.suffix.lower() != ".jsonl":
        raise ValueError("Benchmark input must be a JSONL file.")

    total_bytes = source.stat().st_size
    with source.open("rb") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            yield _compact_case(json.loads(line), line_number)
            if progress:
                progress(handle.tell(), total_bytes)


def load_cases(
    path: str | Path,
    progress: Callable[[int, int], None] | None = None,
) -> list[dict[str, Any]]:
    return list(iter_cases(path, progress=progress))


def retrieved_docids(result: dict[str, Any]) -> list[str]:
    ids = {str(value) for value in result.get("visited_docs", [])}
    ids.update(
        str(item["docid"])
        for item in result.get("evidence", [])
        if item.get("docid") is not None
    )
    return sorted(ids)


def _gold_docids(case: dict[str, Any], field: str) -> set[str]:
    values = case.get(field, []) or []
    return {
        str(item.get("docid") if isinstance(item, dict) else item)
        for item in values
    }


def make_record(
    case: dict[str, Any], result: dict[str, Any], elapsed_seconds: float
) -> dict[str, Any]:
    prediction = str(result.get("final_answer", ""))
    retrieved = retrieved_docids(result)
    evidence_ids = _gold_docids(case, "evidence_docs")
    gold_ids = _gold_docids(case, "gold_docs")

    def recall(expected: set[str]) -> float | None:
        return len(expected.intersection(retrieved)) / len(expected) if expected else None

    return {
        "query_id": case["query_id"],
        "query": case["query"],
        "reference_answer": case["answer"],
        "prediction": prediction,
        **answer_metrics(prediction, case["answer"]),
        "status": "completed" if result.get("termination_reason") == "completed" else "failed",
        "termination_reason": result.get("termination_reason", ""),
        "llm_rounds": int(result.get("search_round", 0)),
        "tool_calls": int(result.get("tool_calls", 0)),
        "total_tokens": int(result.get("total_tokens", 0)),
        "elapsed_seconds": elapsed_seconds,
        "efficiency": result.get("efficiency") or summarize_efficiency(result.get("trajectory", [])),
        "retrieved_docids": retrieved,
        "evidence_recall": recall(evidence_ids),
        "gold_recall": recall(gold_ids),
        "error": None,
        # Required by the COMP5423 submission schema. Keep the original,
        # chronological machine-readable events, not a rendered trace.
        "trajectory": list(result.get("trajectory", [])),
    }


def course_submission_record(record: dict[str, Any]) -> dict[str, Any]:
    """Convert one local evaluation record to the COMP5423 JSONL schema."""
    return {
        "id": str(record["query_id"]),
        "question": str(record["query"]),
        "answer": str(record.get("prediction", "")),
        "trajectory": list(record.get("trajectory", [])),
        "tool_calls": int(record.get("tool_calls", 0)),
        "total_tokens": int(record.get("total_tokens", 0)),
    }


def validate_course_predictions(
    cases: list[dict[str, Any]], records: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Validate a complete official test run before writing its submission."""
    expected = {case["query_id"]: case["query"] for case in cases}
    if len(cases) != 100 or len(expected) != 100:
        raise ValueError("The course test set must contain exactly 100 unique IDs.")
    indexed = {row["query_id"]: row for row in records}
    if len(records) != 100 or set(indexed) != set(expected):
        raise ValueError("The course submission needs one prediction for every test ID.")

    predictions = []
    for qid, question in expected.items():
        row = indexed[qid]
        if row["query"] != question or is_operational_failure(row):
            raise ValueError(f"Test ID {qid} has the wrong question or an operational failure.")
        prediction = course_submission_record(row)
        events = prediction["trajectory"]
        try:
            times = [datetime.fromisoformat(event["timestamp"]) for event in events]
            chronological = bool(times) and times == sorted(times)
        except (KeyError, TypeError, ValueError):
            chronological = False
        if not chronological:
            raise ValueError(f"Test ID {qid} needs a complete chronological trajectory.")
        call_count = sum(event.get("type") == "tool_call" for event in events)
        if prediction["tool_calls"] != call_count:
            raise ValueError(f"Tool count differs from the trajectory for test ID {qid}.")
        billed_events = [
            event for event in events
            if event.get("type") == "llm_response"
            or (
                event.get("type") == "query_plan"
                and event.get("status") in {"success", "invalid_response"}
            )
        ]
        usages = [event.get("total_tokens") for event in billed_events]
        if (
            not prediction["answer"].strip()
            or not any(event.get("type") == "llm_response" for event in billed_events)
            or any(type(value) is not int or value <= 0 for value in usages)
            or prediction["total_tokens"] != sum(usages)
        ):
            raise ValueError(f"Test ID {qid} needs an answer and API-reported token counts.")
        predictions.append(prediction)
    return predictions


def is_operational_failure(record: dict[str, Any]) -> bool:
    """Return whether resume should retry a case rather than score it."""
    return bool(record.get("error")) or record.get("termination_reason") in {
        "exception",
        "llm_error_retries_exhausted",
    }


def percentile(values: list[float], quantile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = max(0, math.ceil(quantile * len(ordered)) - 1)
    return ordered[index]


def summarize(records: Iterable[dict[str, Any]]) -> dict[str, Any]:
    rows = list(records)
    count = len(rows)
    scored_rows = [row for row in rows if not is_operational_failure(row)]
    operational_failures = count - len(scored_rows)

    def mean(field: str, source: list[dict[str, Any]] | None = None) -> float:
        selected = rows if source is None else source
        values = [float(row[field]) for row in selected if row.get(field) is not None]
        return statistics.fmean(values) if values else 0.0

    timings = [float(row.get("elapsed_seconds", 0)) for row in rows]
    total_tokens = sum(int(row.get("total_tokens", 0)) for row in rows)
    exact_count = sum(bool(row.get("exact_match")) for row in rows)
    failed_tokens = sum(
        int(row.get("total_tokens", 0))
        for row in rows
        if not row.get("exact_match")
    )
    reasons = Counter(row.get("termination_reason") or "error" for row in rows)
    efficiency: Counter[str] = Counter()
    for row in rows:
        efficiency.update(row.get("efficiency") or summarize_efficiency(row.get("trajectory", [])))
    return {
        "cases": count,
        "scored_cases": len(scored_rows),
        "operational_failures": operational_failures,
        "raw_accuracy": mean("exact_match"),
        "accuracy": mean("exact_match", scored_rows),
        "contains_accuracy": mean("contains_match", scored_rows),
        "mean_token_f1": mean("token_f1", scored_rows),
        "completion_rate": (
            sum(row.get("status") == "completed" for row in scored_rows) /
            len(scored_rows) if scored_rows else 0.0
        ),
        "unknown_rate": (
            sum(normalize_answer(row.get("prediction")) == "unknown" for row in scored_rows) /
            len(scored_rows) if scored_rows else 0.0
        ),
        "avg_llm_rounds": mean("llm_rounds"),
        "avg_tool_calls": mean("tool_calls"),
        "avg_total_tokens": mean("total_tokens"),
        "total_tokens": total_tokens,
        "tokens_per_exact_match": (
            total_tokens / exact_count if exact_count else None
        ),
        "failed_case_tokens": failed_tokens,
        "avg_elapsed_seconds": mean("elapsed_seconds"),
        "p50_elapsed_seconds": percentile(timings, 0.50),
        "p95_elapsed_seconds": percentile(timings, 0.95),
        "avg_evidence_recall": mean("evidence_recall"),
        "avg_gold_recall": mean("gold_recall"),
        "termination_reasons": dict(sorted(reasons.items())),
        "efficiency": {key: efficiency[key] for key in METRICS},
    }


def official_run_record(record: dict[str, Any]) -> dict[str, Any]:
    """Convert one local result to BrowseComp-Plus' official run schema."""
    return {
        "query_id": record["query_id"],
        "tool_call_counts": {"search": record.get("tool_calls", 0)},
        "status": record.get("status", "failed"),
        "retrieved_docids": record.get("retrieved_docids", []),
        "result": [{"type": "output_text", "output": record.get("prediction", "")}],
    }
