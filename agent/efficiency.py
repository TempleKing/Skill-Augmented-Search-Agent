"""Provider usage and observable sources of avoidable work."""

import json
import logging
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from config import settings


METRICS = (
    "api_requests", "api_errors", "usage_breakdown_requests", "reported_api_tokens", "input_tokens",
    "output_tokens", "cached_input_tokens", "reasoning_tokens", "instrumented_rounds",
    "message_chars", "evidence_items_sent", "repeated_evidence_items",
    "tool_requests", "tool_attempts", "tool_requests_skipped", "redundant_searches_skipped",
    "initial_searches", "deferred_searches", "progress_checks", "no_progress_rounds",
    "no_progress_round_tokens", "no_progress_round_input_tokens",
)


def summarize_efficiency(trajectory: Iterable[dict[str, Any]]) -> dict[str, int]:
    metrics: Counter[str] = Counter()
    rounds: dict[int, tuple[int, int]] = {}
    last_round = 0
    no_progress = set()
    for event in trajectory:
        kind = event.get("type")
        if kind in ("llm_response", "llm_error"):
            context = event.get("context")
            if context is not None:
                metrics["instrumented_rounds"] += 1
                for key in ("message_chars", "evidence_items", "repeated_evidence_items"):
                    target = "evidence_items_sent" if key == "evidence_items" else key
                    metrics[target] += int(context.get(key) or 0)
        if kind == "llm_error":
            metrics["api_requests"] += 1
            metrics["api_errors"] += 1
        if kind in ("llm_response", "query_plan"):
            usage = event.get("usage_metadata") or {}
            metrics["api_requests"] += 1
            metrics["api_errors"] += int(kind == "query_plan" and event.get("status") == "error")
            metrics["usage_breakdown_requests"] += int(
                usage.get("input_tokens") is not None and usage.get("output_tokens") is not None
            )
            total = int(event.get("total_tokens") or 0)
            input_tokens = int(usage.get("input_tokens") or 0)
            metrics["reported_api_tokens"] += total
            metrics["input_tokens"] += input_tokens
            metrics["output_tokens"] += int(usage.get("output_tokens") or 0)
            metrics["cached_input_tokens"] += int((usage.get("input_token_details") or {}).get("cache_read") or 0)
            metrics["reasoning_tokens"] += int((usage.get("output_token_details") or {}).get("reasoning") or 0)
            if kind == "llm_response":
                last_round = event.get("round") or last_round + 1
                rounds[last_round] = (total, input_tokens)
        elif kind == "search_progress":
            metrics["progress_checks"] += 1
            if event.get("made_progress") is False:
                no_progress.add(event.get("round") or last_round)
        elif kind == "tool_call":
            metrics["tool_requests"] += 1
            if event.get("source") == "deterministic_bootstrap":
                metrics["deferred_searches" if event.get("stage") == "deferred" else "initial_searches"] += 1
        elif kind == "tool_result":
            if event.get("executed", True):
                metrics["tool_attempts"] += 1
            else:
                metrics["tool_requests_skipped"] += 1
                output = event.get("output")
                if isinstance(output, dict) and output.get("error") == "redundant_search":
                    metrics["redundant_searches_skipped"] += 1
        elif kind == "search_skipped" and event.get("reason") == "redundant_search":
            metrics["redundant_searches_skipped"] += 1

    # A round classified as unproductive is an attribution signal. It does
    # not imply that every token in the request could have been saved.
    for round_number in no_progress:
        total, input_tokens = rounds.get(round_number, (0, 0))
        metrics["no_progress_round_tokens"] += total
        metrics["no_progress_round_input_tokens"] += input_tokens
    metrics["no_progress_rounds"] = len(no_progress)
    return {key: metrics[key] for key in METRICS}


def record_run_efficiency(
    result: dict[str, Any], run_id: str, elapsed_seconds: float, failed: bool
) -> dict[str, int]:
    metrics = summarize_efficiency(result.get("trajectory", []))
    folder = settings.resolve_path(Path("output/trajectories/efficiency"))
    output = folder / f"{run_id}.json"
    pending = output.with_suffix(".tmp")
    try:
        folder.mkdir(parents=True, exist_ok=True)
        pending.write_text(json.dumps({
            "run_id": run_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "model": settings.llm_model,
            "retrieval_method": settings.retrieval_method,
            "elapsed_seconds": elapsed_seconds,
            "termination_reason": "exception" if failed else result.get("termination_reason"),
            **metrics,
        }, indent=2), encoding="utf-8")
        pending.replace(output)
    except OSError as error:
        logging.getLogger(__name__).warning("Could not write run efficiency report: %s", error)
    return metrics
