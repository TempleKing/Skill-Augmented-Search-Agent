from __future__ import annotations

import argparse
import json
import random
import sys
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluation.benchmark import (
    course_submission_record,
    is_operational_failure,
    load_cases,
    make_record,
    official_run_record,
    summarize,
    validate_course_predictions,
)

# Small defaults keep a direct invocation limited to one question.
DEFAULT_DATASET = ROOT / "data" / "validation.jsonl"
DEFAULT_LIMIT = 1
DEFAULT_SEED = 42
DEFAULT_TOTAL_TOKEN_BUDGET = 25_000


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate the BrowseComp-Plus BM25 baseline.")
    parser.add_argument(
        "--dataset",
        default=str(DEFAULT_DATASET),
        help=f"Course validation or test JSONL file (default: {DEFAULT_DATASET}).",
    )
    parser.add_argument("--output", help="Output directory; defaults to output/evaluations/<timestamp>.")
    parser.add_argument(
        "--limit", type=int, default=DEFAULT_LIMIT,
        help=f"Maximum cases to run (default: {DEFAULT_LIMIT}; use 100 for the full course set).",
    )
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--shuffle", action=argparse.BooleanOptionalAction, default=True,
        help="Deterministically shuffle before selecting cases (default: enabled).",
    )
    parser.add_argument("--resume", action="store_true", help="Skip query IDs already in results.jsonl.")
    parser.add_argument("--workers", type=int, default=1, help="Maximum simultaneously running questions.")
    parser.add_argument(
        "--course-test", action="store_true",
        help="Require a complete unlabelled 100-case test set and validate the final submission.",
    )
    parser.add_argument(
        "--trace", action=argparse.BooleanOptionalAction, default=True,
        help="Show normal gray agent trace output (default: enabled).",
    )
    parser.add_argument(
        "--confirm-paid-run", action="store_true",
        help="Required acknowledgment that this command may call the paid LLM API.",
    )
    parser.add_argument(
        "--max-total-tokens", type=int, default=DEFAULT_TOTAL_TOKEN_BUDGET,
        help=("Stop scheduling cases once completed cases reach this token threshold. "
              "Already running cases may add tokens. "
              f"Use 0 to disable this check (default: {DEFAULT_TOTAL_TOKEN_BUDGET})."),
    )
    parser.add_argument(
        "--plan-only", action="store_true",
        help="Print the selected case IDs without importing or calling the agent.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_path = Path(args.dataset)
    if not dataset_path.exists():
        raise FileNotFoundError(
            f"Test dataset not found: {dataset_path}\n"
            "Place the course JSONL under data/ or pass its actual path with --dataset."
        )
    if args.limit is not None and args.limit < 1:
        raise ValueError("--limit must be at least 1.")
    if args.offset < 0:
        raise ValueError("--offset must be nonnegative.")
    if args.workers < 1:
        raise ValueError("--workers must be at least 1.")
    if args.max_total_tokens < 0:
        raise ValueError("--max-total-tokens must be at least 0.")
    if not args.plan_only and not args.confirm_paid_run:
        raise RuntimeError(
            "Paid evaluation blocked. Re-run with --confirm-paid-run after "
            "reviewing --limit and --max-total-tokens, or use --plan-only."
        )

    print("=" * 60)
    print("BrowseComp-Plus Agent Evaluation")
    print("=" * 60)
    print(f"Dataset: {dataset_path}")
    print(f"Sample:  limit={args.limit}, shuffle={args.shuffle}, seed={args.seed}")
    print(f"Trace:   {args.trace}")

    last_percent = -1

    def loading_progress(read_bytes: int, total_bytes: int) -> None:
        nonlocal last_percent
        percent = int(read_bytes * 100 / total_bytes) if total_bytes else 100
        if percent >= last_percent + 5 or percent == 100:
            last_percent = percent
            print(f"[dataset] loading {percent}%", flush=True)

    print(f"[dataset] reading {dataset_path}", flush=True)
    cases = load_cases(dataset_path, progress=loading_progress)
    source_count = len(cases)
    print(f"[dataset] loaded {len(cases)} cases", flush=True)
    if args.shuffle:
        random.Random(args.seed).shuffle(cases)
    cases = cases[args.offset:]
    if args.limit is not None:
        cases = cases[:args.limit]
    if args.course_test and (
        source_count != 100
        or len(cases) != 100
        or len({case["query_id"] for case in cases}) != 100
        or any(case["answer"].strip() for case in cases)
    ):
        raise ValueError("--course-test requires all 100 unique unlabelled questions.")

    if args.plan_only:
        print("Plan only; no Agent or LLM call will be made.")
        for case in cases:
            print(f"- {case['query_id']}: {case['query']}")
        return

    output = Path(args.output) if args.output else (
        ROOT / "output" / "evaluations" / datetime.now().strftime("%Y%m%d_%H%M%S")
    )
    if not output.is_absolute():
        output = ROOT / output
    output.mkdir(parents=True, exist_ok=True)
    results_path = output / "results.jsonl"
    submission_path = output / "test_prediction.jsonl"
    existing = []
    if not args.resume and results_path.exists():
        raise FileExistsError(f"{results_path} already exists. Use --resume or a new output directory.")
    if args.resume and results_path.exists():
        existing = [json.loads(line) for line in results_path.read_text(encoding="utf-8").splitlines() if line]
        selected = {case["query_id"]: case["query"] for case in cases}
        if len({row["query_id"] for row in existing}) != len(existing) or any(
            selected.get(row["query_id"]) != row["query"] for row in existing
        ):
            raise ValueError("Existing results do not match the selected questions. Use a new output directory.")
        # Interrupted or failed cases must be retried. Rewrite the local file
        # first so each question still has exactly one record after resuming.
        existing = [row for row in existing if not is_operational_failure(row)]
        temporary = results_path.with_suffix(".tmp")
        with temporary.open("w", encoding="utf-8") as handle:
            for row in existing:
                handle.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
        temporary.replace(results_path)
    completed_ids = {row["query_id"] for row in existing}

    from agent.graph import run_agent
    from agent.hooks import ConsoleTraceHook
    from config import settings

    run_config = {
        "dataset": str(dataset_path.resolve()),
        "case_ids": [case["query_id"] for case in cases],
        "model": settings.llm_model,
        "retrieval_method": settings.retrieval_method,
        "max_stalled_searches": settings.max_stalled_searches,
        "retrieval_candidate_k": settings.retrieval_candidate_k,
        "retrieval_rrf_k": settings.retrieval_rrf_k,
        "temperature": settings.llm_temperature,
        "thinking": settings.llm_enable_thinking,
        "reasoning_effort": settings.llm_reasoning_effort,
        "thinking_rounds": settings.llm_thinking_rounds,
        "query_planner": settings.llm_enable_query_planner,
        "retrieval_top_k": settings.retrieval_top_k,
        "max_agent_steps": settings.max_agent_steps,
        "max_tool_calls": settings.max_tool_calls,
        "max_total_tokens_per_run": settings.max_total_tokens_per_run,
        "max_total_tokens_this_run": args.max_total_tokens,
        "workers": args.workers,
    }
    (output / "run_config.json").write_text(
        json.dumps(run_config, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    records = list(existing)
    pending = [case for case in cases if case["query_id"] not in completed_ids]
    if args.workers > 1 and pending:
        from agent.llm import get_llm

        get_llm()

    def evaluate_case(index: int, case: dict) -> dict:
        print(f"[{index}/{len(pending)}] {case['query_id']}", flush=True)
        started = time.perf_counter()
        try:
            result = run_agent(case["query"], hooks=ConsoleTraceHook() if args.trace else None)
            record = make_record(case, result, time.perf_counter() - started)
        except Exception as error:
            record = {
                "query_id": case["query_id"], "query": case["query"],
                "reference_answer": case["answer"], "prediction": "",
                "exact_match": False, "contains_match": False, "token_f1": 0.0,
                "status": "failed", "termination_reason": "exception",
                "llm_rounds": 0, "tool_calls": 0, "total_tokens": 0,
                "elapsed_seconds": time.perf_counter() - started,
                "retrieved_docids": [], "evidence_recall": None, "gold_recall": None,
                "error": f"{type(error).__name__}: {error}",
                "trajectory": [{
                    "type": "run_error",
                    "error_type": type(error).__name__,
                    "error": str(error),
                }],
            }
        return record

    def checkpoint(record: dict) -> None:
        records.append(record)
        with results_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        print(
            f"  {record['query_id']} exact={record['exact_match']} rounds={record['llm_rounds']} "
            f"tools={record['tool_calls']} tokens={record['total_tokens']} "
            f"time={record['elapsed_seconds']:.2f}s",
            flush=True,
        )

    scheduled = iter(enumerate(pending, 1))
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        in_flight = set()
        tasks_remaining = True
        while tasks_remaining or in_flight:
            while tasks_remaining and len(in_flight) < args.workers:
                spent_tokens = sum(int(row.get("total_tokens", 0)) for row in records)
                if 0 < args.max_total_tokens <= spent_tokens:
                    print(f"Batch token budget reached ({spent_tokens} >= {args.max_total_tokens}).", flush=True)
                    tasks_remaining = False
                    break
                scheduled_case = next(scheduled, None)
                if scheduled_case is None:
                    tasks_remaining = False
                    break
                in_flight.add(pool.submit(evaluate_case, *scheduled_case))
            if in_flight:
                finished, in_flight = wait(in_flight, return_when=FIRST_COMPLETED)
                for task in finished:
                    checkpoint(task.result())

    order = {case["query_id"]: index for index, case in enumerate(cases)}
    records.sort(key=lambda row: order[row["query_id"]])
    temporary_results = results_path.with_suffix(".tmp")
    with temporary_results.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    temporary_results.replace(results_path)

    summary = summarize(records)
    (output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    official_dir = output / "official_run"
    official_dir.mkdir(exist_ok=True)
    for record in records:
        (official_dir / f"{record['query_id']}.json").write_text(
            json.dumps(official_run_record(record), ensure_ascii=False, indent=2), encoding="utf-8"
        )
    predictions = (
        validate_course_predictions(cases, records)
        if args.course_test
        else [course_submission_record(row) for row in records]
    )
    temporary = submission_path.with_suffix(".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for prediction in predictions:
            handle.write(json.dumps(
                prediction,
                ensure_ascii=False,
                default=str,
            ) + "\n")
    temporary.replace(submission_path)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Saved evaluation to {output}")
    print(f"Course submission: {submission_path}")


if __name__ == "__main__":
    main()
