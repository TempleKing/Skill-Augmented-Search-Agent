"""LLM-assisted semantic query planning with deterministic fallback."""

from __future__ import annotations

import json
import re
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from agent.llm import configure_llm_for_required_tool, get_llm
from agent.question_analysis import build_bootstrap_queries
from config import settings


PLANNER_SYSTEM_PROMPT = """You plan BM25 searches for difficult multi-hop web questions.
Convert indirect descriptions into likely concrete terminology while preserving
rare phrases, dates, titles, occupations and relationships. For example, map a
generic description of a competition, award, disease, work or institution to
plausible names or standard category terms when justified.

Use these general concept bridges as hypotheses, not fixed answers:
- renowned competition -> Olympics, championship, tournament, contest
- highest national recognition -> national medal, national award, state honor
- regulatory molecule -> hormone, cytokine, protein, RNA, growth regulator
- disease-combating molecule -> antibody, antibiotic, immune factor
- degree and institute clues -> alumni, graduated, doctorate, university
- army clues -> military service, enlisted, armed forces
- film/book/song clues -> credits, cast, director, writer, discography, bibliography
- company clues -> acquisition, shareholder, account, rebrand, subsidiary
- sport clues -> final, roster, transfer, statistics, tournament
- approximate dates -> include plausible concrete years in separate queries

Return JSON only in this schema:
{"queries": ["short keyword query", "..."]}

Rules:
- Return 4 to 6 complementary queries, each 3 to 10 terms.
- Do not copy the full question.
- Cover different clue paths rather than paraphrasing one query.
- At least two queries must use useful standard terms or hypotheses that are
  not copied verbatim from the question.
- A possible entity may be included as a hypothesis, never as a claimed answer.
- No explanation and no Markdown.
"""


def _extract_json(text: str) -> dict[str, Any]:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", cleaned)
    try:
        value = json.loads(cleaned)
        return value if isinstance(value, dict) else {}
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", cleaned, re.DOTALL)
        if not match:
            return {}
        try:
            value = json.loads(match.group())
            return value if isinstance(value, dict) else {}
        except json.JSONDecodeError:
            return {}


def normalize_planned_queries(value: Any, limit: int = 6) -> list[str]:
    if not isinstance(value, list):
        return []
    queries = []
    for item in value:
        query = " ".join(str(item).split()).strip(" -\t")
        if not 3 <= len(query.split()) <= 12:
            continue
        if query.casefold() in {existing.casefold() for existing in queries}:
            continue
        queries.append(query)
        if len(queries) >= limit:
            break
    return queries


def plan_queries(question: str) -> tuple[list[str], dict[str, Any]]:
    """Return semantic queries and observable provider diagnostics."""
    fallback = build_bootstrap_queries(question)
    if not settings.llm_enable_query_planner:
        return fallback, {"status": "disabled", "total_tokens": 0}

    model = get_llm()
    if hasattr(model, "model_copy"):
        options: dict[str, Any] = {"max_tokens": settings.llm_planner_max_tokens or None}
        provider_url = settings.llm_base_url.casefold()
        if "api.deepseek.com" in provider_url:
            model = configure_llm_for_required_tool(model)
        model = model.model_copy(update=options)
    try:
        response = model.invoke([
            SystemMessage(content=PLANNER_SYSTEM_PROMPT),
            HumanMessage(content=question),
        ])
    except Exception as error:
        return fallback, {
            "status": "error",
            "error": f"{type(error).__name__}: {error}",
            "total_tokens": 0,
        }

    parsed = _extract_json(str(response.content))
    queries = normalize_planned_queries(parsed.get("queries"))
    usage = response.usage_metadata or {}
    return (queries or fallback), {
        "status": "success" if queries else "invalid_response",
        "content": response.content,
        "total_tokens": int(usage.get("total_tokens", 0) or 0),
        "usage_metadata": usage,
        "finish_reason": response.response_metadata.get("finish_reason"),
    }
