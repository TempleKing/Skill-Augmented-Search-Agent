"""Deterministic question decomposition for retrieval planning.

This module does not guess the answer. It turns a long clue question into
small, auditable constraint groups that BM25 and the LLM can use.
"""

from __future__ import annotations

import math
import re


QUERY_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "been", "being", "but",
    "by", "did", "do", "does", "for", "from", "had", "has", "have", "he",
    "her", "hers", "him", "his", "how", "i", "in", "is", "it", "its",
    "known", "later", "me", "not", "of", "on", "or", "our", "said", "she",
    "so", "that", "the", "their", "them", "they", "this", "to", "under",
    "was", "were", "what", "when", "where", "which", "who", "with", "you",
    "your", "also", "interestingly", "particular", "same", "type",
    "role", "led", "awarded", "discovering", "plays", "pursued",
    "located", "completed", "participated", "unrelated",
}


def _final_request_start(question: str) -> int:
    question_mark = question.rfind("?")
    if question_mark < 0:
        return len(question)
    boundaries = [
        match.end()
        for match in re.finditer(r"[.!?](?:\s+|$)", question[:question_mark])
    ]
    return boundaries[-1] if boundaries else 0


def extract_search_constraints(question: str) -> list[str]:
    """Return descriptive clue clauses, excluding the final answer request."""
    clue_text = question[:_final_request_start(question)].strip()
    if not clue_text:
        clue_text = question.strip().rstrip("?")

    parts = [
        " ".join(part.split()).strip(" ,;:-")
        for part in re.split(r"(?<=[.!?;])\s+", clue_text)
    ]
    constraints = [part for part in parts if len(part.split()) >= 4]
    return constraints or ([clue_text] if clue_text else [])


def is_multi_hop_question(question: str) -> bool:
    """Classify questions that require several independent clue relations."""
    constraints = extract_search_constraints(question)
    word_count = len(re.findall(r"[^\W_]+", question, re.UNICODE))
    relation_markers = len(re.findall(
        r"\b(?:after|before|during|later|same year|different|while|whose|"
        r"according to|which led|served|awarded|discovered)\b",
        question,
        re.IGNORECASE,
    ))
    return (
        len(constraints) >= 3
        or (len(constraints) >= 2 and relation_markers >= 2)
        or (word_count >= 70 and relation_markers >= 2)
    )


def _query_terms(text: str, limit: int = 7) -> list[str]:
    terms: list[str] = []
    for token in re.findall(r"[A-Za-z0-9][A-Za-z0-9'’-]*", text):
        normalized = token.casefold().strip("'’-")
        if len(normalized) < 2 or normalized in QUERY_STOPWORDS:
            continue
        if normalized not in {term.casefold() for term in terms}:
            terms.append(token)
        if len(terms) >= limit:
            break
    return terms


def build_bootstrap_queries(question: str, max_queries: int = 3) -> list[str]:
    """Build complementary clue-group queries without answer-specific rules."""
    constraints = extract_search_constraints(question)
    if not constraints:
        return []

    query_count = min(max_queries, len(constraints))
    chunk_size = math.ceil(len(constraints) / query_count)
    queries: list[str] = []
    for start in range(0, len(constraints), chunk_size):
        terms = _query_terms(" ".join(constraints[start:start + chunk_size]))
        query = " ".join(terms)
        if query and query.casefold() not in {item.casefold() for item in queries}:
            queries.append(query)
    return queries[:max_queries]


def build_constraint_queries(question: str) -> list[str]:
    """Build one compact query per independent clue sentence."""
    queries: list[str] = []
    for constraint in extract_search_constraints(question):
        query = " ".join(_query_terms(constraint))
        if query and query.casefold() not in {item.casefold() for item in queries}:
            queries.append(query)
    return queries


def format_search_plan(question: str) -> str:
    constraints = extract_search_constraints(question)
    queries = build_bootstrap_queries(question)
    lines = ["Independent constraints:"]
    lines.extend(f"{index}. {item}" for index, item in enumerate(constraints, 1))
    if queries:
        lines.append("Complementary BM25 starting points:")
        lines.extend(f"- {query}" for query in queries)
    return "\n".join(lines)
