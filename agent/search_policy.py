import re
from typing import Any

from agent.answer_validator import validate_answer
from config import settings


QUERY_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "born", "by", "did",
    "for", "from", "had", "has", "have", "he", "her", "his", "how",
    "in", "is", "it", "later", "of", "on", "or", "same", "she", "that",
    "the", "their", "they", "this", "to", "was", "were", "what", "when",
    "where", "which", "who", "with", "year",
}

PERSON_ENTITY_BLOCKLIST = {
    "award", "college", "company", "corporation", "effect", "games",
    "institute", "medal", "olympics", "prize", "school", "steel",
    "university",
}

def normalized_query_tokens(query: str) -> set[str]:
    return {
        token
        for token in re.findall(r"[^\W_]+", query.casefold(), re.UNICODE)
        if len(token) > 1 and token not in QUERY_STOPWORDS
    }


def query_similarity(left: str, right: str) -> float:
    left_tokens = normalized_query_tokens(left)
    right_tokens = normalized_query_tokens(right)
    if not left_tokens or not right_tokens:
        return 0.0
    return len(left_tokens & right_tokens) / len(left_tokens | right_tokens)


def query_redundancy(
    latest_queries: list[str],
    previous_queries: list[str],
    threshold: float = 0.68,
) -> dict[str, Any]:
    best_score = 0.0
    best_pair: tuple[str, str] | None = None
    for current in latest_queries:
        for previous in previous_queries:
            score = query_similarity(current, previous)
            if score > best_score:
                best_score = score
                best_pair = (current, previous)
    return {
        "redundant_query": best_score >= threshold,
        "max_query_similarity": round(best_score, 4),
        "similar_query_pair": list(best_pair) if best_pair else [],
    }


def successful_searches(trajectory: list[dict[str, Any]]) -> list[tuple[str, int, bool]]:
    calls = {}
    results = []
    for event in trajectory:
        if event.get("tool") != "search_documents":
            continue
        call_id = event.get("tool_call_id")
        if event.get("type") == "tool_call":
            calls[call_id] = event.get("input") or {}
        elif (
            event.get("type") == "tool_result"
            and event.get("status") == "success"
            and event.get("executed", True)
            and isinstance(event.get("output"), list)
        ):
            args = calls.get(call_id, {})
            query = args.get("query")
            top_k = args.get("top_k", settings.retrieval_top_k)
            if isinstance(query, str) and isinstance(top_k, int):
                results.append((query, top_k, bool(event["output"])))
    return results


def redundant_search(query: str, top_k: int, previous: list[tuple[str, int, bool]]) -> str | None:
    """Find a prior request that already covers this query and retrieval depth."""
    terms = normalized_query_tokens(query)
    for prior, prior_top_k, had_results in reversed(previous):
        if top_k > prior_top_k:
            continue
        if " ".join(query.casefold().split()) == " ".join(prior.casefold().split()):
            return prior
        # Keep queries that introduce new clues, or relax a query with no hits.
        if had_results and terms and terms <= normalized_query_tokens(prior):
            if query_redundancy([query], [prior])["redundant_query"]:
                return prior
    return None


def _titlecase_sequences(query: str) -> list[list[str]]:
    tokens = re.findall(r"[A-Za-z][A-Za-z'.-]*", query)
    sequences: list[list[str]] = []
    current: list[str] = []
    for token in tokens:
        is_name_token = token[:1].isupper() and (
            token[1:].islower() or token.isupper() or len(token) == 1
        )
        if is_name_token:
            current.append(token.strip(".'-"))
        else:
            if len(current) >= 2:
                sequences.append(current)
            current = []
    if len(current) >= 2:
        sequences.append(current)
    return sequences


def extract_candidate_entities(
    query: str,
    question: str,
    required_facts: list[dict[str, Any]],
) -> list[str]:
    """Extract answer-like named entities introduced by a search query."""
    fact_types = {
        fact.get("type")
        for fact in required_facts
        if fact.get("required", True)
    }
    if len(fact_types) != 1 or not fact_types.intersection({
        "person", "organization", "location"
    }):
        return []

    fact_type = next(iter(fact_types))
    question_normalized = " ".join(question.casefold().split())
    candidates: list[str] = []
    for sequence in _titlecase_sequences(query):
        if fact_type == "person":
            kept = []
            for token in sequence:
                if token.casefold() in PERSON_ENTITY_BLOCKLIST:
                    if token.casefold() in {
                        "college", "company", "corporation", "effect",
                        "games", "institute", "olympics", "school",
                        "university",
                    } and len(kept) > 2:
                        kept.pop()
                    break
                kept.append(token)
            sequence = kept[:4]
        else:
            sequence = sequence[:6]

        if len(sequence) < 2:
            continue
        candidate = " ".join(sequence)
        if candidate.casefold() in question_normalized:
            continue
        if candidate not in candidates:
            candidates.append(candidate)
    return candidates


def latest_and_previous_search_queries(
    trajectory: list[dict[str, Any]],
) -> tuple[list[str], list[str]]:
    last_llm_index = next(
        (
            index
            for index in range(len(trajectory) - 1, -1, -1)
            if trajectory[index].get("type") == "llm_response"
        ),
        -1,
    )
    previous: list[str] = []
    latest: list[str] = []
    skipped = {
        event.get("tool_call_id") for event in trajectory
        if event.get("type") == "tool_result" and event.get("executed") is False
    }
    for index, event in enumerate(trajectory):
        if event.get("type") != "tool_call" or event.get("tool") != "search_documents":
            continue
        if event.get("tool_call_id") in skipped:
            continue
        query = str((event.get("input") or {}).get("query", "")).strip()
        if not query:
            continue
        (latest if index > last_llm_index else previous).append(query)
    return latest, previous


def candidate_early_stop(
    candidates: list[str],
    evidence: list[dict[str, Any]],
    required_facts: list[dict[str, Any]],
) -> dict[str, Any] | None:
    """Return a type-safe candidate supported by at least two documents."""
    for candidate in reversed(candidates):
        normalized = " ".join(candidate.casefold().split())
        supporting_items = [
            item
            for item in evidence
            if normalized in " ".join(
                str(item.get("text", "")).casefold().split()
            )
        ]
        supporting_docs = list(dict.fromkeys(
            str(item.get("docid"))
            for item in supporting_items
            if item.get("docid") is not None
        ))
        if len(supporting_docs) < 2:
            continue

        validation = validate_answer(candidate, evidence, required_facts)
        if not validation.get("valid"):
            continue
        return {
            "candidate": candidate,
            "supporting_docids": supporting_docs,
            "supporting_evidence_ids": list(dict.fromkeys(
                item.get("evidence_id")
                for item in supporting_items
                if item.get("evidence_id")
            )),
            "validation": validation,
        }
    return None


def query_mentions_candidate(query: str, candidates: list[str]) -> bool:
    normalized = " ".join(query.casefold().split())
    return any(
        " ".join(candidate.casefold().split()) in normalized
        for candidate in candidates
    )
