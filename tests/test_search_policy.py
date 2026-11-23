from agent.answer_validator import extract_required_facts
from agent.search_policy import (
    candidate_early_stop,
    extract_candidate_entities,
    query_redundancy,
    query_similarity,
)


def person_facts():
    return extract_required_facts("Who is the scientist described by these clues?")


def evidence(evidence_id: str, docid: str, text: str):
    return {
        "evidence_id": evidence_id,
        "docid": docid,
        "text": text,
        "source_tool": "search_documents",
        "query": "",
    }


def test_candidate_entity_is_detected_from_targeted_query():
    candidates = extract_candidate_entities(
        "Ahmed Ezz band musician corruption",
        "Who was the tycoon described by these clues?",
        person_facts(),
    )
    assert candidates == ["Ahmed Ezz"]


def test_non_person_title_sequence_is_not_a_person_candidate():
    candidates = extract_candidate_entities(
        "Campion School Mumbai founded 1943",
        "Who was the actor?",
        person_facts(),
    )
    assert candidates == []


def test_person_candidate_stops_before_following_school_name():
    candidates = extract_candidate_entities(
        "Rishi Kapoor Campion School Mumbai education",
        "Who was the actor?",
        person_facts(),
    )
    assert candidates == ["Rishi Kapoor"]


def test_company_name_is_not_mistaken_for_person():
    candidates = extract_candidate_entities(
        "Ezz Steel 2023 majority shareholder",
        "Who was the businessman?",
        person_facts(),
    )
    assert candidates == []


def test_query_redundancy_uses_token_overlap_not_exact_text():
    score = query_similarity(
        "Mandela Effect percentage Americans familiar phenomenon",
        "percentage of Americans familiar with Mandela Effect phenomenon",
    )
    result = query_redundancy(
        ["Mandela Effect percentage Americans familiar phenomenon"],
        ["percentage of Americans familiar with Mandela Effect phenomenon"],
    )
    assert score > 0.8
    assert result["redundant_query"] is True


def test_candidate_early_stop_requires_two_independent_documents():
    items = [
        evidence("ev-1", "1", "Ahmed Ezz was an Egyptian steel tycoon."),
        evidence("ev-2", "2", "Businessman Ahmed Ezz founded Ezz Steel."),
    ]
    stopped = candidate_early_stop(["Ahmed Ezz"], items, person_facts())
    assert stopped is not None
    assert stopped["candidate"] == "Ahmed Ezz"
    assert stopped["supporting_docids"] == ["1", "2"]

    assert candidate_early_stop(
        ["Ahmed Ezz"],
        [items[0], evidence("ev-3", "1", "Ahmed Ezz returned to business.")],
        person_facts(),
    ) is None
