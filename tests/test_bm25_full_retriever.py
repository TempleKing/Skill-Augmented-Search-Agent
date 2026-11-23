import sqlite3
from contextlib import closing

from retrieval.bm25_full_retriever import (
    BM25FullRetriever,
    build_query_variants,
)
from agent.tool_registry import search_documents
from config import settings


def test_query_variants_remove_stopwords_and_keep_entity_phrases():
    variants = build_query_variants(
        "Which university held the Fifth Cultural Week in May 2002?"
    )
    assert any('"Fifth Cultural Week"' in value for value in variants)
    assert any('"2002"' in value for value in variants)
    focused_or = variants[-1]
    assert '"which"' not in focused_or.lower()
    assert '"the"' not in focused_or.lower()


def test_query_variants_have_strict_and_broad_forms():
    variants = build_query_variants("Queen Arwa University established year")
    assert any(" AND " in value for value in variants)
    assert any(" OR " in value for value in variants)


def test_idf_variant_prioritizes_rare_terms_and_uses_shorter_conjunction():
    variants = build_query_variants(
        "common clue rare anchor another detail",
        {
            "common": 10000,
            "clue": 5000,
            "rare": 2,
            "anchor": 3,
            "another": 100,
            "detail": 200,
        },
    )
    strict = next(value for value in variants if " AND " in value)
    assert strict == '"rare" AND "anchor" AND "another"'


def test_rrf_search_returns_unique_limited_results_and_known_document(tmp_path):
    db_path = tmp_path / "corpus.sqlite"
    with closing(sqlite3.connect(db_path)) as conn, conn:
        conn.execute("CREATE VIRTUAL TABLE documents USING fts5(docid UNINDEXED, text, url UNINDEXED, tokenize='porter unicode61')")
        conn.executemany("INSERT INTO documents VALUES (?, ?, '')", [
            ("target", "The maritime archive records the museum collection."),
            *[(f"other-{n}", f"The archive catalogue lists collection number {n}.") for n in range(8)],
        ])
    retriever = BM25FullRetriever(db_path)
    try:
        results = retriever.search(
            "maritime archive museum collection",
            top_k=5,
            candidate_k=20,
        )
    finally:
        retriever.close()

    assert len(results) == 5
    assert len({item["docid"] for item in results}) == 5
    assert "target" in {item["docid"] for item in results}
    assert [item["rank"] for item in results] == [1, 2, 3, 4, 5]
    assert all(item["score"] > 0 for item in results)


def test_interleaving_protects_unique_results_from_both_rankings():
    left = [
        {"docid": "shared", "rank": 1},
        {"docid": "left", "rank": 2},
    ]
    right = [
        {"docid": "shared", "rank": 1},
        {"docid": "right", "rank": 2},
    ]
    merged = BM25FullRetriever._interleave_rankings([left, right], 3)
    assert [item["docid"] for item in merged] == ["shared", "left", "right"]
    assert [item["rank"] for item in merged] == [1, 2, 3]


def test_search_tool_uses_configured_result_count():
    assert search_documents.args["top_k"]["default"] == settings.retrieval_top_k


def test_long_query_uses_indexed_terms_after_the_first_thirty_words(tmp_path):
    db_path = tmp_path / "corpus.sqlite"
    with closing(sqlite3.connect(db_path)) as connection, connection:
        connection.execute(
            "CREATE VIRTUAL TABLE documents USING fts5("
            "docid UNINDEXED, text, url UNINDEXED, tokenize='porter unicode61')"
        )
        connection.executemany("INSERT INTO documents VALUES (?, ?, '')", [
            ("distractor", "astronomy research observations"),
            ("target", "The catalogue lists asteroids."),
        ])
    filler = " ".join(f"missing{number}" for number in range(35))
    with closing(BM25FullRetriever(db_path)) as retriever:
        assert retriever.search(f"{filler} asteroid", top_k=1)[0]["docid"] == "target"
