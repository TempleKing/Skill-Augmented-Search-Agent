import re
import sqlite3
from collections import defaultdict
from pathlib import Path

from config import settings


DB_PATH = settings.resolve_path(settings.bm25_db_path)
TOKEN_PATTERN = re.compile(r"[^\W_]+(?:['’-][^\W_]+)*", re.UNICODE)
STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "been", "but", "by",
    "did", "do", "does", "for", "from", "had", "has", "have", "he",
    "her", "his", "how", "i", "in", "into", "is", "it", "its", "of",
    "on", "or", "our", "she", "that", "the", "their", "them", "there",
    "these", "they", "this", "those", "to", "was", "were", "what",
    "when", "where", "which", "who", "whom", "whose", "why", "with",
}


def _quoted(value: str) -> str:
    return f'"{value.replace(chr(34), "")}"'


def _tokens(query: str) -> list[str]:
    return TOKEN_PATTERN.findall(query)


def _entity_phrases(query: str) -> list[str]:
    """Extract useful capitalized 2-4 token phrases without an NLP dependency."""
    words = _tokens(query)
    runs: list[list[str]] = []
    current: list[str] = []
    for word in words:
        is_entity_word = word[:1].isupper() and word.casefold() not in STOPWORDS
        if is_entity_word:
            current.append(word)
        else:
            if len(current) >= 2:
                runs.append(current)
            current = []
    if len(current) >= 2:
        runs.append(current)

    phrases: list[str] = []
    for run in runs:
        # Long title-like runs are represented by overlapping windows. This
        # avoids requiring an entire noisy question fragment to occur verbatim.
        width = min(4, len(run))
        for start in range(0, len(run) - width + 1):
            phrases.append(" ".join(run[start:start + width]))
        if len(run) > 2:
            phrases.append(" ".join(run[-3:]))
        phrases.append(" ".join(run[:2]))
    return list(dict.fromkeys(phrases))[:4]


def build_query_variants(
    query: str,
    document_frequencies: dict[str, int] | None = None,
) -> list[str]:
    """Build strict and broad FTS5 queries for recall-oriented fusion."""
    raw_tokens = _tokens(query)[:30]
    focused = [word for word in raw_tokens if word.casefold() not in STOPWORDS]
    focused = list(dict.fromkeys(focused))[:20]
    if not focused:
        focused = raw_tokens
    if not focused:
        return []

    if document_frequencies:
        original_order = {word: index for index, word in enumerate(focused)}
        focused = sorted(
            focused,
            key=lambda word: (
                document_frequencies.get(word.casefold(), 10**9),
                original_order[word],
            ),
        )

    variants: list[str] = []
    phrases = _entity_phrases(query)
    if phrases:
        # Phrase matches receive their own ranking rather than being drowned
        # out by common terms in the broad query.
        variants.append(" OR ".join(_quoted(phrase) for phrase in phrases))

    # A moderately strict query rewards documents that connect several clues.
    strict_width = 3 if document_frequencies else 8
    strict_terms = focused[: min(strict_width, len(focused))]
    if len(strict_terms) >= 2:
        variants.append(" AND ".join(_quoted(word) for word in strict_terms))

    broad_terms = focused[:6] if document_frequencies else focused
    variants.append(" OR ".join(_quoted(word) for word in broad_terms))
    return list(dict.fromkeys(variants))


class BM25FullRetriever:
    def __init__(self, db_path=DB_PATH):
        path = Path(db_path).resolve()
        self.conn = sqlite3.connect(path.as_uri() + "?mode=ro", uri=True, timeout=15)
        self._frequency_lookup_ready = False

    @staticmethod
    def prepare_query(query: str) -> str:
        """Return the broad cleaned query for backward compatibility."""
        variants = build_query_variants(query)
        return variants[-1] if variants else ""

    def _search_expression(self, expression: str, limit: int) -> list[dict]:
        rows = self.conn.execute(
            """
            SELECT
                docid,
                rank AS score,
                snippet(documents, 1, '[', ']', '...', 50) AS snippet,
                url
            FROM documents
            WHERE documents MATCH ?
            ORDER BY rank
            LIMIT ?
            """,
            (expression, limit),
        ).fetchall()
        return [
            {
                "rank": rank,
                "docid": row[0],
                "bm25_score": float(row[1]),
                "snippet": row[2],
                "url": row[3],
            }
            for rank, row in enumerate(rows, 1)
        ]

    def _document_frequencies(self, query: str) -> dict[str, int]:
        frequencies = {}
        for word in list(dict.fromkeys(_tokens(query)))[:30]:
            if word.casefold() in STOPWORDS:
                continue
            try:
                count = self.conn.execute(
                    "SELECT COUNT(*) FROM documents WHERE documents MATCH ?",
                    (_quoted(word),),
                ).fetchone()[0]
            except sqlite3.OperationalError:
                count = 10**9
            frequencies[word.casefold()] = int(count)
        return frequencies

    def indexed_term_frequencies(self, words: list[str]) -> dict[str, int]:
        """Look up document frequencies with the index's own tokenizer."""
        if not self._frequency_lookup_ready:
            self.conn.execute(
                "CREATE VIRTUAL TABLE temp.bm25_vocab "
                "USING fts5vocab(main, documents, row)"
            )
            self.conn.execute(
                "CREATE VIRTUAL TABLE temp.bm25_terms "
                "USING fts5(term, tokenize='porter unicode61')"
            )
            self.conn.execute(
                "CREATE VIRTUAL TABLE temp.bm25_term_vocab "
                "USING fts5vocab(bm25_terms, row)"
            )
            self._frequency_lookup_ready = True

        frequencies = {}
        with self.conn:
            for word in words:
                if word.isalnum():
                    self.conn.execute("DELETE FROM temp.bm25_terms")
                    self.conn.execute("INSERT INTO temp.bm25_terms VALUES (?)", (word,))
                    row = self.conn.execute(
                        "SELECT doc FROM temp.bm25_vocab "
                        "WHERE term = (SELECT term FROM temp.bm25_term_vocab)"
                    ).fetchone()
                    count = row[0] if row else 0
                else:
                    count = self.conn.execute(
                        "SELECT count(*) FROM documents WHERE documents MATCH ?",
                        (_quoted(word),),
                    ).fetchone()[0]
                if count:
                    frequencies[word] = count
        return frequencies

    def _search_full_question(self, words: list[str], top_k: int) -> list[dict]:
        """Build a BM25 query from the rarest indexed terms across the whole question."""
        frequencies = self.indexed_term_frequencies(words)
        terms = sorted((count, word) for word, count in frequencies.items())[:12]
        expression = " OR ".join(_quoted(word) for _, word in terms)
        if not expression:
            return []
        return [
            {**row, "score": 1.0 / (settings.retrieval_rrf_k + row["rank"])}
            for row in self._search_expression(expression, top_k)
        ]

    def _search_variants(
        self,
        variants: list[str],
        top_k: int,
        candidate_k: int | None,
    ) -> list[dict]:
        if not variants or top_k <= 0:
            return []
        pool_size = max(top_k, candidate_k or settings.retrieval_candidate_k)
        fused_scores: dict[str, float] = defaultdict(float)
        best: dict[str, dict] = {}
        best_variant_rank: dict[str, int] = {}

        for expression in variants:
            for item in self._search_expression(expression, pool_size):
                docid = str(item["docid"])
                rank = int(item["rank"])
                fused_scores[docid] += 1.0 / (settings.retrieval_rrf_k + rank)
                if docid not in best or rank < best_variant_rank[docid]:
                    best[docid] = item
                    best_variant_rank[docid] = rank

        ordered = sorted(
            best,
            key=lambda docid: (-fused_scores[docid], best_variant_rank[docid], docid),
        )[:top_k]
        return [
            {
                "rank": rank,
                "docid": docid,
                "score": fused_scores[docid],
                "bm25_score": best[docid]["bm25_score"],
                "snippet": best[docid]["snippet"],
                "url": best[docid]["url"],
            }
            for rank, docid in enumerate(ordered, 1)
        ]

    def search_fixed_order(
        self,
        query: str,
        top_k: int = 5,
        candidate_k: int | None = None,
    ) -> list[dict]:
        """Fuse the original-token-order query variants for one retrieval view."""
        return self._search_variants(
            build_query_variants(query), top_k, candidate_k
        )


    def search_idf_order(
        self,
        query: str,
        top_k: int = 5,
        candidate_k: int | None = None,
    ) -> list[dict]:
        """Rank query terms by corpus rarity for an independent retrieval view."""
        frequencies = self._document_frequencies(query)
        variants = build_query_variants(query, frequencies)
        return self._search_variants(variants, top_k, candidate_k)

    @staticmethod
    def _interleave_rankings(
        rankings: list[list[dict]],
        top_k: int,
    ) -> list[dict]:
        merged = []
        seen = set()
        depth = 0
        while len(merged) < top_k:
            made_progress = False
            for ranking in rankings:
                if depth >= len(ranking):
                    continue
                made_progress = True
                item = ranking[depth]
                docid = str(item["docid"])
                if docid in seen:
                    continue
                seen.add(docid)
                merged.append({**item, "rank": len(merged) + 1})
                if len(merged) >= top_k:
                    break
            if not made_progress:
                break
            depth += 1
        return merged

    def search_interleaved(
        self,
        query: str,
        top_k: int = 5,
        candidate_k: int | None = None,
    ) -> list[dict]:
        """Protect diversity by alternating positional and IDF rankings."""
        fixed = self.search_fixed_order(query, top_k, candidate_k)
        idf = self.search_idf_order(query, top_k, candidate_k)
        return self._interleave_rankings([fixed, idf], top_k)

    def search(self, query: str, top_k: int = 5, candidate_k: int | None = None):
        """Use full-question IDF for long inputs and preserve focused search."""
        words = list(dict.fromkeys(
            word.casefold() for word in _tokens(query)
            if len(word) > 1 and word.casefold() not in STOPWORDS
        ))
        if len(words) > 12 and top_k > 0:
            rare = self._search_full_question(words, top_k)
            if len(rare) == top_k:
                return rare
            baseline = self.search_interleaved(query, top_k, candidate_k)
            seen = {row["docid"] for row in rare}
            merged = [*rare, *(row for row in baseline if row["docid"] not in seen)]
            return [{**row, "rank": rank} for rank, row in enumerate(merged[:top_k], 1)]
        return self.search_interleaved(query, top_k, candidate_k)

    def count(self):
        return self.conn.execute("SELECT COUNT(*) FROM documents").fetchone()[0]

    def close(self):
        self.conn.close()
