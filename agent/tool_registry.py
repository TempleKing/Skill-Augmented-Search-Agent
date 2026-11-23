"""The baseline exposes one BM25 search tool and two document tools."""
from contextlib import closing

from langchain_core.tools import tool
from pydantic import BaseModel, Field

from config import settings
from retrieval.bm25_full_retriever import BM25FullRetriever
from tools.document_tools import DOCUMENT_TOOLS


class SearchDocumentsInput(BaseModel):
    query: str = Field(min_length=1, description="Focused keyword search query")
    top_k: int = Field(
        default=settings.retrieval_top_k, ge=1, le=20,
        description="Number of ranked documents. Omit to use the baseline default of eight.",
    )


@tool(args_schema=SearchDocumentsInput)
def search_documents(query: str, top_k: int = settings.retrieval_top_k) -> list[dict]:
    """Search BrowseComp-Plus using BM25 keyword variants and reciprocal rank fusion.

    Returns ranked document IDs, retrieval scores, evidence snippets and source URLs.
    Use this tool to find documents relevant to a question or unresolved clue.
    """
    # Each invocation owns its connection so concurrent evaluations remain safe.
    if settings.retrieval_mode in {"bm25_qwen", "bm25_bge_qwen"}:
        from retrieval.qwen_dense import search_qwen_hybrid
        return search_qwen_hybrid(query, top_k)
    if settings.retrieval_mode == "bm25_bge":
        from retrieval.bge_hybrid import search_hybrid
        return search_hybrid(query, top_k)
    with closing(BM25FullRetriever(settings.resolve_path(settings.bm25_db_path))) as retriever:
        return retriever.search(query=query, top_k=top_k)


ALL_TOOLS = [search_documents, *DOCUMENT_TOOLS]
