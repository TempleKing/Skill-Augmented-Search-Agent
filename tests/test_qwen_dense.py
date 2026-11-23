import pytest
from config import settings
from retrieval.qwen_dense import format_query,fuse_document_rankings

def test_official_query_instruction_template():
    assert format_query('clue')=='Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery:clue'

def test_three_way_document_rrf_and_duplicate_votes():
    result=fuse_document_rankings([[{'docid':'a','retrieval_source':'bm25'}],[{'docid':'b','retrieval_source':'bge'}],[{'docid':'a','retrieval_source':'qwen'},{'docid':'a','retrieval_source':'qwen'}]],2)
    assert [r['docid'] for r in result]==['a','b']
    assert result[0]['score']==pytest.approx(2/61)
    assert result[0]['retrieval_sources']==['bm25','qwen']

def test_qwen_dispatch_keeps_single_search_tool(monkeypatch):
    import retrieval.qwen_dense as qwen
    from agent.tool_registry import ALL_TOOLS,search_documents
    monkeypatch.setattr(settings,'retrieval_mode','bm25_qwen')
    monkeypatch.setattr(qwen,'search_qwen_hybrid',lambda query,top_k: [{'docid':'test','score':1}])
    assert search_documents.invoke({'query':'clue'})[0]['docid']=='test'
    assert [t.name for t in ALL_TOOLS]==['search_documents','open_document','find_in_document']
