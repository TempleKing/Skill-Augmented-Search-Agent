import pytest
from pydantic import ValidationError

from config import Settings,settings
from retrieval.bge_hybrid import fuse_rankings

def test_rrf_uses_rank_not_incompatible_raw_scores():
    result=fuse_rankings([{'docid':'a','score':-999,'snippet':'sparse'},{'docid':'b','score':-1}],
                         [{'docid':'b','score':.99},{'docid':'c','score':.98}],3)
    assert [r['docid'] for r in result]==['b','a','c']
    assert result[0]['score']==pytest.approx(1/62+1/61)
    assert result[1]['snippet']=='sparse'

def test_duplicate_passages_do_not_add_rrf_votes():
    result=fuse_rankings([], [{'docid':'a'},{'docid':'a'},{'docid':'b'}],3)
    assert len(result)==2
    assert result[0]['score']==pytest.approx(1/61)

def test_hybrid_is_dispatched_through_existing_tool(monkeypatch):
    import retrieval.bge_hybrid as hybrid
    from agent.tool_registry import search_documents
    expected=[{'docid':'42','rank':1,'score':.02,'snippet':'true text','url':'https://example.org'}]
    monkeypatch.setattr(settings,'retrieval_mode','bm25_bge')
    monkeypatch.setattr(hybrid,'search_hybrid',lambda query,top_k: expected if query=='clue' and top_k==8 else [])
    assert search_documents.invoke({'query':'clue','top_k':8})==expected

def test_unimplemented_modes_are_rejected():
    with pytest.raises(ValidationError):
        Settings(retrieval_mode='auto')
