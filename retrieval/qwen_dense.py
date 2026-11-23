"""Pinned instruction-aware Qwen embeddings over the certified BGE boundaries."""
from contextlib import closing
from functools import lru_cache
import json
from pathlib import Path
import threading

from config import settings
from retrieval.bge_hybrid import BGERetriever,digest,get_dense
from retrieval.bm25_full_retriever import BM25FullRetriever

REVISION='97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3'
TASK='Given a web search query, retrieve relevant passages that answer the query'
MAX_TOKENS=8192
_LOCK=threading.Lock()

def format_query(query):
    return f'Instruct: {TASK}\nQuery:{query}'

def fuse_document_rankings(rankings,top_k,rrf_k=60):
    scores={};records={};sources={}
    for ranking in rankings:
        seen=set()
        for rank,item in enumerate(ranking,1):
            docid=str(item['docid'])
            if docid in seen:continue
            seen.add(docid);scores[docid]=scores.get(docid,0)+1/(rrf_k+rank)
            records.setdefault(docid,item)
            sources.setdefault(docid,[]).append(item.get('retrieval_source','unspecified'))
    return [{**records[d],'rank':rank,'score':scores[d],'retrieval_sources':sources[d]} for rank,d in enumerate(sorted(scores,key=lambda d:(-scores[d],d))[:top_k],1)]

class QwenRetriever(BGERetriever):
    """Reuse only ID resolution/passage reading; never BGE's encoder or vectors."""
    def __init__(self,index_path,model_path,db_path,device):
        import numpy as np
        import faiss
        import torch
        from sentence_transformers import SentenceTransformer
        self.np=np;root=Path(index_path);model=Path(model_path);self.db=Path(db_path)
        cert=json.loads((root/'verified.json').read_text(encoding='utf-8'))
        if not cert.get('complete'):raise ValueError('Incomplete Qwen index')
        for directory,key in [(root,'files'),(model,'model_files')]:
            for name,expected in cert[key].items():
                if digest(directory/name)!=expected:raise ValueError('Qwen checksum mismatch: '+name)
        if digest(self.db)!=cert['sqlite']['sha256']:raise ValueError('Qwen SQLite corpus mismatch')
        self.manifest=json.loads((root/'manifest.json').read_text(encoding='utf-8'));cfg=self.manifest['config']
        if (cfg['model']!='Qwen/Qwen3-Embedding-0.6B' or cfg['revision']!=REVISION or cfg['dimension']!=1024 or cfg['pooling']!='last_token' or not cfg['normalized'] or cfg['document_instruction']!='' or cfg['task']!=TASK or cfg['query_template']!='Instruct: {task}\nQuery:{query}' or cfg['max_tokens']!=MAX_TOKENS):
            raise ValueError('Qwen encoding configuration mismatch')
        if (model/'revision.txt').read_text().strip()!=REVISION:raise ValueError('Qwen revision mismatch')
        faiss.omp_set_num_threads(8);torch.set_num_threads(min(8,torch.get_num_threads()))
        self.index=faiss.read_index(str(root/'vectors.faiss'))
        self.chunks=np.load(root/'chunks.npy',mmap_mode='r',allow_pickle=False)
        self.documents=json.loads((root/'documents.json').read_text(encoding='utf-8'))
        if not isinstance(self.index,faiss.IndexFlatIP) or self.index.d!=1024 or self.index.ntotal!=348411 or self.chunks.shape!=(348411,3) or len(self.documents)!=100195:
            raise ValueError('Qwen index dimensions or mapping counts differ')
        for start in range(0,self.index.ntotal,8192):
            v=self.index.reconstruct_n(start,min(8192,self.index.ntotal-start))
            if not np.isfinite(v).all() or not np.allclose(np.linalg.norm(v,axis=1),1,atol=2e-4):raise ValueError('Non-normalized Qwen vectors')
        self.encoder=SentenceTransformer(str(model),device=device,local_files_only=True,
            model_kwargs={'torch_dtype':torch.float32,'attn_implementation':'sdpa'},tokenizer_kwargs={'padding_side':'left'})
        self.encoder.max_seq_length=MAX_TOKENS;self.encoder[0].auto_model.config.use_cache=False
        if self.encoder.get_sentence_embedding_dimension()!=1024 or not self.encoder[1].pooling_mode_lasttoken:
            raise ValueError('Qwen pooling or dimension differs')
        self.encode_lock=threading.Lock()

    def encode_texts(self,texts,batch_size=8):
        import faiss
        with self.encode_lock:
            v=self.encoder.encode(texts,batch_size=batch_size,prompt='',normalize_embeddings=True,convert_to_numpy=True,show_progress_bar=False)
        v=self.np.ascontiguousarray(v,dtype='float32')
        if v.shape!=(len(texts),1024) or not self.np.isfinite(v).all():raise ValueError('Invalid Qwen embeddings')
        faiss.normalize_L2(v)
        return v

    def query_vector(self,query):
        return self.encode_texts([format_query(query)],1)

@lru_cache(maxsize=1)
def _cached(index,model,db,device):return QwenRetriever(index,model,db,device)

def get_qwen():
    with _LOCK:
        return _cached(str(settings.resolve_path(settings.qwen_index_path)),str(settings.resolve_path(settings.qwen_model_path)),str(settings.resolve_path(settings.bm25_db_path)),settings.qwen_device)

def search_qwen_hybrid(query,top_k):
    if top_k<=0:return []
    depth=max(top_k,settings.hybrid_candidate_k)
    with closing(BM25FullRetriever(settings.resolve_path(settings.bm25_db_path))) as bm25:
        rankings=[[{**x,'retrieval_source':'bm25'} for x in bm25.search(query,top_k=depth)]]
    if settings.retrieval_mode=='bm25_bge_qwen':rankings.append([{**x,'retrieval_source':'bge'} for x in get_dense().search(query,depth)])
    rankings.append([{**x,'retrieval_source':'qwen'} for x in get_qwen().search(query,depth)])
    return fuse_document_rankings(rankings,top_k,settings.hybrid_rrf_k)
