"""Optional local BGE adapter; original BM25 ranking is left intact."""
from contextlib import closing
from functools import lru_cache
import hashlib
import json
from pathlib import Path
import sqlite3
import threading

from config import settings
from retrieval.bm25_full_retriever import BM25FullRetriever

REVISION = "5c38ec7c405ec4b44b94cc5a9bb96e735b38267a"
PREFIX = "Represent this sentence for searching relevant passages: "
_LOAD_LOCK = threading.Lock()

def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream,'sha256').hexdigest()

def fuse_rankings(bm25, dense, top_k, rrf_k=60, bm25_weight=1.0, dense_weight=1.0):
    """Document-level weighted RRF, never addition of raw retrieval scores."""
    scores, records = {}, {}
    for ranking,weight in [(bm25,bm25_weight),(dense,dense_weight)]:
        seen=set()
        for rank,item in enumerate(ranking,1):
            docid=str(item['docid'])
            if docid in seen:
                continue
            seen.add(docid)
            scores[docid]=scores.get(docid,0)+weight/(rrf_k+rank)
            # Preserve original BM25 snippet if both branches hit a document.
            records.setdefault(docid,item)
    order=sorted(scores,key=lambda d:(-scores[d],d))[:top_k]
    return [{**records[d],'rank':rank,'score':scores[d]} for rank,d in enumerate(order,1)]

class BGERetriever:
    def __init__(self,index_path,model_path,db_path,corpus_path,device):
        import numpy as np
        import faiss
        import torch
        from sentence_transformers import SentenceTransformer
        self.np=np
        root,model=Path(index_path),Path(model_path)
        self.db=Path(db_path)
        cert=json.loads((root/'verified.json').read_text(encoding='utf-8'))
        for directory,key in [(root,'files'),(model,'model_files')]:
            for name,expected in cert[key].items():
                if digest(directory/name)!=expected:
                    raise ValueError('BGE artifact checksum mismatch: '+name)
        if digest(self.db)!=cert['sqlite']['sha256']:
            raise ValueError('BGE certificate does not match the current SQLite corpus')
        for name,identity in cert['corpus_files'].items():
            p=Path(corpus_path)/name
            if p.stat().st_size!=identity['size'] or p.stat().st_mtime_ns!=identity['mtime_ns']:
                if digest(p)!=identity['sha256']:
                    raise ValueError('BGE corpus shard changed: '+name)
        self.manifest=json.loads((root/'manifest.json').read_text(encoding='utf-8'))
        cfg=self.manifest['config']
        if cfg['model']!='BAAI/bge-small-en-v1.5' or cfg['revision']!=REVISION or cfg['query_prefix']!=PREFIX or cfg['dimension']!=384 or not cfg['normalized'] or cfg['max_chunks_per_document']!=4:
            raise ValueError('Unsupported BGE index configuration')
        if (model/'revision.txt').read_text().strip()!=REVISION:
            raise ValueError('BGE model revision mismatch')
        faiss.omp_set_num_threads(8)
        self.index=faiss.read_index(str(root/'vectors.faiss'))
        self.chunks=np.load(root/'chunks.npy',mmap_mode='r',allow_pickle=False)
        self.documents=json.loads((root/'documents.json').read_text(encoding='utf-8'))
        if not isinstance(self.index,faiss.IndexFlatIP) or self.index.d!=384 or self.index.ntotal!=348411 or self.chunks.shape!=(348411,3) or len(self.documents)!=100195:
            raise ValueError('BGE vector/mapping dimensions or counts differ')
        for start in range(0,self.index.ntotal,8192):
            vectors=self.index.reconstruct_n(start,min(8192,self.index.ntotal-start))
            if not np.isfinite(vectors).all() or not np.allclose(np.linalg.norm(vectors,axis=1),1,atol=2e-4):
                raise ValueError('BGE index contains non-normalized vectors')
        torch.set_num_threads(min(8,torch.get_num_threads()))
        self.encoder=SentenceTransformer(str(model),device=device,local_files_only=True)
        self.encoder.max_seq_length=256
        self.encode_lock=threading.Lock()

    def query_vector(self,query):
        import faiss
        with self.encode_lock:
            vector=self.encoder.encode([PREFIX+query],normalize_embeddings=True,show_progress_bar=False,convert_to_numpy=True)
        vector=self.np.ascontiguousarray(vector,dtype='float32')
        if vector.shape!=(1,384) or not self.np.isfinite(vector).all():
            raise ValueError('Invalid BGE query vector')
        faiss.normalize_L2(vector)
        return vector

    def search(self,query,top_k=50):
        if top_k<=0:return []
        vector=self.query_vector(query)
        scores,positions=self.index.search(vector,min(self.index.ntotal,top_k*4))
        results=[];seen=set()
        with closing(sqlite3.connect(self.db.resolve().as_uri()+'?mode=ro',uri=True)) as con:
            for score,position in zip(scores[0],positions[0]):
                if position<0:continue
                doc_index,start,end=map(int,self.chunks[int(position)])
                metadata=self.documents[doc_index];docid=metadata['docid']
                if docid in seen:continue
                # Resolve by actual ID, never by old Arrow or SQLite row order.
                row=con.execute('SELECT d.docid,d.url,length(d.text),substr(d.text,?,?) FROM doc_locator l JOIN documents d ON d.rowid=l.document_rowid WHERE l.docid=?',(start+1,end-start,docid)).fetchone()
                if not row or row[0]!=docid or (row[1] or '')!=metadata['url'] or not 0<=start<end<=row[2]:
                    raise ValueError('BGE passage does not match current corpus: '+str(docid))
                seen.add(docid)
                results.append({'docid':docid,'rank':len(results)+1,'score':float(score),'dense_score':float(score),'snippet':row[3],'url':row[1],'start_offset':start,'end_offset':end,'passage':{'start':start,'end':end,'chunk_id':int(position)}})
                if len(results)>=top_k:break
        return results

@lru_cache(maxsize=1)
def _cached_dense(index,model,db,corpus,device):
    return BGERetriever(index,model,db,corpus,device)

def get_dense():
    with _LOAD_LOCK:
        return _cached_dense(str(settings.resolve_path(settings.bge_index_path)),str(settings.resolve_path(settings.bge_model_path)),str(settings.resolve_path(settings.bm25_db_path)),str(settings.resolve_path(settings.corpus_path)),settings.bge_device)

def search_hybrid(query,top_k):
    if top_k<=0:return []
    depth=max(top_k,settings.hybrid_candidate_k)
    with closing(BM25FullRetriever(settings.resolve_path(settings.bm25_db_path))) as bm25:
        sparse=bm25.search(query,top_k=depth)
    dense=get_dense().search(query,top_k=depth)
    return fuse_rankings(sparse,dense,top_k,settings.hybrid_rrf_k,settings.hybrid_bm25_weight,settings.hybrid_bge_weight)
