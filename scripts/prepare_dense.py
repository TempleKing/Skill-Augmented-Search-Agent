"""Build local BGE or Qwen assets from the same corpus as the SQLite index."""

import argparse
from contextlib import closing
import json
import math
from pathlib import Path
import shutil
import sqlite3
import sys
from tempfile import TemporaryDirectory

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import settings
from retrieval.bge_hybrid import PREFIX, REVISION as BGE_REVISION, digest
from retrieval.qwen_dense import MAX_TOKENS, REVISION as QWEN_REVISION, TASK


def passage_spans(text, tokenizer):
    last = max(0, len(text) - 1600)
    count = min(math.ceil(last / 1200) + 1, 4)
    starts = [0] if count == 1 else [round(i * last / (count - 1)) for i in range(count)]
    for start in starts:
        offsets = tokenizer(
            text[start:start + 1600], add_special_tokens=False,
            truncation=True, max_length=254, return_offsets_mapping=True,
        )["offset_mapping"]
        if offsets:
            yield start + offsets[0][0], start + offsets[-1][1]


def get_model_files(name, revision, destination):
    from huggingface_hub import snapshot_download

    included = [
        "*.json", "vocab.txt", "merges.txt", "model.safetensors",
        "1_Pooling/config.json", "README.md", "LICENSE*", "NOTICE*",
    ]
    marker = destination / "revision.txt"
    if destination.exists() and not marker.exists():
        raise ValueError(f"Existing model directory has no revision marker: {destination}")
    if marker.exists():
        if marker.read_text(encoding="utf-8").strip() != revision:
            raise ValueError(f"Model revision mismatch: {destination}")
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=destination.parent) as temp:
        stage = Path(temp) / "model"
        snapshot_download(name, revision=revision, local_dir=stage, allow_patterns=included)
        marker = stage / "revision.txt"
        marker.write_text(revision + "\n", encoding="utf-8")
        stage.rename(destination)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("encoder", choices=["bge", "qwen"])
    parser.add_argument("--device", default="cpu", help="CPU or an available PyTorch CUDA device")
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()
    if args.batch_size <= 0:
        parser.error("--batch-size must be positive")

    import faiss
    import numpy as np
    import torch
    from datasets import load_from_disk
    from transformers import AutoTokenizer
    from sentence_transformers import SentenceTransformer

    resolve = settings.resolve_path
    qwen = args.encoder == "qwen"
    dest = resolve(settings.qwen_index_path if qwen else settings.bge_index_path)
    model_path = resolve(settings.qwen_model_path if qwen else settings.bge_model_path)
    corpus_path = resolve(settings.corpus_path)
    db = resolve(settings.bm25_db_path)
    if dest.exists():
        raise FileExistsError(f"Choose an unused index path: {dest}")
    ds = load_from_disk(str(corpus_path))
    if len(ds) != 100195 or len(set(ds["docid"])) != 100195:
        raise ValueError("Expected the 100,195 uniquely identified BrowseComp-Plus documents")
    bge_path = resolve(settings.bge_model_path)
    get_model_files("BAAI/bge-small-en-v1.5", BGE_REVISION, bge_path)
    span_tokenizer = AutoTokenizer.from_pretrained(str(bge_path), local_files_only=True)
    name, revision, dim = (("Qwen/Qwen3-Embedding-0.6B", QWEN_REVISION, 1024) if qwen
                           else ("BAAI/bge-small-en-v1.5", BGE_REVISION, 384))
    if qwen:
        get_model_files(name, revision, model_path)
    torch.set_num_threads(min(8, torch.get_num_threads()))
    options = dict(device=args.device, local_files_only=True)
    if qwen:
        options.update(model_kwargs={"attn_implementation": "sdpa"},
                       tokenizer_kwargs={"padding_side": "left"})
    model = SentenceTransformer(str(model_path), **options)
    model.max_seq_length = MAX_TOKENS if qwen else 256
    model[0].auto_model.config.use_cache = False
    if model.get_sentence_embedding_dimension() != dim:
        raise ValueError("Unexpected embedding dimension")
    if qwen and not model[1].pooling_mode_lasttoken:
        raise ValueError("Expected Qwen last-token pooling")

    index = faiss.IndexFlatIP(dim)
    docs, chunks, pending = [], [], []
    db_hash = digest(db)
    corpus_files = {}
    for entry in json.loads((corpus_path / "state.json").read_text(encoding="utf-8"))["_data_files"]:
        shard = corpus_path / entry["filename"]
        corpus_files[shard.name] = {
            "sha256": digest(shard), "size": shard.stat().st_size, "mtime_ns": shard.stat().st_mtime_ns,
        }

    def flush():
        if not pending:
            return
        if qwen:
            lengths = model.tokenizer(pending, truncation=False)["input_ids"]
            if any(len(ids) > MAX_TOKENS for ids in lengths):
                raise ValueError("A passage exceeds the configured Qwen token limit")
        vectors = model.encode(
            pending, batch_size=args.batch_size, prompt="",
            normalize_embeddings=True, show_progress_bar=False,
        )
        vectors = np.ascontiguousarray(vectors, dtype=np.float32)
        if vectors.shape != (len(pending), dim) or not np.isfinite(vectors).all():
            raise ValueError("Invalid embedding batch")
        faiss.normalize_L2(vectors)
        if not np.allclose(np.linalg.norm(vectors, axis=1), 1, atol=2e-4):
            raise ValueError("Invalid embedding norms")
        index.add(vectors)
        pending.clear()

    with closing(sqlite3.connect(db.as_uri() + "?mode=ro", uri=True)) as con:
        count = con.execute("SELECT count(*) FROM doc_locator").fetchone()[0]
        if count != len(ds):
            raise ValueError("Build the SQLite document locator first")
        for doc in ds:
            row = con.execute(
                "SELECT d.docid, d.text, d.url FROM doc_locator l JOIN documents d "
                "ON l.document_rowid=d.rowid WHERE l.docid=?", (doc["docid"],),
            ).fetchone()
            if row != (doc["docid"], doc["text"], doc["url"] or ""):
                raise ValueError(f"Corpus and SQLite differ for {doc['docid']}")
            spans = list(passage_spans(doc["text"], span_tokenizer))
            if not spans:
                raise ValueError(f"No passage for {doc['docid']}")
            for start, end in spans:
                chunks.append((len(docs), start, end))
                pending.append(doc["text"][start:end])
                if len(pending) == args.batch_size:
                    flush()
            docs.append({"docid": doc["docid"], "url": doc["url"] or ""})
            if len(docs) % 1000 == 0:
                print(f"Prepared {len(docs):,} documents", flush=True)
    flush()
    if index.ntotal != len(chunks) or len(chunks) != 348411:
        raise ValueError("Unexpected passage count for the reference corpus")
    if digest(db) != db_hash or any(digest(corpus_path / n) != h["sha256"] for n, h in corpus_files.items()):
        raise ValueError("Corpus or SQLite index changed during build")
    config = {
        "model": name, "revision": revision, "dimension": dim, "normalized": True,
        "index": "IndexFlatIP", "max_chunks_per_document": 4,
    }
    if qwen:
        config.update(pooling="last_token", document_instruction="", task=TASK,
                      query_template="Instruct: {task}\nQuery:{query}", max_tokens=MAX_TOKENS)
    else:
        config.update(query_prefix=PREFIX, max_tokens=256)

    dest.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=dest.parent) as temp:
        stage = Path(temp) / "assets"
        stage.mkdir()
        faiss.write_index(index, str(stage / "vectors.faiss"))
        np.save(stage / "chunks.npy", np.asarray(chunks, dtype=np.int64), allow_pickle=False)
        (stage / "documents.json").write_text(json.dumps(docs, ensure_ascii=False), encoding="utf-8")
        manifest = {"config": config, "document_count": len(docs), "passage_count": len(chunks)}
        (stage / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        certificate = {
            "complete": True, "model_revision": revision, "sqlite": {"sha256": db_hash},
            "files": {p.name: digest(p) for p in stage.iterdir()},
            "model_files": {p.relative_to(model_path).as_posix(): digest(p)
                            for p in model_path.rglob("*") if p.is_file() and ".cache" not in p.parts},
            "corpus_files": corpus_files,
        }
        (stage / "verified.json").write_text(json.dumps(certificate, indent=2), encoding="utf-8")
        stage.rename(dest)
    print(f"Saved {len(chunks):,} {dim}-dimensional vectors in {dest}")


if __name__ == "__main__":
    main()
