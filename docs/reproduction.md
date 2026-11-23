# Reproduce the search agent

Follow the [README](../README.md) for Python, credentials, corpus preparation, SQLite FTS5, and the frontend build. `requirements.txt` covers the baseline. Set `RETRIEVAL_MODE=bm25` for the lightest setup. If Python cannot create an FTS5 virtual table, use a Python distribution built with SQLite FTS5. The full corpus and SQLite together occupied about 7.8 GB on the reference machine. Allow extra space for downloads and SQLite temporary files.

## Prepare dense retrieval

Install the optional requirements in the same virtual environment. The builder fetches pinned model files from their original repositories and encodes the local corpus. The existing dense runtime also accepts locally held compatible bundles with their `verified.json` certificates. There is no hosted project index.

```powershell
$python = '.\.venv\Scripts\python.exe'
& $python -m pip install -r requirements-hybrid.txt
& $python scripts/prepare_dense.py bge --device cpu --batch-size 16
& $python scripts/prepare_dense.py qwen --device cpu --batch-size 16
```

Build only the encoder required by your mode. Qwen indexing also obtains the BGE tokenizer to reproduce the same passage boundaries. A CUDA-enabled PyTorch installation can accelerate encoding using `--device cuda`. CPU builds, particularly Qwen, can take substantially longer. Lower `--batch-size` if memory is limited.

| Encoder | Pinned revision | Dimensions | Vector file size at reference count |
| --- | --- | ---: | ---: |
| BAAI/bge-small-en-v1.5 | `5c38ec7c405ec4b44b94cc5a9bb96e735b38267a` | 384 | ~535 MB |
| Qwen/Qwen3-Embedding-0.6B | `97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3` | 1024 | ~1.43 GB |

Each full index has 348,411 normalized float32 vectors in an IndexFlatIP index and covers 100,195 document IDs. The builder verifies the SQLite document text against the local corpus and saves vectors, character offsets, document metadata, a model manifest, and SHA256 checksums. Qwen uses last-token pooling, an instruction on queries, and plain passages for document encoding. Model weights, vector memory, and temporary files require additional resources. The full build is compute-intensive. Local regeneration establishes compatible assets and can differ numerically from the historical vectors due to device, dtype, and library versions.

An existing bundle must match the certificate's SQLite SHA256 and its model revision. A fresh SQLite build can have a different file hash even with identical text. For that setup, build against the fresh SQLite database so the new certificate covers your files. If the corpus yields a different passage count, the builder stops to prevent accidental substitution.

```powershell
$env:RETRIEVAL_MODE = 'bm25_bge_qwen'
& $python main.py
```

For a local evaluation plan with no API requests, supply your own approved validation file. `--plan-only` prints the selected IDs and questions.

```powershell
& $python scripts/evaluate_agent.py --dataset data/validation.jsonl --limit 1 --no-shuffle --plan-only
```

See [configuration](configuration.md), [evaluation](evaluation.md), and the [technical report](../Technical_Report.pdf) for detailed behavior and observed results.
