"""Download the fixed retrieval corpus and save it in Dataset disk format."""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import settings


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=settings.resolve_path(settings.corpus_path))
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"{output} already exists. Reuse it or choose a new --output directory.")

    from datasets import load_dataset

    corpus = load_dataset("Tevatron/browsecomp-plus-corpus", split="train")
    if len(corpus) != 100195:
        raise ValueError(f"Expected 100195 documents, got {len(corpus)}.")
    for field in ("docid", "text", "url"):
        if field not in corpus.column_names:
            raise ValueError(f"Missing required corpus field: {field}")
    output.parent.mkdir(parents=True, exist_ok=True)
    corpus.save_to_disk(str(output))
    print(f"Saved {len(corpus)} documents to {output}")


if __name__ == "__main__":
    main()
