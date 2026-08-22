import argparse
import json
from pathlib import Path

from src.ingest.chunk import chunk_pages
from src.schemas.document import Chunk


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--processed_dir", type=str, default="data/processed")
    parser.add_argument("--out_path", type=str, default="data/chunks/chunks.jsonl")
    parser.add_argument("--chunk_size", type=int, default=500)
    parser.add_argument("--overlap", type=int, default=50)
    args = parser.parse_args()

    pages = []
    for path in sorted(Path(args.processed_dir).glob("*.jsonl")):
        with path.open(encoding="utf-8") as f:
            for line in f:
                pages.append(Chunk(**json.loads(line)))

    chunks = chunk_pages(pages, chunk_size=args.chunk_size, overlap=args.overlap)

    out_path = Path(args.out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for chunk in chunks:
            f.write(chunk.model_dump_json() + "\n")

    print(f"[done] {len(pages)} pages -> {len(chunks)} chunks -> {out_path}")


if __name__ == "__main__":
    main()
