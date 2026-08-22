import argparse

from src.core.config import load_settings
from src.retrieve.pipeline import SearchPipeline


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("query", type=str)
    parser.add_argument("--top_k", type=int, default=None)
    args = parser.parse_args()

    settings = load_settings()
    pipeline = SearchPipeline(settings)

    results = pipeline.search(args.query, top_k=args.top_k)

    print(f"query: {args.query}\n")
    for r in results:
        print(f"- rerank_score={r['rerank_score']:.4f} | {r['doc_id']} p.{r['page']}")
        print(f"  {r['text'][:150]}")
        print()


if __name__ == "__main__":
    main()
