"""
RAG 검색 파이프라인의 구간별 소요 시간 측정: 임베딩 / 하이브리드 검색 / 재랭킹.
eval/qa_set.jsonl의 질문 일부를 실제 파이프라인에 그대로 흘려보내 측정한다.
"""

import argparse
import json
import time
from pathlib import Path
from statistics import mean, median

from src.core.config import load_settings
from src.embedding.embedder import Embedder
from src.index.bm25_store import BM25Store
from src.index.vector_store import FaissVectorStore
from src.rerank.cross_encoder import CrossEncoderReranker
from src.retrieve.hybrid import HybridRetriever


def load_queries(path: str, n: int) -> list[str]:
    queries = []
    with Path(path).open(encoding="utf-8") as f:
        for line in f:
            queries.append(json.loads(line)["question"])
    return queries[:n]


def summarize(name: str, values: list[float]):
    print(f"{name:10s} 평균={mean(values):.3f}s  중앙값={median(values):.3f}s  "
          f"최소={min(values):.3f}s  최대={max(values):.3f}s")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=20, help="측정에 사용할 질문 개수")
    parser.add_argument("--qa_set_path", type=str, default="eval/qa_set.jsonl")
    args = parser.parse_args()

    settings = load_settings()
    queries = load_queries(args.qa_set_path, args.n)
    print(f"측정 질문 수: {len(queries)}\n")

    embedder = Embedder(settings.embedding_model_name)
    vector_store = FaissVectorStore(settings.faiss_index_dir)
    vector_store.load()
    bm25_store = BM25Store(settings.bm25_index_dir)
    bm25_store.load()
    retriever = HybridRetriever(vector_store, bm25_store)
    reranker = CrossEncoderReranker(settings.reranker_model_name)

    embed_times, search_times, rerank_times, total_times = [], [], [], []

    for q in queries:
        t0 = time.perf_counter()
        query_vector = embedder.encode([q])[0]
        t1 = time.perf_counter()

        candidates = retriever.search(
            q, query_vector,
            top_k=settings.rerank_candidate_k,
            candidate_k=settings.rerank_candidate_k,
            rrf_k=settings.rrf_k,
            vector_weight=settings.vector_weight,
            bm25_weight=settings.bm25_weight,
        )
        t2 = time.perf_counter()

        reranker.rerank(q, candidates, top_k=settings.top_k)
        t3 = time.perf_counter()

        embed_times.append(t1 - t0)
        search_times.append(t2 - t1)
        rerank_times.append(t3 - t2)
        total_times.append(t3 - t0)

        print(f"[{q[:24]:24s}] embed={t1-t0:.3f}s  search={t2-t1:.3f}s  "
              f"rerank={t3-t2:.3f}s  total={t3-t0:.3f}s")

    print()
    summarize("임베딩", embed_times)
    summarize("하이브리드검색", search_times)
    summarize("재랭킹", rerank_times)
    summarize("전체", total_times)


if __name__ == "__main__":
    main()
