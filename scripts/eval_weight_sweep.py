"""
가중치(vector_weight : bm25_weight) 5개 조합을 51문항 평가셋(운영 설정)으로 그리드서치.
BM25 인덱스는 kiwi 형태소 + bi/tri-gram 토큰화로 재빌드된 운영 인덱스를 그대로 사용.
"""

import json
from pathlib import Path

from src.core.config import load_settings
from src.embedding.embedder import Embedder
from src.evaluation.metrics import recall_at_k, reciprocal_rank
from src.index.bm25_store import BM25Store
from src.index.vector_store import FaissVectorStore
from src.retrieve.hybrid import HybridRetriever

KS = [1, 3, 5]
COMBOS = [(1.0, 1.0), (1.0, 0.3), (0.3, 1.0), (0.6, 0.4), (0.3, 0.7)]


def load_qa_set(path: str) -> list[dict]:
    items = []
    with Path(path).open(encoding="utf-8") as f:
        for line in f:
            items.append(json.loads(line))
    return items


def main():
    settings = load_settings()
    qa_set = load_qa_set("eval/qa_set.jsonl")
    print(f"평가셋: eval/qa_set.jsonl ({len(qa_set)}문항), BM25: kiwi+bi/tri-gram (재빌드된 운영 인덱스)\n")

    embedder = Embedder(settings.embedding_model_name)
    vector_store = FaissVectorStore(settings.faiss_index_dir)
    vector_store.load()
    bm25_store = BM25Store(settings.bm25_index_dir)
    bm25_store.load()
    retriever = HybridRetriever(vector_store, bm25_store)

    query_vectors = {}
    for item in qa_set:
        query_vectors[item["question"]] = embedder.encode([item["question"]])[0]

    results_summary = []
    for vw, bw in COMBOS:
        recalls = {k: [] for k in KS}
        rrs = []
        for item in qa_set:
            query = item["question"]
            relevant = [(r["doc_id"], r["page"]) for r in item["relevant"]]
            results = retriever.search(
                query,
                query_vectors[query],
                top_k=max(KS),
                candidate_k=settings.rerank_candidate_k,
                rrf_k=settings.rrf_k,
                vector_weight=vw,
                bm25_weight=bw,
            )
            retrieved = [(r["doc_id"], r["page"]) for r in results]
            for k in KS:
                recalls[k].append(recall_at_k(retrieved, relevant, k))
            rrs.append(reciprocal_rank(retrieved, relevant))

        avg_recalls = {k: sum(v) / len(v) for k, v in recalls.items()}
        avg_mrr = sum(rrs) / len(rrs)
        results_summary.append((vw, bw, avg_recalls, avg_mrr))

        print(f"=== vector={vw}, bm25={bw} ===")
        for k in KS:
            print(f"Recall@{k}: {avg_recalls[k]:.3f}")
        print(f"MRR: {avg_mrr:.3f}")
        print()

    best = max(results_summary, key=lambda r: r[3])
    print(f"[best by MRR] vector={best[0]}, bm25={best[1]} -> MRR={best[3]:.3f}")


if __name__ == "__main__":
    main()
