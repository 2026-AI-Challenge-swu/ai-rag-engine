"""
naive(벡터만) vs hybrid(현재 storage/bm25/bm25.pkl에 저장된 BM25 인덱스)를
현재 평가셋(eval/qa_set.jsonl)으로 비교.

BM25 토큰화 전략을 바꿔가며(형태소만 / 형태소+n-gram 등) 인덱스를 재빌드한 뒤
이 스크립트를 다시 실행하면, 그때그때의 실제 디스크 인덱스 기준 결과가 나온다.
(메모리 시뮬레이션 없음 — 항상 디스크에 저장된 인덱스를 그대로 로드해서 평가)
"""

import json
from pathlib import Path

from src.core.config import load_settings
from src.embedding.embedder import Embedder
from src.evaluation.metrics import recall_at_k, reciprocal_rank
from src.index.bm25_store import BM25Store
from src.index.bm25_store import tokenize as bm25_tokenize
from src.index.vector_store import FaissVectorStore
from src.retrieve.rrf import reciprocal_rank_fusion

KS = [1, 3, 5]
VECTOR_WEIGHT = 1.0
BM25_WEIGHT = 1.0


def load_qa_set(path: str) -> list[dict]:
    items = []
    with Path(path).open(encoding="utf-8") as f:
        for line in f:
            items.append(json.loads(line))
    return items


def hybrid_search(query, query_vector, vector_store, bm25_store, top_k, candidate_k, rrf_k):
    vector_results = vector_store.search(query_vector, candidate_k)
    scores = bm25_store.bm25.get_scores(bm25_tokenize(query))
    ranked_idx = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:candidate_k]
    bm25_results = [{**bm25_store.metadatas[i], "score": float(scores[i])} for i in ranked_idx]
    fused = reciprocal_rank_fusion(
        [vector_results, bm25_results], weights=[VECTOR_WEIGHT, BM25_WEIGHT], k=rrf_k
    )
    return fused[:top_k]


def evaluate(name, qa_set, search_fn, embedder):
    recalls = {k: [] for k in KS}
    rrs = []
    for item in qa_set:
        query = item["question"]
        relevant = [(r["doc_id"], r["page"]) for r in item["relevant"]]
        query_vector = embedder.encode([query])[0]
        results = search_fn(query, query_vector)
        retrieved = [(r["doc_id"], r["page"]) for r in results]
        for k in KS:
            recalls[k].append(recall_at_k(retrieved, relevant, k))
        rrs.append(reciprocal_rank(retrieved, relevant))

    print(f"=== {name} ===")
    for k in KS:
        avg = sum(recalls[k]) / len(recalls[k])
        print(f"Recall@{k}: {avg:.3f}")
    print(f"MRR: {sum(rrs) / len(rrs):.3f}")
    print()


def main():
    settings = load_settings()
    qa_set = load_qa_set("eval/qa_set.jsonl")
    print(f"평가셋: eval/qa_set.jsonl ({len(qa_set)}문항)")
    print(f"하이브리드 가중치: vector={VECTOR_WEIGHT}, bm25={BM25_WEIGHT} (고정값)")
    print(f"BM25 인덱스: {settings.bm25_index_dir}/bm25.pkl (현재 디스크에 저장된 그대로)\n")

    embedder = Embedder(settings.embedding_model_name)

    vector_store = FaissVectorStore(settings.faiss_index_dir)
    vector_store.load()

    bm25_store = BM25Store(settings.bm25_index_dir)
    bm25_store.load()

    evaluate(
        "naive (벡터만)",
        qa_set,
        lambda q, qv: vector_store.search(qv, max(KS)),
        embedder,
    )

    evaluate(
        "hybrid",
        qa_set,
        lambda q, qv: hybrid_search(
            q, qv, vector_store, bm25_store,
            top_k=max(KS), candidate_k=settings.rerank_candidate_k, rrf_k=settings.rrf_k,
        ),
        embedder,
    )


if __name__ == "__main__":
    main()
