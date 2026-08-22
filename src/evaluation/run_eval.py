import json
from pathlib import Path

from src.core.config import load_settings
from src.embedding.embedder import Embedder
from src.evaluation.metrics import reciprocal_rank, recall_at_k
from src.index.bm25_store import BM25Store
from src.index.vector_store import FaissVectorStore
from src.rerank.cross_encoder import CrossEncoderReranker
from src.retrieve.hybrid import HybridRetriever


def load_qa_set(path: str) -> list[dict]:
    items = []
    with Path(path).open(encoding="utf-8") as f:
        for line in f:
            items.append(json.loads(line))
    return items


def run_eval(qa_set_path: str):
    settings = load_settings()
    qa_set = load_qa_set(qa_set_path)

    embedder = Embedder(settings.embedding_model_name)
    vector_store = FaissVectorStore(settings.faiss_index_dir)
    vector_store.load()
    bm25_store = BM25Store(settings.bm25_index_dir)
    bm25_store.load()
    retriever = HybridRetriever(vector_store, bm25_store)
    reranker = CrossEncoderReranker(settings.reranker_model_name)

    ks = [1, 3, 5]

    for use_rerank in (False, True):
        recalls = {k: [] for k in ks}
        rrs = []

        for item in qa_set:
            query = item["question"]
            relevant = [(r["doc_id"], r["page"]) for r in item["relevant"]]

            query_vector = embedder.encode([query])[0]
            candidates = retriever.search(
                query,
                query_vector,
                top_k=settings.rerank_candidate_k,
                candidate_k=settings.rerank_candidate_k,
                rrf_k=settings.rrf_k,
                vector_weight=settings.vector_weight,
                bm25_weight=settings.bm25_weight,
            )

            if use_rerank:
                results = reranker.rerank(query, candidates, top_k=max(ks))
            else:
                results = candidates[: max(ks)]

            retrieved = [(r["doc_id"], r["page"]) for r in results]

            for k in ks:
                recalls[k].append(recall_at_k(retrieved, relevant, k))
            rrs.append(reciprocal_rank(retrieved, relevant))

        name = "hybrid + reranker" if use_rerank else "hybrid (no rerank)"
        print(f"=== {name} ===")
        for k in ks:
            avg = sum(recalls[k]) / len(recalls[k])
            print(f"Recall@{k}: {avg:.3f}")
        print(f"MRR: {sum(rrs) / len(rrs):.3f}")
        print()


if __name__ == "__main__":
    run_eval("eval/qa_set.jsonl")
