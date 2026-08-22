import numpy as np

from src.index.bm25_store import BM25Store
from src.index.vector_store import FaissVectorStore
from src.retrieve.rrf import reciprocal_rank_fusion


class HybridRetriever:
    def __init__(self, vector_store: FaissVectorStore, bm25_store: BM25Store):
        self.vector_store = vector_store
        self.bm25_store = bm25_store

    def search(
        self,
        query: str,
        query_vector: np.ndarray,
        top_k: int,
        candidate_k: int = 20,
        rrf_k: int = 60,
        vector_weight: float = 1.0,
        bm25_weight: float = 1.0,
    ) -> list[dict]:
        vector_results = self.vector_store.search(query_vector, candidate_k)
        bm25_results = self.bm25_store.search(query, candidate_k)

        fused = reciprocal_rank_fusion(
            [vector_results, bm25_results],
            weights=[vector_weight, bm25_weight],
            k=rrf_k,
        )
        return fused[:top_k]
