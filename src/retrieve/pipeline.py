from src.core.config import Settings
from src.embedding.embedder import Embedder
from src.index.bm25_store import BM25Store
from src.index.vector_store import FaissVectorStore
from src.rerank.cross_encoder import CrossEncoderReranker
from src.retrieve.hybrid import HybridRetriever


class SearchPipeline:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.embedder = Embedder(settings.embedding_model_name)

        self.vector_store = FaissVectorStore(settings.faiss_index_dir)
        self.vector_store.load()

        self.bm25_store = BM25Store(settings.bm25_index_dir)
        self.bm25_store.load()

        self.retriever = HybridRetriever(self.vector_store, self.bm25_store)
        self.reranker = CrossEncoderReranker(settings.reranker_model_name)

    def search(self, query: str, top_k: int | None = None) -> list[dict]:
        top_k = top_k or self.settings.top_k
        query_vector = self.embedder.encode([query])[0]

        candidates = self.retriever.search(
            query,
            query_vector,
            top_k=self.settings.rerank_candidate_k,
            candidate_k=self.settings.rerank_candidate_k,
            rrf_k=self.settings.rrf_k,
            vector_weight=self.settings.vector_weight,
            bm25_weight=self.settings.bm25_weight,
        )
        return self.reranker.rerank(query, candidates, top_k=top_k)
