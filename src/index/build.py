import json
from pathlib import Path

from src.core.config import load_settings
from src.embedding.embedder import Embedder
from src.index.bm25_store import BM25Store
from src.index.vector_store import FaissVectorStore
from src.schemas.document import Chunk


def build_indices(chunks_path: str):
    settings = load_settings()

    chunks = []
    with Path(chunks_path).open(encoding="utf-8") as f:
        for line in f:
            chunks.append(Chunk(**json.loads(line)))

    metadatas = [c.model_dump() for c in chunks]

    embedder = Embedder(settings.embedding_model_name)
    vectors = embedder.encode([c.text for c in chunks])

    vector_store = FaissVectorStore(settings.faiss_index_dir)
    vector_store.build(vectors, metadatas)
    vector_store.save()

    bm25_store = BM25Store(settings.bm25_index_dir)
    bm25_store.build([c.text for c in chunks], metadatas)
    bm25_store.save()

    print(f"[done] {len(chunks)} chunks -> {settings.faiss_index_dir}, {settings.bm25_index_dir}")
