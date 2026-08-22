import pickle
from pathlib import Path

import faiss
import numpy as np


class FaissVectorStore:
    def __init__(self, index_dir: str):
        self.index_dir = Path(index_dir)
        self.index: faiss.Index | None = None
        self.metadatas: list[dict] = []

    def build(self, vectors: np.ndarray, metadatas: list[dict]):
        dim = vectors.shape[1]
        self.index = faiss.IndexFlatIP(dim)
        self.index.add(vectors)
        self.metadatas = metadatas

    def save(self):
        self.index_dir.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(self.index_dir / "index.faiss"))
        with (self.index_dir / "metadatas.pkl").open("wb") as f:
            pickle.dump(self.metadatas, f)

    def load(self):
        self.index = faiss.read_index(str(self.index_dir / "index.faiss"))
        with (self.index_dir / "metadatas.pkl").open("rb") as f:
            self.metadatas = pickle.load(f)

    def search(self, query_vector: np.ndarray, top_k: int) -> list[dict]:
        scores, indices = self.index.search(query_vector.reshape(1, -1), top_k)
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            results.append({**self.metadatas[idx], "score": float(score)})
        return results
