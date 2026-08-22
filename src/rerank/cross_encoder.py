from sentence_transformers import CrossEncoder


class CrossEncoderReranker:
    def __init__(self, model_name: str):
        self.model = CrossEncoder(model_name)

    def rerank(self, query: str, candidates: list[dict], top_k: int) -> list[dict]:
        pairs = [(query, c["text"]) for c in candidates]
        scores = self.model.predict(pairs)
        ranked = sorted(zip(candidates, scores), key=lambda pair: pair[1], reverse=True)
        return [{**c, "rerank_score": float(s)} for c, s in ranked[:top_k]]
