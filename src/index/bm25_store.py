import pickle
from pathlib import Path

from kiwipiepy import Kiwi
from rank_bm25 import BM25Okapi

_kiwi = Kiwi()
_KEEP_POS_PREFIXES = ("N", "V", "SL", "SH", "SN")  # 명사류, 용언(동사/형용사), 외국어, 한자, 숫자


def _char_ngrams(text: str, n: int) -> list[str]:
    """어절(공백 기준) 내부에서만 문자 단위 n-gram 생성 — 어절 경계를 넘어가지 않음"""
    grams = []
    for word in text.split():
        if len(word) >= n:
            grams.extend(word[i:i + n] for i in range(len(word) - n + 1))
        elif word:
            grams.append(word)
    return grams


def tokenize(text: str) -> list[str]:
    morphemes = [token.form for token in _kiwi.tokenize(text) if token.tag.startswith(_KEEP_POS_PREFIXES)]
    return morphemes + _char_ngrams(text, 2) + _char_ngrams(text, 3)


class BM25Store:
    def __init__(self, index_dir: str):
        self.index_dir = Path(index_dir)
        self.bm25: BM25Okapi | None = None
        self.metadatas: list[dict] = []

    def build(self, texts: list[str], metadatas: list[dict]):
        tokenized = [tokenize(t) for t in texts]
        self.bm25 = BM25Okapi(tokenized)
        self.metadatas = metadatas

    def save(self):
        self.index_dir.mkdir(parents=True, exist_ok=True)
        with (self.index_dir / "bm25.pkl").open("wb") as f:
            pickle.dump({"bm25": self.bm25, "metadatas": self.metadatas}, f)

    def load(self):
        with (self.index_dir / "bm25.pkl").open("rb") as f:
            data = pickle.load(f)
        self.bm25 = data["bm25"]
        self.metadatas = data["metadatas"]

    def search(self, query: str, top_k: int) -> list[dict]:
        scores = self.bm25.get_scores(tokenize(query))
        ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
        return [{**self.metadatas[i], "score": float(scores[i])} for i in ranked]
