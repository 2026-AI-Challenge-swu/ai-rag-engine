from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    embedding_model_name: str = "dragonkue/snowflake-arctic-embed-l-v2.0-ko"
    faiss_index_dir: str = "storage/faiss"
    bm25_index_dir: str = "storage/bm25"
    top_k: int = 5
    rrf_k: int = 60
    vector_weight: float = 0.5
    bm25_weight: float = 0.5

    reranker_model_name: str = "Dongjin-kr/ko-reranker"
    rerank_candidate_k: int = 20

    openai_api_key: str = ""
    vision_model_name: str = "gpt-4o-mini"


def load_settings() -> Settings:
    return Settings()
