from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI

from src.api.router import api_router
from src.core.config import load_settings
from src.retrieve.pipeline import SearchPipeline


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = load_settings()
    app.state.pipeline = SearchPipeline(settings)
    yield


app = FastAPI(
    title="ai-rag-engine",
    version="0.1.0",
    lifespan=lifespan,
)
app.include_router(api_router)


if __name__ == "__main__":
    uvicorn.run("src.main:app", host="0.0.0.0", port=8001)
