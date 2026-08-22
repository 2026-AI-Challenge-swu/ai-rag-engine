from fastapi import APIRouter, Request

from src.schemas.search import SearchRequest, SearchResponse

router = APIRouter()


@router.post("/search", response_model=SearchResponse)
async def search(req: SearchRequest, request: Request) -> SearchResponse:
    pipeline = request.app.state.pipeline
    results = pipeline.search(req.query)
    return SearchResponse(context=results)
