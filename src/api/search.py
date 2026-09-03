import asyncio

from fastapi import APIRouter, Request

from src.schemas.search import SearchRequest, SearchResponse

router = APIRouter()


@router.post("/search", response_model=SearchResponse)
async def search(req: SearchRequest, request: Request) -> SearchResponse:
    pipeline = request.app.state.pipeline
    # pipeline.search()는 임베딩/재랭킹 등 CPU 바운드 동기 연산이라, 그냥 호출하면
    # 요청 하나가 끝날 때까지 이벤트 루프 전체(다른 모든 요청 포함)가 막힘 — 별도
    # 스레드로 돌려서 서버가 계속 다른 요청을 받을 수 있게 함.
    results = await asyncio.to_thread(pipeline.search, req.query)
    return SearchResponse(context=results)
