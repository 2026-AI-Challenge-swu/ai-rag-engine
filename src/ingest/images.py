import base64
import time
from pathlib import Path

import pymupdf
from openai import OpenAI, RateLimitError

SYSTEM_PROMPT = (
    "당신은 문서 내 다양한 형태의 이미지를 분석하여, 검색 기반 질문응답 시스템(RAG)에 활용 가능한 "
    "텍스트 설명을 생성하는 AI입니다. 이미지는 인포그래픽, 표, 그래프, 다이어그램 등 다양한 유형일 "
    "수 있으며, 다음 기준에 따라 요약을 작성하세요.\n"
    "- 이미지의 주제와 목적을 명확하게 파악하고 자연어로 요약합니다.\n"
    "- 이미지가 전달하는 구조나 흐름이 있다면 순차적으로 설명합니다. (예: 단계, 관계, 비교 등)\n"
    "- 표, 그래프, 수치 정보는 전체 흐름과 특징적인 차이만 요약하고, 수치 나열은 피합니다.\n"
    "- 시각적 요소(색상, 도형, 배치 등)는 정보 전달에 필요할 경우에만 간단히 설명합니다.\n"
    "- 설명은 검색 가능한 핵심 키워드를 포함하고, 감상이나 해석 없이 사실 중심 문장으로 구성해야 합니다.\n"
    "- 최종 출력은 3~5문장 이내의 단일 문단으로 구성합니다."
)


def extract_visual_pages(path: Path, min_image_size: int = 150, min_drawings: int = 20) -> list[tuple[int, bytes, str]]:
    results = []
    doc = pymupdf.open(path)
    for page_num, page in enumerate(doc, start=1):
        has_large_image = False
        for img in page.get_images(full=True):
            xref = img[0]
            base_image = doc.extract_image(xref)
            if base_image["width"] >= min_image_size and base_image["height"] >= min_image_size:
                has_large_image = True
                break

        has_many_drawings = len(page.get_drawings()) >= min_drawings

        if not (has_large_image or has_many_drawings):
            continue

        pix = page.get_pixmap(dpi=200)
        results.append((page_num, pix.tobytes("png"), "png"))
    return results


def describe_image(image_bytes: bytes, ext: str, client: OpenAI, model: str, max_retries: int = 5) -> str:
    b64 = base64.b64encode(image_bytes).decode("utf-8")
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "이 이미지는 문서 내 시각 자료입니다. 핵심 정보를 요약해 주세요."},
                {"type": "image_url", "image_url": {"url": f"data:image/{ext};base64,{b64}"}},
            ],
        },
    ]

    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(model=model, messages=messages, temperature=0.25)
            return response.choices[0].message.content.strip()
        except RateLimitError:
            if attempt == max_retries - 1:
                raise
            wait = 2**attempt
            print(f"    [rate limit] {wait}초 대기 후 재시도...")
            time.sleep(wait)
