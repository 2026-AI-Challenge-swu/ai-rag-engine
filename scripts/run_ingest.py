import argparse
from pathlib import Path

from openai import OpenAI

from src.core.config import load_settings
from src.ingest.extract import extract_pdf
from src.ingest.images import describe_image, extract_visual_pages
from src.schemas.document import Chunk

FILE_METADATA = {
    "KFTC_퇴직연금제도일반.pdf": {
        "doc_id": "kftc_general",
        "source": "금융결제원(KFTC)",
        "category": "일반상식",
    },
    "[근로복지공단]퇴직연금 가입자 교육교재(2025년).pdf": {
        "doc_id": "kcomwel_2025",
        "source": "근로복지공단",
        "category": "일반상식",
    },
    "[신한은행]퇴직연금 교육자료.pdf": {
        "doc_id": "shinhan_edu",
        "source": "신한은행",
        "category": "일반상식",
    },
    "공인노무사 곽동환이 알려주는 퇴직연금 교육.pdf": {
        "doc_id": "labor_attorney_edu",
        "source": "공인노무사 곽동환",
        "category": "일반상식",
    },
    "미래에셋은퇴연구소_셀프연금의 의미와 효과적 활용방안.pdf": {
        "doc_id": "miraeasset_self_pension",
        "source": "미래에셋은퇴연구소",
        "category": "일반상식",
    },
    "실태로 본 개인형 사적연금.pdf": {
        "doc_id": "irp_status_report",
        "source": "정책보고서",
        "category": "일반상식",
    },
    "연금전환 정책보고서.pdf": {
        "doc_id": "pension_conversion_policy",
        "source": "정책보고서",
        "category": "일반상식",
    },
    "현대증권_바람직한 노후설계의 방향.pdf": {
        "doc_id": "hyundai_securities_retirement",
        "source": "현대증권",
        "category": "일반상식",
    },
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_dir", type=str, default="data/raw")
    parser.add_argument("--out_dir", type=str, default="data/processed")
    args = parser.parse_args()

    raw_dir = Path(args.raw_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    settings = load_settings()
    client = OpenAI(api_key=settings.openai_api_key)

    for filename, meta in FILE_METADATA.items():
        pdf_path = raw_dir / filename
        if not pdf_path.exists():
            print(f"[skip] not found: {filename}")
            continue

        text_pages = extract_pdf(
            pdf_path,
            doc_id=meta["doc_id"],
            source=meta["source"],
            category=meta["category"],
        )

        image_records = []
        for page_num, image_bytes, ext in extract_visual_pages(pdf_path):
            description = describe_image(image_bytes, ext, client, settings.vision_model_name)
            image_records.append(
                Chunk(
                    doc_id=meta["doc_id"],
                    source=meta["source"],
                    category=meta["category"],
                    page=page_num,
                    text=description,
                )
            )
            print(f"  [image] {filename} p.{page_num} described")

        page_records = text_pages + image_records

        out_path = out_dir / f"{meta['doc_id']}.jsonl"
        with out_path.open("w", encoding="utf-8") as f:
            for page_record in page_records:
                f.write(page_record.model_dump_json() + "\n")

        print(f"[done] {filename} -> {out_path} ({len(text_pages)} text + {len(image_records)} image records)")


if __name__ == "__main__":
    main()
