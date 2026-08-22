import re

from src.schemas.document import Chunk

_PARAGRAPH_SEP = re.compile(r"\n\s*\n+")
_SENTENCE_SEP = re.compile(r"(?<=[.!?])\s+")


def _split_recursive(text: str, chunk_size: int) -> list[str]:
    if len(text) <= chunk_size:
        return [text]

    for separator in (_PARAGRAPH_SEP, _SENTENCE_SEP):
        units = [u for u in separator.split(text) if u.strip()]
        if len(units) > 1:
            pieces = []
            for unit in units:
                pieces.extend(_split_recursive(unit, chunk_size))
            return pieces

    return [text[i : i + chunk_size] for i in range(0, len(text), chunk_size)]


def _merge_with_overlap(units: list[str], chunk_size: int, overlap: int) -> list[str]:
    chunks = []
    current = ""

    for unit in units:
        candidate = f"{current}\n{unit}" if current else unit
        if len(candidate) <= chunk_size:
            current = candidate
            continue

        if current:
            chunks.append(current)
        current = f"{current[-overlap:]}\n{unit}" if overlap and current else unit

    if current:
        chunks.append(current)

    return chunks


def chunk_pages(pages: list[Chunk], chunk_size: int = 500, overlap: int = 50) -> list[Chunk]:
    result = []
    for page in pages:
        units = _split_recursive(page.text, chunk_size)
        for text in _merge_with_overlap(units, chunk_size, overlap):
            result.append(
                Chunk(
                    doc_id=page.doc_id,
                    source=page.source,
                    category=page.category,
                    page=page.page,
                    text=text,
                )
            )
    return result
