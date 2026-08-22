from pathlib import Path

import pdfplumber

from src.schemas.document import Chunk


def _cluster_boxes(boxes: list[tuple[float, float, float, float]], gap: float = 5) -> list[list[tuple]]:
    n = len(boxes)
    parent = list(range(n))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i: int, j: int) -> None:
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[ri] = rj

    def near(a: tuple, b: tuple) -> bool:
        return a[0] - gap < b[2] and a[2] + gap > b[0] and a[1] - gap < b[3] and a[3] + gap > b[1]

    for i in range(n):
        for j in range(i + 1, n):
            if near(boxes[i], boxes[j]):
                union(i, j)

    clusters: dict[int, list[tuple]] = {}
    for i in range(n):
        clusters.setdefault(find(i), []).append(boxes[i])
    return list(clusters.values())


def extract_pdf(
    path: Path,
    doc_id: str,
    source: str,
    category: str,
    x_tolerance: float = 5,
    min_image_size: float = 150,
    min_cluster_elements: int = 15,
    margin: float = 10,
) -> list[Chunk]:
    chunks = []
    with pdfplumber.open(path) as pdf:
        for page_num, page in enumerate(pdf.pages, start=1):
            exclude_boxes = []

            for img in page.images:
                if img["width"] >= min_image_size and img["height"] >= min_image_size:
                    exclude_boxes.append((img["x0"], img["top"], img["x1"], img["bottom"]))

            elements = page.rects + page.lines + page.curves
            boxes = [(e["x0"], e["top"], e["x1"], e["bottom"]) for e in elements]
            for cluster in _cluster_boxes(boxes):
                if len(cluster) < min_cluster_elements:
                    continue
                exclude_boxes.append(
                    (
                        min(b[0] for b in cluster),
                        min(b[1] for b in cluster),
                        max(b[2] for b in cluster),
                        max(b[3] for b in cluster),
                    )
                )

            px0, ptop, px1, pbottom = page.bbox
            clean_page = page
            for x0, top, x1, bottom in exclude_boxes:
                box = (
                    max(x0 - margin, px0),
                    max(top - margin, ptop),
                    min(x1 + margin, px1),
                    min(bottom + margin, pbottom),
                )
                if box[0] >= box[2] or box[1] >= box[3]:
                    continue
                clean_page = clean_page.outside_bbox(box)

            text = (clean_page.extract_text(x_tolerance=x_tolerance) or "").strip()
            if not text:
                continue
            chunks.append(
                Chunk(
                    doc_id=doc_id,
                    source=source,
                    category=category,
                    page=page_num,
                    text=text,
                )
            )
    return chunks
