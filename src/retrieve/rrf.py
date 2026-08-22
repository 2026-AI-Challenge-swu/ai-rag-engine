def reciprocal_rank_fusion(
    rankings: list[list[dict]],
    weights: list[float] | None = None,
    k: int = 60,
) -> list[dict]:
    if weights is None:
        weights = [1.0] * len(rankings)

    scores: dict[str, float] = {}
    items: dict[str, dict] = {}
    for ranking, weight in zip(rankings, weights):
        for rank, item in enumerate(ranking, start=1):
            key = item["text"]
            scores[key] = scores.get(key, 0.0) + weight / (k + rank)
            items[key] = item

    ranked_keys = sorted(scores, key=lambda key: scores[key], reverse=True)
    return [{**items[key], "rrf_score": scores[key]} for key in ranked_keys]
