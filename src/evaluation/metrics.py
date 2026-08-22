def recall_at_k(retrieved: list[tuple], relevant: list[tuple], k: int) -> float:
    relevant_set = set(relevant)
    if not relevant_set:
        return 0.0
    top_k = set(retrieved[:k])
    return len(top_k & relevant_set) / len(relevant_set)


def reciprocal_rank(retrieved: list[tuple], relevant: list[tuple]) -> float:
    relevant_set = set(relevant)
    for rank, item in enumerate(retrieved, start=1):
        if item in relevant_set:
            return 1.0 / rank
    return 0.0
