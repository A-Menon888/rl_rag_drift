import numpy as np

from src.data.facts import extract_values


def _states_gold(query, chunk):
    return query.gold_answer in extract_values(query.answer_pattern, chunk.text)


def retrieval_diagnostics(retriever, queries, top_k=3, version=0):
    """Return auditable per-query ranking records for one KB snapshot.

    A retrieved chunk is relevant when it states the query's gold value for its
    answer slot. Unanswerable queries (gold None) have no relevant chunk.
    """
    rows = []
    for query in queries:
        results = retriever.search(query, top_k)
        answerable = query.gold_answer is not None
        rank = next((position + 1 for position, result in enumerate(results)
                     if answerable and _states_gold(query, result.fact)), 0)
        rows.append({
            "query_id": query.query_id,
            "query": query.text,
            "expected_chunk_id": next((fact.document_id for fact in retriever.facts
                                       if answerable and fact.source == query.source and _states_gold(query, fact)), None),
            "rank": rank,
            "top_1_chunk_id": results[0].fact.document_id if results else None,
            "top_1_source": results[0].fact.source if results else None,
            "top_1_score": results[0].score if results else None,
            "top_2_score": results[1].score if len(results) > 1 else None,
            "score_margin": results[0].score - results[1].score if len(results) > 1 else None,
            "retrieved_chunk_ids": "|".join(result.fact.document_id for result in results),
        })
    return rows


def summarize_retrieval_diagnostics(rows, top_k=3):
    if not rows:
        return {"recall_at_1": 0.0, f"recall_at_{top_k}": 0.0, "mrr": 0.0}
    ranks = np.array([row["rank"] for row in rows])
    return {
        "recall_at_1": float(np.mean(ranks == 1)),
        f"recall_at_{top_k}": float(np.mean((ranks >= 1) & (ranks <= top_k))),
        "mrr": float(np.mean([1.0 / rank if rank else 0.0 for rank in ranks])),
    }


def summarize(infos):
    """Per-episode metrics from the terminal info record of each question."""
    if not infos:
        return {}
    def mean(values):
        values = list(values)
        return float(np.mean(values)) if values else None
    drifted = [item for item in infos if item["affected_by_drift"]]
    stable = [item for item in infos if not item["affected_by_drift"]]
    # Correlated facts share a drift event; weight each event once so that one
    # change affecting many facts is not counted as many independent changes.
    by_event = {}
    for item in drifted:
        by_event.setdefault(item.get("drift_event") or item["query_id"], []).append(item["correct"])
    return {
        "accuracy": mean(item["correct"] for item in infos),
        "average_reward": mean(item["episode_return"] for item in infos),
        "average_searches": mean(item["searches"] for item in infos),
        "retrieval_rate": mean(item["searches"] > 0 for item in infos),
        "retrieval_cost": mean(item["retrieval_cost"] for item in infos),
        # Searched although closed-book answering was already correct.
        "unnecessary_retrieval_rate": mean(
            item["searches"] > 0 and item["memory_status"] == "correct" and item["answerable"] for item in infos),
        "give_up_rate": mean(item["final_action"] == "give_up" for item in infos),
        "wrong_answer_rate": mean(item["final_action"] == "answer" and not item["correct"] for item in infos),
        "unanswerable_accuracy": mean(item["correct"] for item in infos if not item["answerable"]),
        "drifted_accuracy": mean(item["correct"] for item in drifted),
        "drifted_event_accuracy": mean(mean(values) for values in by_event.values()),
        "drifted_events": len(by_event),
        "stable_accuracy": mean(item["correct"] for item in stable),
    }
