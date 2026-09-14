import numpy as np


def retrieval_diagnostics(retriever, queries, top_k=3, version=0):
    """Return auditable per-query ranking records for one KB snapshot."""
    rows = []
    for query in queries:
        results = retriever.search(query, top_k)
        expected = query.current_answer
        rank = next((position + 1 for position, result in enumerate(results) if result.fact.text == expected), 0)
        rows.append({
            "query_id": query.query_id,
            "query": query.text,
            "expected_chunk_id": next((fact.document_id for fact in retriever.facts if fact.text == expected), None),
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


def summarize(infos, window=30, threshold=0.9):
    if not infos:
        return {"accuracy": 0.0, "average_reward": 0.0, "retrieval_rate": 0.0, "retrieval_cost": 0.0,
                "drifted_accuracy": None, "stable_accuracy": None}
    rewards = np.array([item["reward"] for item in infos])
    correct = np.array([item["correct"] for item in infos])

    # Drift-aware breakdown — only when info records carry the field
    drifted = [item for item in infos if item.get("affected_by_drift") is True]
    stable  = [item for item in infos if item.get("affected_by_drift") is False]
    drifted_accuracy = float(np.mean([i["correct"] for i in drifted])) if drifted else None
    stable_accuracy  = float(np.mean([i["correct"] for i in stable]))  if stable  else None

    return {
        "accuracy": float(correct.mean()),
        "average_reward": float(rewards.mean()),
        "retrieval_rate": float(np.mean([item["action"] for item in infos])),
        "retrieval_cost": float(np.mean([item["retrieval_cost"] for item in infos])),
        "drifted_accuracy": drifted_accuracy,
        "stable_accuracy": stable_accuracy,
    }
