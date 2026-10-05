import numpy as np


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
