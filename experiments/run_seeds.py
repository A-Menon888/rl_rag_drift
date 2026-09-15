import argparse
import csv
import json
import math
import statistics
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments.run_all import run_experiment
from src.data.documents import generate_document_queries, load_knowledge_base

METRICS = (
    "accuracy",
    "average_reward",
    "retrieval_rate",
    "retrieval_cost",
    "drifted_accuracy",
    "stable_accuracy",
)


def _numeric_values(rows, metric):
    values = []
    for row in rows:
        value = row.get(metric, "")
        if value not in ("", None):
            values.append(float(value))
    return values


def aggregate_rows(seed_rows):
    grouped = {}
    for rows in seed_rows:
        for row in rows:
            key = (row["policy"], row["knowledge_base"])
            grouped.setdefault(key, []).append(row)

    aggregate = []
    for (policy, knowledge_base), rows in sorted(grouped.items()):
        output = {"policy": policy, "knowledge_base": knowledge_base}
        for metric in METRICS:
            values = _numeric_values(rows, metric)
            output[f"{metric}_mean"] = statistics.fmean(values) if values else ""
            output[f"{metric}_std"] = statistics.stdev(values) if len(values) > 1 else 0.0 if values else ""
            output[f"{metric}_min"] = min(values) if values else ""
            output[f"{metric}_max"] = max(values) if values else ""
            output[f"{metric}_values"] = json.dumps(values)
        aggregate.append(output)
    return aggregate


def write_csv(path, rows):
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def alignment_issues(config):
    corpus = ROOT / config["corpus_dir"]
    kb_a = load_knowledge_base(corpus, "kb_a")
    kb_b = load_knowledge_base(corpus, "kb_b")
    issues = []
    for seed in range(config["base_seed"], config["base_seed"] + config["n_seeds"]):
        queries_a, queries_b = generate_document_queries(kb_a, kb_b, seed=seed)
        if len(queries_a) != len(queries_b):
            issues.append((seed, "query count mismatch"))
            continue
        for query_a, query_b in zip(queries_a, queries_b):
            if query_a.query_id != query_b.query_id:
                issues.append((seed, f"query ID mismatch: {query_a.query_id}/{query_b.query_id}"))
            if query_a.memorized_answer != query_b.memorized_answer:
                issues.append((seed, f"memorized answer mismatch: {query_a.query_id}"))
            if not query_a.current_answer or not query_b.current_answer:
                issues.append((seed, f"empty answer: {query_a.query_id}"))
    return issues


def main():
    parser = argparse.ArgumentParser(description="Run the RL-RAG benchmark across deterministic seeds.")
    parser.add_argument("--n-seeds", type=int, default=10)
    parser.add_argument("--base-seed", type=int, default=0)
    args = parser.parse_args()
    if args.n_seeds < 1:
        parser.error("--n-seeds must be at least 1")

    with (ROOT / "configs/default.yaml").open(encoding="utf-8") as handle:
        base_config = yaml.safe_load(handle)
    base_config["n_seeds"] = args.n_seeds
    base_config["base_seed"] = args.base_seed

    metrics_root = ROOT / "results" / "metrics"
    seed_rows = []
    for seed in range(args.base_seed, args.base_seed + args.n_seeds):
        config = dict(base_config)
        config["seed"] = seed
        output_root = metrics_root / f"seed_{seed}"
        print(f"\n=== Running seed {seed} ===")
        rows = run_experiment(config, output_root=output_root, summary_path=output_root / "summary.csv")
        seed_rows.append(rows)

    aggregate = aggregate_rows(seed_rows)
    aggregate_path = metrics_root / "aggregate_summary.csv"
    write_csv(aggregate_path, aggregate)
    issues = alignment_issues(base_config)

    print("\n--- Multi-seed summary (accuracy and retrieval rate) ---")
    for row in aggregate:
        print(f"{row['policy']:<24} {row['knowledge_base']:<5} "
              f"accuracy={float(row['accuracy_mean']):.3f} +/- {float(row['accuracy_std']):.3f} "
              f"retrieval_rate={float(row['retrieval_rate_mean']):.3f} +/- "
              f"{float(row['retrieval_rate_std']):.3f}")
    print(f"\nWrote aggregate results to {aggregate_path}")
    print(f"Alignment issues: {len(issues)}")
    for seed, issue in issues:
        print(f"  seed {seed}: {issue}")


if __name__ == "__main__":
    main()
