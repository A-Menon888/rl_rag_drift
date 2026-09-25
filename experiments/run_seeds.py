import argparse
import csv
import json
import math
import statistics
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
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


def run_seed(base_config, seed, metrics_root):
    """Run one isolated seed; suitable for sequential or process-parallel use."""
    config = dict(base_config)
    config["seed"] = seed
    output_root = Path(metrics_root) / f"seed_{seed}"
    print(f"\n=== Running seed {seed} ===", flush=True)
    rows = run_experiment(config, output_root=output_root, summary_path=output_root / "summary.csv")
    return seed, rows


def load_completed_seed(seed, metrics_root):
    """Load a complete seed only when both aggregate and query artifacts exist."""
    output_root = Path(metrics_root) / f"seed_{seed}"
    summary_path = output_root / "summary.csv"
    query_results_path = output_root / "kb_b_query_results.csv"
    if not summary_path.exists() or not query_results_path.exists():
        return None
    with summary_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    return rows if rows else None


def main():
    parser = argparse.ArgumentParser(description="Run the RL-RAG benchmark across deterministic seeds.")
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs" / "default.yaml",
        help="Experiment YAML configuration (default: configs/default.yaml).",
    )
    parser.add_argument("--n-seeds", type=int, default=10)
    parser.add_argument("--base-seed", type=int, default=0)
    parser.add_argument(
        "--jobs",
        type=int,
        default=1,
        help="Independent seed processes to run concurrently (default: 1).",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Reuse seeds that already have summary and paired query-result files.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=ROOT / "results" / "metrics",
        help="Directory for seed subdirectories and aggregate_summary.csv.",
    )
    parser.add_argument(
        "--retrieval-cost",
        type=float,
        default=None,
        help="Optional experiment override, useful for explicit control runs.",
    )
    args = parser.parse_args()
    if args.n_seeds < 1:
        parser.error("--n-seeds must be at least 1")
    if args.jobs < 1:
        parser.error("--jobs must be at least 1")

    config_path = args.config if args.config.is_absolute() else ROOT / args.config
    with config_path.open(encoding="utf-8") as handle:
        base_config = yaml.safe_load(handle)
    base_config["n_seeds"] = args.n_seeds
    base_config["base_seed"] = args.base_seed
    if args.retrieval_cost is not None:
        base_config["retrieval_cost"] = args.retrieval_cost

    metrics_root = args.output_root
    seeds = list(range(args.base_seed, args.base_seed + args.n_seeds))
    rows_by_seed = {}
    pending_seeds = []
    for seed in seeds:
        completed = load_completed_seed(seed, metrics_root) if args.resume else None
        if completed is None:
            pending_seeds.append(seed)
        else:
            rows_by_seed[seed] = completed
            print(f"=== Reusing completed seed {seed} ===", flush=True)
    if args.jobs == 1:
        for seed in pending_seeds:
            completed_seed, rows = run_seed(base_config, seed, metrics_root)
            rows_by_seed[completed_seed] = rows
    else:
        with ProcessPoolExecutor(max_workers=args.jobs) as executor:
            futures = {
                executor.submit(run_seed, base_config, seed, metrics_root): seed
                for seed in pending_seeds
            }
            for future in as_completed(futures):
                completed_seed, rows = future.result()
                rows_by_seed[completed_seed] = rows
                print(f"=== Completed seed {completed_seed} ===", flush=True)
    seed_rows = [rows_by_seed[seed] for seed in seeds]

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
