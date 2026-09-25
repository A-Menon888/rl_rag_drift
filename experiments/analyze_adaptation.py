"""Paired statistical analysis of adaptation versus full retraining.

The seed is the unit of inference. Query-level bootstrap intervals are diagnostic
only and never affect the across-seed decision.
"""

import argparse
import csv
import json
import math
import statistics
from pathlib import Path

import numpy as np
from scipy.optimize import brentq
from scipy.stats import nct, t


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_Z_VALUE = 1.96
DEFAULT_ALPHA = 0.05
DEFAULT_POWER = 0.80


def read_csv(path):
    with Path(path).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _policy_accuracy(rows, policy, knowledge_base="kb_b"):
    matches = [
        row for row in rows
        if row["policy"] == policy and row["knowledge_base"] == knowledge_base
    ]
    if len(matches) != 1:
        raise ValueError(
            f"Expected one {policy}/{knowledge_base} row, found {len(matches)}"
        )
    return float(matches[0]["accuracy"])


def load_seed_pairs(metrics_root, n_seeds=None, base_seed=0):
    metrics_root = Path(metrics_root)
    aggregate_path = metrics_root / "aggregate_summary.csv"
    if not aggregate_path.exists():
        raise FileNotFoundError(f"Missing aggregate results: {aggregate_path}")
    # Reading this file is intentional: it is the public multi-seed artifact.
    # Per-seed summaries remain authoritative for preserving pair identity.
    read_csv(aggregate_path)

    if n_seeds is None:
        seed_dirs = sorted(
            (path for path in metrics_root.glob("seed_*") if path.is_dir()),
            key=lambda path: int(path.name.split("_", 1)[1]),
        )
    else:
        seed_dirs = [metrics_root / f"seed_{seed}" for seed in range(base_seed, base_seed + n_seeds)]

    pairs = []
    for seed_dir in seed_dirs:
        seed = int(seed_dir.name.split("_", 1)[1])
        summary_path = seed_dir / "summary.csv"
        if not summary_path.exists():
            raise FileNotFoundError(f"Missing per-seed summary: {summary_path}")
        rows = read_csv(summary_path)
        pairs.append({
            "seed": seed,
            "old_policy_accuracy": _policy_accuracy(rows, "old_policy"),
            "adapted_policy_accuracy": _policy_accuracy(rows, "adapted_policy"),
            "full_retrain_policy_accuracy": _policy_accuracy(rows, "full_retrain_policy"),
        })
    if len(pairs) < 2:
        raise ValueError("Paired inference requires at least two seeds")
    return pairs


def minimum_detectable_effect(n_seeds, observed_std, alpha=DEFAULT_ALPHA, power=DEFAULT_POWER):
    """Two-sided paired-t MDE, returned in standardized and accuracy units."""
    if n_seeds < 2:
        raise ValueError("Power calculation requires at least two seeds")
    if not 0.0 < alpha < 1.0 or not 0.0 < power < 1.0:
        raise ValueError("alpha and power must lie strictly between zero and one")
    degrees_freedom = n_seeds - 1
    critical = t.ppf(1.0 - alpha / 2.0, degrees_freedom)

    def achieved_power(effect_size):
        noncentrality = effect_size * math.sqrt(n_seeds)
        return nct.cdf(-critical, degrees_freedom, noncentrality) + nct.sf(
            critical, degrees_freedom, noncentrality
        )

    upper = 1.0
    while achieved_power(upper) < power:
        upper *= 2.0
    standardized = brentq(lambda effect: achieved_power(effect) - power, 0.0, upper)
    return {
        "standardized": standardized,
        "accuracy": standardized * observed_std,
        "alpha": alpha,
        "power": power,
    }


def effect_size_bin(value):
    if value is None or math.isnan(value):
        return "undefined"
    magnitude = abs(value)
    if magnitude < 0.2:
        return "negligible"
    if magnitude < 0.5:
        return "small"
    if magnitude < 0.8:
        return "medium"
    return "large"


def analyze_pairs(pairs, z_value=DEFAULT_Z_VALUE, alpha=DEFAULT_ALPHA, power=DEFAULT_POWER):
    if z_value <= 0.0:
        raise ValueError("z_value must be positive")
    differences = [
        pair["full_retrain_policy_accuracy"] - pair["adapted_policy_accuracy"]
        for pair in pairs
    ]
    n_seeds = len(differences)
    mean_difference = statistics.fmean(differences)
    std_difference = statistics.stdev(differences)
    standard_error = std_difference / math.sqrt(n_seeds)
    tau = z_value * standard_error

    if std_difference == 0.0:
        if mean_difference == 0.0:
            cohens_d = None
            decision = "adaptation statistically equivalent to full retrain"
        else:
            cohens_d = math.copysign(math.inf, mean_difference)
            decision = (
                "full retrain meaningfully better"
                if mean_difference > 0.0
                else "adaptation meaningfully better"
            )
    else:
        cohens_d = mean_difference / std_difference
        if mean_difference >= tau:
            decision = "full retrain meaningfully better"
        elif mean_difference <= -tau:
            decision = "adaptation meaningfully better"
        else:
            decision = "adaptation statistically equivalent to full retrain"

    old_mean = statistics.fmean(pair["old_policy_accuracy"] for pair in pairs)
    adapted_mean = statistics.fmean(pair["adapted_policy_accuracy"] for pair in pairs)
    retrain_mean = statistics.fmean(pair["full_retrain_policy_accuracy"] for pair in pairs)
    recovery_denominator = retrain_mean - old_mean
    recovery_ratio = (
        None
        if math.isclose(recovery_denominator, 0.0, abs_tol=1e-15)
        else (adapted_mean - old_mean) / recovery_denominator
    )

    return {
        "n_seeds": n_seeds,
        "per_seed_accuracy_pairs": [
            {**pair, "difference_full_minus_adapted": difference}
            for pair, difference in zip(pairs, differences)
        ],
        "mean_old_policy_kb_b_accuracy": old_mean,
        "mean_adapted_policy_kb_b_accuracy": adapted_mean,
        "mean_full_retrain_policy_kb_b_accuracy": retrain_mean,
        "recovery_ratio": recovery_ratio,
        "recovery_ratio_note": (
            "undefined: old policy already matches full retrain"
            if recovery_ratio is None else None
        ),
        "mean_paired_difference": mean_difference,
        "std_paired_difference": std_difference,
        "standard_error": standard_error,
        "z_value": z_value,
        "tau": tau,
        "cohens_d_paired": cohens_d,
        "absolute_effect_size_bin": effect_size_bin(cohens_d),
        "decision": decision,
        "unusual_adaptation_better_edge_case": decision == "adaptation meaningfully better",
        "minimum_detectable_effect": minimum_detectable_effect(
            n_seeds, std_difference, alpha=alpha, power=power
        ),
    }


def percentile_interval(values, confidence=0.95):
    tail = (1.0 - confidence) / 2.0
    low, high = np.quantile(values, [tail, 1.0 - tail])
    return float(low), float(high)


def bootstrap_query_accuracy(
    metrics_root, pairs, resamples=200, confidence=0.95, bootstrap_seed=1729
):
    if resamples < 1:
        raise ValueError("resamples must be positive")
    rng = np.random.default_rng(bootstrap_seed)
    results = []
    for pair in pairs:
        seed = pair["seed"]
        path = Path(metrics_root) / f"seed_{seed}" / "kb_b_query_results.csv"
        if not path.exists():
            raise FileNotFoundError(
                f"Missing query-level records for seed {seed}: {path}. Re-run run_seeds.py."
            )
        rows = read_csv(path)
        adapted = np.asarray([int(row["adapted_policy_correct"]) for row in rows], dtype=float)
        retrain = np.asarray([int(row["full_retrain_policy_correct"]) for row in rows], dtype=float)
        if not len(adapted) or len(adapted) != len(retrain):
            raise ValueError(f"Invalid paired query records for seed {seed}")
        indices = rng.integers(0, len(adapted), size=(resamples, len(adapted)))
        adapted_samples = adapted[indices].mean(axis=1)
        retrain_samples = retrain[indices].mean(axis=1)
        adapted_interval = percentile_interval(adapted_samples, confidence)
        retrain_interval = percentile_interval(retrain_samples, confidence)
        results.append({
            "seed": seed,
            "n_queries": len(adapted),
            "adapted_interval": list(adapted_interval),
            "adapted_interval_width": adapted_interval[1] - adapted_interval[0],
            "full_retrain_interval": list(retrain_interval),
            "full_retrain_interval_width": retrain_interval[1] - retrain_interval[0],
        })
    return {
        "resamples": resamples,
        "confidence": confidence,
        "per_seed": results,
        "mean_adapted_interval_width": statistics.fmean(
            item["adapted_interval_width"] for item in results
        ),
        "mean_full_retrain_interval_width": statistics.fmean(
            item["full_retrain_interval_width"] for item in results
        ),
    }


def analyze_directory(
    metrics_root,
    n_seeds=None,
    base_seed=0,
    z_value=DEFAULT_Z_VALUE,
    alpha=DEFAULT_ALPHA,
    power=DEFAULT_POWER,
    bootstrap_resamples=200,
):
    pairs = load_seed_pairs(metrics_root, n_seeds=n_seeds, base_seed=base_seed)
    analysis = analyze_pairs(pairs, z_value=z_value, alpha=alpha, power=power)
    analysis["query_bootstrap"] = bootstrap_query_accuracy(
        metrics_root, pairs, resamples=bootstrap_resamples
    )
    return analysis


def json_safe(value):
    """Convert non-finite floats recursively because strict JSON rejects them."""
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_safe(item) for item in value]
    return value


def print_analysis(label, analysis):
    mde = analysis["minimum_detectable_effect"]
    bootstrap = analysis["query_bootstrap"]
    print(f"\n--- {label} ({analysis['n_seeds']} paired seeds) ---")
    ratio = analysis["recovery_ratio"]
    print(f"Recovery ratio R: {'undefined' if ratio is None else f'{ratio:.6f}'}")
    print(
        f"full_retrain - adapted: mean={analysis['mean_paired_difference']:.6f}, "
        f"std={analysis['std_paired_difference']:.6f}, "
        f"SE={analysis['standard_error']:.6f}, tau={analysis['tau']:.6f} "
        f"(z={analysis['z_value']:.3f})"
    )
    effect = analysis["cohens_d_paired"]
    effect_text = "undefined" if effect is None else f"{effect:.6f}"
    print(f"Cohen's d_paired: {effect_text} ({analysis['absolute_effect_size_bin']})")
    print(f"Decision: {analysis['decision']}")
    print(
        f"Minimum detectable effect (alpha={mde['alpha']:.3f}, power={mde['power']:.2f}): "
        f"standardized={mde['standardized']:.6f}, accuracy={mde['accuracy']:.6f}"
    )
    print(
        "Mean within-seed 95% bootstrap CI width: "
        f"adapted={bootstrap['mean_adapted_interval_width']:.6f}, "
        f"full_retrain={bootstrap['mean_full_retrain_interval_width']:.6f}"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics-root", type=Path, default=ROOT / "results" / "metrics")
    parser.add_argument("--n-seeds", type=int, default=None)
    parser.add_argument("--base-seed", type=int, default=0)
    parser.add_argument("--compare-n-seeds", type=int, default=None)
    parser.add_argument("--control-root", type=Path, default=None)
    parser.add_argument("--control-n-seeds", type=int, default=None)
    parser.add_argument("--z-value", type=float, default=DEFAULT_Z_VALUE)
    parser.add_argument("--alpha", type=float, default=DEFAULT_ALPHA)
    parser.add_argument("--power", type=float, default=DEFAULT_POWER)
    parser.add_argument("--bootstrap-resamples", type=int, default=200)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    main_analysis = analyze_directory(
        args.metrics_root, args.n_seeds, args.base_seed, args.z_value,
        args.alpha, args.power, args.bootstrap_resamples,
    )
    report = {"analysis": main_analysis}
    print_analysis("Primary analysis", main_analysis)

    if args.compare_n_seeds is not None:
        comparison = analyze_directory(
            args.metrics_root, args.compare_n_seeds, args.base_seed, args.z_value,
            args.alpha, args.power, args.bootstrap_resamples,
        )
        report["comparison"] = comparison
        report["decision_changed"] = comparison["decision"] != main_analysis["decision"]
        old_mde = comparison["minimum_detectable_effect"]["accuracy"]
        new_mde = main_analysis["minimum_detectable_effect"]["accuracy"]
        report["mde_accuracy_reduction"] = old_mde - new_mde
        report["mde_accuracy_reduction_fraction"] = (
            None if old_mde == 0.0 else (old_mde - new_mde) / old_mde
        )
        print_analysis(f"Comparison prefix", comparison)
        print(f"Decision changed: {report['decision_changed']}")
        print(f"Raw-accuracy MDE reduction: {report['mde_accuracy_reduction']:.6f}")

    if args.control_root is not None:
        control = analyze_directory(
            args.control_root, args.control_n_seeds, args.base_seed, args.z_value,
            args.alpha, args.power, args.bootstrap_resamples,
        )
        passed = control["decision"] == "full retrain meaningfully better"
        report["control"] = control
        report["control_validation_passed"] = passed
        print_analysis("Adversarial control", control)
        print(f"Control validation: {'PASSED' if passed else 'FAILED'}")

    output = args.output or args.metrics_root / "adaptation_analysis.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        json.dump(json_safe(report), handle, indent=2, allow_nan=False)
    print(f"\nWrote analysis report to {output}")


if __name__ == "__main__":
    main()
