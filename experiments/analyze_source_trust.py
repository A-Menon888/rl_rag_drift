"""Tables for the source-reliability drift experiment (reads runs_seed*.jsonl from run_source_trust.py).

Usage: python experiments/analyze_source_trust.py --output-root <dir>
Writes per_run.csv next to the inputs and prints markdown tables.
Statistics: mean +- sample sd over runs (page seed x training seed); references
(optima, oracle) do not depend on the training seed, so their sd is over page seeds.
"""
import argparse
import collections
import csv
import json
from pathlib import Path

import numpy as np

ARMS = ("frozen", "dpo", "rl")


def load(root):
    stage1, cells = [], []
    for path in sorted(Path(root).glob("runs_seed*.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            record = json.loads(line)
            (stage1 if record["kind"] == "stage1" else cells).append(record)
    return stage1, cells


def ms(xs, digits=3):
    xs = [x for x in xs if x is not None]
    if not xs:
        return "n/a"
    sd = np.std(xs, ddof=1) if len(xs) > 1 else 0.0
    return f"{np.mean(xs):.{digits}f} ± {sd:.{digits}f}"


def table(header, rows):
    print("| " + " | ".join(header) + " |")
    print("|" + "---|" * len(header))
    for row in rows:
        print("| " + " | ".join(str(x) for x in row) + " |")
    print()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", required=True)
    args = parser.parse_args()
    stage1, cells = load(args.output_root)
    levels = sorted({c["level"] for c in cells})
    by_level = {lv: [c for c in cells if c["level"] == lv] for lv in levels}
    print(f"{len(stage1)} training seeds, {len(cells)} runs\n")

    print("## Stage 1: frozen KB-A policy\n")
    table(["train seed", "KB-A train acc / ret", "KB-A test acc", "KB-A test return", "searches (A/B)",
           "first action", "final action", "KB-A test optimum ret", "R2 pairs on KB-A (outcome-chg)", "root / A_answered"],
          [(s["train_seed"], f"{s['kb_a_train']['accuracy']:.3f} / {s['kb_a_train']['return']:.3f}",
            f"{s['kb_a_test']['accuracy']:.3f}", f"{s['kb_a_test']['return']:.3f}",
            f"{s['kb_a_test']['searches']:.2f} ({s['kb_a_test']['searches_a']:.2f}/{s['kb_a_test']['searches_b']:.2f})",
            s["kb_a_test"]["first_action"], s["kb_a_test"]["final_action"],
            f"{s['kb_a_test_optimum']['return']:.3f}",
            f"{s['r2_on_kb_a']['pairs']} ({s['r2_on_kb_a']['outcome_changing']})",
            f"{s['frozen_key_actions']['root']} / {s['frozen_key_actions']['A_answered']}") for s in stage1])
    s0 = stage1[0]
    print(f"Stage-1 budget per seed: {s0['budget']}\n")

    print("## Main table (KB-B held-out questions, mean ± sd over runs)\n")
    rows = []
    for lv in levels:
        cs = by_level[lv]
        refs = {c["page_seed"]: c for c in cs}.values()
        rows.append([f"{lv:.0%}"] + [ms([c[a]["test"]["return"] for c in cs]) for a in ARMS]
                    + [ms([c["heldout_optimum"]["return"] for c in refs]),
                       ms([c["train_fitted_exact"]["return"] for c in refs]), ms([c["oracle_return"] for c in refs])])
    table(["drift", "frozen return", "DPO return", "RL return", "held-out obs. optimum", "train-fitted exact",
           "per-question oracle"], rows)
    rows = []
    for lv in levels:
        cs = by_level[lv]
        refs = {c["page_seed"]: c for c in cs}.values()
        rows.append([f"{lv:.0%}"] + [ms([c[a]["test"]["accuracy"] for c in cs]) for a in ARMS]
                    + [ms([c["heldout_optimum"]["accuracy"] for c in refs])]
                    + [ms([c[a]["test"]["searches"] for c in cs], 2) for a in ARMS]
                    + [ms([c["heldout_optimum"]["searches"] for c in refs], 2)])
    table(["drift", "frozen acc", "DPO acc", "RL acc", "optimum acc", "frozen searches", "DPO searches",
           "RL searches", "optimum searches"], rows)

    print("## Improvement and gap closed (paired per run)\n")
    rows = []
    for lv in levels:
        cs = by_level[lv]
        d = {a: [c[a]["test"]["return"] - c["frozen"]["test"]["return"] for c in cs] for a in ("dpo", "rl")}
        diff = [c["dpo"]["test"]["return"] - c["rl"]["test"]["return"] for c in cs]
        gap = lambda a: [(c[a]["test"]["return"] - c["frozen"]["test"]["return"]) /
                         (c["heldout_optimum"]["return"] - c["frozen"]["test"]["return"])
                         for c in cs if c["heldout_optimum"]["return"] - c["frozen"]["test"]["return"] > 0.05]
        ratio = lambda a: [c[a]["test"]["return"] / c["heldout_optimum"]["return"] for c in cs]
        rows.append([f"{lv:.0%}", ms(d["dpo"]), ms(d["rl"]), ms(diff),
                     f"{sum(x > 1e-9 for x in diff)}/{sum(abs(x) <= 1e-9 for x in diff)}/{sum(x < -1e-9 for x in diff)}",
                     f"{ms(gap('dpo'))} (n={len(gap('dpo'))})", ms(gap("rl")), ms(ratio("dpo")), ms(ratio("rl"))])
    table(["drift", "DPO − frozen", "RL − frozen", "DPO − RL", "DPO>RL / tie / DPO<RL", "DPO gap closed",
           "RL gap closed", "DPO / optimum", "RL / optimum"], rows)

    print("## Per page seed (return, mean over training seeds; frozen | DPO | RL | optimum)\n")
    rows = []
    for lv in levels:
        row = [f"{lv:.0%}"]
        for ps in sorted({c["page_seed"] for c in by_level[lv]}):
            cs = [c for c in by_level[lv] if c["page_seed"] == ps]
            row.append(" | ".join(f"{np.mean([c[a]['test']['return'] for c in cs]):.2f}" for a in ARMS)
                       + f" | {cs[0]['heldout_optimum']['return']:.2f}")
        rows.append(row)
    table(["drift"] + [f"page seed {ps}" for ps in sorted({c["page_seed"] for c in cells})], rows)

    print("## Action distributions on held-out questions (mean share)\n")
    rows = []
    for lv in levels:
        cs = by_level[lv]
        for a in ARMS + ("heldout_optimum",):
            get = (lambda c: c[a]["test"]) if a in ARMS else (lambda c: c[a])
            first = collections.defaultdict(float)
            final = collections.defaultdict(float)
            for c in cs:
                for k, v in get(c)["first_action"].items():
                    first[k] += v / len(cs)
                for k, v in get(c)["final_action"].items():
                    final[k] += v / len(cs)
            rows.append([f"{lv:.0%}", a, ", ".join(f"{k} {v:.2f}" for k, v in sorted(first.items())),
                         ", ".join(f"{k} {v:.2f}" for k, v in sorted(final.items())),
                         f"{np.mean([get(c)['searches_a'] for c in cs]):.2f} / {np.mean([get(c)['searches_b'] for c in cs]):.2f}",
                         ms([get(c)["accuracy_stale_cache"] for c in cs]), ms([get(c)["accuracy_fresh_cache"] for c in cs])])
    table(["drift", "policy", "first action", "final action", "searches A / B", "acc stale-cache q", "acc fresh-cache q"],
          rows)

    print("## Greedy action at key states (counts over runs)\n")
    rows = []
    for lv in levels:
        cs = by_level[lv]
        for name in ("root", "A_answered", "A_silent", "B_answered"):
            cnt = lambda get: dict(collections.Counter(get(c)[name] for c in cs))
            rows.append([f"{lv:.0%}", name, cnt(lambda c: c["frozen_key_actions"]), cnt(lambda c: c["dpo"]["key_actions"]),
                         cnt(lambda c: c["rl"]["key_actions"]), cnt(lambda c: c["train_fitted_exact_key_actions"]),
                         cnt(lambda c: c["heldout_optimum_key_actions"])])
    table(["drift", "state", "frozen", "DPO", "RL", "train-fitted exact", "held-out optimum"], rows)

    print("## R2 preference signal and fit/generalization diagnostics\n")
    rows = []
    for lv in levels:
        cs = by_level[lv]
        prefs = collections.Counter()
        for c in cs:
            prefs.update(c["r2"]["preferred_actions"])
        rows.append([f"{lv:.0%}", ms([c["r2"]["pairs"] for c in cs], 1), ms([c["r2"]["outcome_changing"] for c in cs], 1),
                     ms([c["r2"]["at_root"] for c in cs], 1), ms([c["r2"]["individually_worse"] for c in cs], 1),
                     f"{sum(c['r2']['heldout_match'] for c in cs)}/{sum(c['r2']['heldout_supported'] for c in cs)}",
                     dict(prefs.most_common(4)),
                     f"{sum(c['dpo']['pair_states_now_preferred'] for c in cs)}/{sum(c['r2']['pairs'] for c in cs)}",
                     f"{sum(c['rl']['pair_states_now_preferred'] for c in cs)}/{sum(c['r2']['pairs'] for c in cs)}"])
    table(["drift", "pairs", "outcome-changing", "at root", "chosen worse for its q", "pref = held-out opt",
           "frozen->preferred (top)", "DPO follows pref", "RL follows pref"], rows)
    rows = []
    for lv in levels:
        cs = by_level[lv]
        rows.append([f"{lv:.0%}"] + [f"{ms([c[a]['train']['return'] for c in cs])} / {ms([c[a]['test']['return'] for c in cs])}"
                                     for a in ARMS] + [ms([c["train_optimum_return"] for c in cs])])
    table(["drift", "frozen train / test", "DPO train / test", "RL train / test", "train obs. optimum"], rows)

    print("## Adaptation budget per run (mean)\n")
    rows = []
    for a in ("dpo", "rl"):
        b = [c[a]["budget"] for c in cells]
        keys = sorted({k for x in b for k in x if isinstance(x[k], (int, float))})
        rows += [[a, k, f"{np.mean([x[k] for x in b if k in x]):.2f}"] for k in keys]
    table(["arm", "quantity", "mean"], rows)

    with open(Path(args.output_root) / "per_run.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["level", "page_seed", "train_seed", "stale_pages"]
                   + [f"{a}_{m}" for a in ARMS for m in ("return", "accuracy", "searches")]
                   + ["optimum_return", "optimum_accuracy", "train_fitted_return", "oracle_return", "r2_pairs",
                      "r2_outcome_changing", "dpo_updates", "rl_updates"])
        for c in sorted(cells, key=lambda c: (c["level"], c["page_seed"], c["train_seed"])):
            w.writerow([c["level"], c["page_seed"], c["train_seed"], "+".join(c["stale_pages"])]
                       + [round(c[a]["test"][m], 4) for a in ARMS for m in ("return", "accuracy", "searches")]
                       + [round(c["heldout_optimum"]["return"], 4), round(c["heldout_optimum"]["accuracy"], 4),
                          round(c["train_fitted_exact"]["return"], 4), round(c["oracle_return"], 4), c["r2"]["pairs"],
                          c["r2"]["outcome_changing"], c["dpo"]["budget"]["gradient_updates"],
                          c["rl"]["budget"]["gradient_updates"]])


if __name__ == "__main__":
    main()
