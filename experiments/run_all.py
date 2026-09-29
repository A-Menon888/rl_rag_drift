import csv, json, os, sys
from pathlib import Path
import numpy as np
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data.documents import load_knowledge_base
from src.data.facts import build_fact_queries, describe_queries, load_facts
from src.environment.rl_rag_env import RLRAGEnv
from src.agents.baselines import AnswerDirectly, RandomPolicy, SearchThenAnswer
from src.agents.rl_agent import RLAgent
from src.evaluation.metrics import summarize
from src.evaluation.plots import plot_series

ROOT = Path(__file__).resolve().parents[1]
TRAINED_POLICIES = ("old_policy", "adapted_policy", "full_retrain_policy")
APPROVAL_BASELINE = "search_once"


def make_env(queries, chunks, config):
    return RLRAGEnv(
        queries, chunks,
        top_k=config["top_k"],
        max_searches=config["max_searches"],
        search_cost=config["retrieval_cost"],
        correct_reward=config["correct_reward"],
        incorrect_reward=config["incorrect_reward"],
        give_up_reward=config["give_up_reward"],
    )


def make_agent(env, config):
    return RLAgent(env.observation_space.shape[0], config["learning_rate"], config["seed"],
                   entropy_coef=config["entropy_coef"])


def evaluate_policy(policy, env, explore=False):
    """Run every question once, in order; return (summary, terminal infos)."""
    infos = []
    for index in range(len(env.queries)):
        observation, info = env.reset(options={"query_index": index})
        terminated = False
        while not terminated:
            observation, _, terminated, _, info = env.step(policy.act(observation, info, explore=explore))
        infos.append(info)
    return summarize(infos), infos


def train_policy(policy, env, epochs, batch_size, seed):
    """Each epoch visits every question once in a seeded random order.

    Returns the mean training episode return of each epoch.
    """
    rng = np.random.default_rng(seed)
    curve = []
    for _ in range(epochs):
        order = rng.permutation(len(env.queries))
        infos = []
        for start in range(0, len(order), batch_size):
            infos += policy.train_batch(env, order[start:start + batch_size].tolist())
        curve.append(summarize(infos)["average_reward"])
    return curve


def result_row(policy_name, knowledge_base, metrics, training_final_return=""):
    return {"policy": policy_name, "knowledge_base": knowledge_base,
            "training_final_return": training_final_return, **metrics}


def approval_decision(candidate, baseline, margin=0.0):
    """Approve when accuracy clears the baseline bar and cost stays bounded."""
    accuracy_bar = baseline["accuracy"] - margin
    cost_bar = baseline["retrieval_cost"] + margin
    approved = candidate["accuracy"] > accuracy_bar and candidate["retrieval_cost"] <= cost_bar
    return {
        "approved": approved,
        "baseline_accuracy": baseline["accuracy"],
        "baseline_retrieval_cost": baseline["retrieval_cost"],
        "accuracy_bar": accuracy_bar,
        "retrieval_cost_bar": cost_bar,
    }


def recovery_decision(candidate, pre_drift_baseline):
    """Recovery means matching or exceeding the old policy's KB-A accuracy."""
    baseline_accuracy = pre_drift_baseline["accuracy"]
    return {
        "recovered": candidate["accuracy"] >= baseline_accuracy,
        "baseline_accuracy": baseline_accuracy,
    }


def run_experiment(config, output_root=ROOT / "results", summary_path=None):
    output_root = Path(output_root)
    summary_path = Path(summary_path) if summary_path else output_root / "metrics" / "summary.csv"
    for sub in ("metrics", "checkpoints", "figures"):
        os.makedirs(output_root / sub, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    # --- Load the memory snapshot and both KBs; build fact-level queries ---
    kb_0_chunks, kb_a_chunks, kb_b_chunks = (
        load_knowledge_base(ROOT / config["corpus_dir"], name) for name in ("kb_0", "kb_a", "kb_b")
    )
    queries_a, queries_b = build_fact_queries(
        load_facts(ROOT / config["facts_path"]), kb_0_chunks, kb_a_chunks, kb_b_chunks
    )
    for label, queries in (("KB-A", queries_a), ("KB-B", queries_b)):
        print(f"{label} queries={len(queries)} {describe_queries(queries)}")

    env_a = make_env(queries_a, kb_a_chunks, config)
    env_b = make_env(queries_b, kb_b_chunks, config)
    epochs, batch_size, seed = config["train_epochs"], config["batch_size"], config["seed"]

    all_rows = []
    kb_b_infos = {}

    def train_and_record(name, agent, env, kb_label):
        curve = train_policy(agent, env, epochs, batch_size, seed)
        torch.save(agent.policy.state_dict(), output_root / f"checkpoints/rl_{name}_{kb_label}.pt")
        plot_series(curve, output_root / f"figures/{name}_{kb_label}_training.png",
                    "Mean episode return", f"{name} training on {kb_label}")
        return curve[-1]

    # ── Arm 1: Old policy trained on KB-A, then frozen ───────────────────────
    old_policy = make_agent(env_a, config)
    old_final_return = train_and_record("old_policy", old_policy, env_a, "kb_a")
    for kb_label, env in (("kb_a", env_a), ("kb_b", env_b)):
        metrics, infos = evaluate_policy(old_policy, env)
        if kb_label == "kb_b":
            kb_b_infos["old_policy"] = infos
        all_rows.append(result_row("old_policy", kb_label, metrics, old_final_return if kb_label == "kb_a" else ""))

    # ── Arm 2: Adapted policy — starts from old weights, trains on KB-B ──────
    adapted_policy = make_agent(env_b, config)
    adapted_policy.policy.load_state_dict(old_policy.policy.state_dict())
    adapted_final_return = train_and_record("adapted_policy", adapted_policy, env_b, "kb_b")

    # ── Arm 3: Full retrain from scratch on KB-B ─────────────────────────────
    retrain_policy = make_agent(env_b, config)
    retrain_final_return = train_and_record("full_retrain_policy", retrain_policy, env_b, "kb_b")

    for name, agent, final_return in (("adapted_policy", adapted_policy, adapted_final_return),
                                      ("full_retrain_policy", retrain_policy, retrain_final_return)):
        metrics, kb_b_infos[name] = evaluate_policy(agent, env_b)
        all_rows.append(result_row(name, "kb_b", metrics, final_return))

    # ── Baselines ─────────────────────────────────────────────────────────────
    baselines = {
        "answer_directly": AnswerDirectly(),
        "search_once": SearchThenAnswer(1),
        "search_all": SearchThenAnswer(config["max_searches"]),
        "random": RandomPolicy(seed),
    }
    for name, policy in baselines.items():
        for kb_label, env in (("kb_a", env_a), ("kb_b", env_b)):
            all_rows.append(result_row(name, kb_label, evaluate_policy(policy, env, explore=True)[0]))

    # ── Baseline-relative approval and recovery decisions ────────────────────
    rows = {(r["policy"], r["knowledge_base"]): r for r in all_rows}
    approval_baseline = rows[(APPROVAL_BASELINE, "kb_b")]
    old_policy_a = rows[("old_policy", "kb_a")]
    approval_margin = config.get("approval_margin", 0.0)

    print("\n--- Evaluation on KB-B (3-way) ---")
    for name in TRAINED_POLICIES:
        r = rows[(name, "kb_b")]
        print(f"  {name:<22} Acc={r['accuracy']:.3f} Return={r['average_reward']:.3f} "
              f"Searches={r['average_searches']:.2f} GiveUp={r['give_up_rate']:.3f} "
              f"Drifted={r['drifted_accuracy']:.3f} Stable={r['stable_accuracy']:.3f}")

    decisions = {name: approval_decision(rows[(name, "kb_b")], approval_baseline, approval_margin)
                 for name in ("adapted_policy", "full_retrain_policy")}
    recovery = {name: recovery_decision(rows[(name, "kb_b")], old_policy_a) for name in TRAINED_POLICIES}
    print(f"\n  Approval baseline ({APPROVAL_BASELINE}/kb_b): accuracy={approval_baseline['accuracy']:.3f} "
          f"retrieval_cost={approval_baseline['retrieval_cost']:.3f} margin={approval_margin:.3f}")
    for name, decision in decisions.items():
        print(f"  {name:<22} -> {'APPROVED' if decision['approved'] else 'DECLINED'}")
    print(f"  Recovery baseline (old_policy/kb_a): accuracy={old_policy_a['accuracy']:.3f}")
    for name, decision in recovery.items():
        print(f"  {name:<22} recovery={'RECOVERED' if decision['recovered'] else 'NOT_RECOVERED'}")

    for row in all_rows:
        row["approval_baseline_accuracy"] = approval_baseline["accuracy"]
        row["approval_baseline_retrieval_cost"] = approval_baseline["retrieval_cost"]
        row["recovery_baseline_accuracy"] = old_policy_a["accuracy"]
        on_b = row["knowledge_base"] == "kb_b"
        row["approval_status"] = (("APPROVED" if decisions[row["policy"]]["approved"] else "DECLINED")
                                  if on_b and row["policy"] in decisions else "N/A")
        row["recovery_status"] = (("RECOVERED" if recovery[row["policy"]]["recovered"] else "NOT_RECOVERED")
                                  if on_b and row["policy"] in decisions else "N/A")

    with open(summary_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(all_rows[0].keys()))
        writer.writeheader()
        writer.writerows(all_rows)

    # Paired per-question KB-B records for seed-level paired analysis.
    query_rows = []
    for index, query in enumerate(queries_b):
        row = {"seed": seed, "query_index": index, "query_id": query.query_id,
               "affected_by_drift": query.affected_by_drift, "drift_type": query.drift_type,
               "memory_status": query.memory_status}
        for name in TRAINED_POLICIES:
            info = kb_b_infos[name][index]
            assert info["query_id"] == query.query_id
            row[f"{name}_correct"] = int(bool(info["correct"]))
            row[f"{name}_action"] = info["final_action"]
            row[f"{name}_searches"] = info["searches"]
        query_rows.append(row)
    query_results_path = summary_path.parent / "kb_b_query_results.csv"
    with open(query_results_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(query_rows[0].keys()))
        writer.writeheader()
        writer.writerows(query_rows)
    with open(summary_path.parent / "config.json", "w", encoding="utf-8") as handle:
        json.dump(config, handle, indent=2)
    print(f"Wrote {len(all_rows)} result rows to {summary_path}")
    print(f"Wrote {len(query_rows)} paired KB-B query rows to {query_results_path}")
    return all_rows


def main():
    with open(ROOT / "configs/default.yaml", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    run_experiment(config)


if __name__ == "__main__":
    main()
