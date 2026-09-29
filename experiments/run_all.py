import csv, json, os, sys
from pathlib import Path
import numpy as np
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data.documents import load_knowledge_base
from src.data.facts import build_fact_queries, describe_queries, load_facts, select_facts, split_facts
from src.environment.rl_rag_env import RLRAGEnv
from src.agents.baselines import AnswerDirectly, RandomPolicy, SearchThenAnswer
from src.agents.rl_agent import RLAgent
from src.agents.dpo import (build_preference_pairs, collect_trajectories, frozen_policy_actions,
                            stopping_strategies, train_dpo)
from src.environment.rl_rag_env import FEATURES
from src.evaluation.metrics import summarize
from src.evaluation.plots import plot_series

ROOT = Path(__file__).resolve().parents[1]
TRAINED_POLICIES = ("old_policy", "dpo_policy", "rl_finetune_policy", "full_retrain_policy")
ADAPTED_POLICIES = TRAINED_POLICIES[1:]
APPROVAL_BASELINE = "search_once"
REQUIRED_CONFIG = {
    "seed": int, "corpus_dir": str, "facts_path": str, "test_fraction": float, "split_seed": int,
    "top_k": int, "max_searches": int, "observe_query_embedding": bool,
    "train_epochs": int, "batch_size": int, "learning_rate": float, "entropy_coef": float,
    "retrieval_cost": float, "correct_reward": float, "incorrect_reward": float, "give_up_reward": float,
    "dpo_epochs": int, "dpo_batch_size": int, "dpo_beta": float, "dpo_learning_rate": float,
}


def validate_config(config):
    missing = sorted(set(REQUIRED_CONFIG) - set(config))
    if missing:
        raise ValueError(f"config is missing: {missing}")
    for key, expected in REQUIRED_CONFIG.items():
        value = config[key]
        if expected is bool:
            ok = isinstance(value, bool)
        elif expected is float:
            ok = isinstance(value, (int, float)) and not isinstance(value, bool)
        else:
            ok = isinstance(value, expected) and not isinstance(value, bool)
        if not ok:
            raise ValueError(f"config[{key!r}] must be {expected.__name__}, got {value!r}")


def make_env(queries, chunks, config):
    return RLRAGEnv(
        queries, chunks,
        top_k=config["top_k"],
        max_searches=config["max_searches"],
        search_cost=config["retrieval_cost"],
        correct_reward=config["correct_reward"],
        incorrect_reward=config["incorrect_reward"],
        give_up_reward=config["give_up_reward"],
        observe_query_embedding=config["observe_query_embedding"],
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

    Returns (mean training episode return per epoch, training budget).
    """
    rng = np.random.default_rng(seed)
    curve = []
    budget = {"adaptation_episodes": 0, "gradient_updates": 0, "policy_step_evaluations": 0}
    for _ in range(epochs):
        order = rng.permutation(len(env.queries))
        infos = []
        for start in range(0, len(order), batch_size):
            batch_infos = policy.train_batch(env, order[start:start + batch_size].tolist())
            budget["gradient_updates"] += 1
            budget["policy_step_evaluations"] += sum(info["searches"] + 1 for info in batch_infos)
            infos += batch_infos
        budget["adaptation_episodes"] += len(infos)
        curve.append(summarize(infos)["average_reward"])
    return curve, budget


BUDGET_FIELDS = ("adaptation_episodes", "gradient_updates", "policy_step_evaluations")


def result_row(policy_name, knowledge_base, split, metrics, training_final_return="", budget=None):
    """`budget` is the environment/compute cost of producing the policy from its starting point."""
    budget = budget or {}
    return {"policy": policy_name, "knowledge_base": knowledge_base, "split": split,
            "training_final_return": training_final_return, **metrics,
            **{field: budget.get(field, "") for field in BUDGET_FIELDS}}


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
    validate_config(config)
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
    # --- Fact-level train/test split. Both splits query the same full KB; ---
    # --- only the questions differ. Test facts are never trained on.     ---
    train_facts, test_facts = split_facts(queries_b, config["test_fraction"], config["split_seed"])
    queries = {
        ("kb_a", "train"): select_facts(queries_a, train_facts),
        ("kb_a", "test"): select_facts(queries_a, test_facts),
        ("kb_b", "train"): select_facts(queries_b, train_facts),
        ("kb_b", "test"): select_facts(queries_b, test_facts),
    }
    for (kb_label, split), split_queries in queries.items():
        print(f"{kb_label} {split:<5} queries={len(split_queries)} {describe_queries(split_queries)}")
    chunks = {"kb_a": kb_a_chunks, "kb_b": kb_b_chunks}
    envs = {key: make_env(split_queries, chunks[key[0]], config) for key, split_queries in queries.items()}
    for key in (("kb_a", "train"), ("kb_b", "train")):
        assert not {query.fact_id for query in envs[key].queries} & test_facts, "test fact in training set"

    epochs, batch_size, seed = config["train_epochs"], config["batch_size"], config["seed"]
    all_rows = []
    kb_b_infos = {}

    def train_and_record(name, agent, env, kb_label):
        curve, budget = train_policy(agent, env, epochs, batch_size, seed)
        agent.save(output_root / f"checkpoints/rl_{name}_{kb_label}.pt")
        plot_series(curve, output_root / f"figures/{name}_{kb_label}_training.png",
                    "Mean episode return", f"{name} training on {kb_label}")
        return curve[-1], budget

    # ── Arm 1: Old policy trained on KB-A train questions, then frozen ───────
    # Evaluated on KB-A train (fit), KB-A test (pre-drift baseline) and
    # KB-B test (frozen-policy degradation on the same held-out facts).
    old_policy = make_agent(envs[("kb_a", "train")], config)
    old_final_return, old_budget = train_and_record("old_policy", old_policy, envs[("kb_a", "train")], "kb_a")
    old_checkpoint = output_root / "checkpoints/rl_old_policy_kb_a.pt"
    for key in (("kb_a", "train"), ("kb_a", "test"), ("kb_b", "test")):
        metrics, infos = evaluate_policy(old_policy, envs[key])
        if key == ("kb_b", "test"):
            kb_b_infos["old_policy"] = infos
        if key[1] == "train":
            all_rows.append(result_row("old_policy", *key, metrics, old_final_return, old_budget))
        else:
            all_rows.append(result_row("old_policy", *key, metrics))

    # ── Arm 2: DPO adaptation of the frozen policy from KB-B trajectory pairs ─
    # Reference and trainable policies both start from the saved old policy.
    # Trajectories come only from KB-B *train* questions; the generator stays frozen.
    env_b_train = envs[("kb_b", "train")]
    reference_policy = make_agent(env_b_train, config).load(old_checkpoint)
    dpo_policy = make_agent(env_b_train, config).load(old_checkpoint)
    trajectories = collect_trajectories(env_b_train)
    pairs = build_preference_pairs(trajectories, frozen_policy_actions(reference_policy, env_b_train))
    dpo_stats = train_dpo(dpo_policy, reference_policy, pairs, epochs=config["dpo_epochs"],
                          batch_size=config["dpo_batch_size"], beta=config["dpo_beta"],
                          learning_rate=config["dpo_learning_rate"], seed=seed)
    dpo_policy.save(output_root / "checkpoints/dpo_policy_kb_b.pt")
    plot_series(dpo_stats["loss_curve"], output_root / "figures/dpo_policy_kb_b_training.png",
                "Mean DPO loss", "dpo_policy training on kb_b")
    dpo_budget = {"adaptation_episodes": sum(len(t) for t in trajectories),
                  "gradient_updates": dpo_stats["gradient_updates"],
                  "policy_step_evaluations": dpo_stats["policy_step_evaluations"]}
    write_preference_pairs(summary_path.parent / "dpo_pairs.csv", pairs)
    print(f"DPO: {len(pairs)} preference pairs from {dpo_budget['adaptation_episodes']} KB-B train episodes; "
          f"loss {dpo_stats['loss_curve'][0]:.3f} -> {dpo_stats['loss_curve'][-1]:.3f}")

    # ── Arm 2b: equal-data control — REINFORCE fine-tuning of the old policy ──
    # Exactly as many KB-B train episodes as DPO used (whole epochs over the
    # train questions); same REINFORCE settings as the other RL arms.
    matched_epochs, remainder = divmod(dpo_budget["adaptation_episodes"], len(env_b_train.queries))
    assert remainder == 0, "DPO episode budget must be a whole number of epochs"
    rl_finetune_policy = make_agent(env_b_train, config).load(old_checkpoint)
    rl_finetune_curve, rl_finetune_budget = train_policy(rl_finetune_policy, env_b_train, matched_epochs,
                                                         batch_size, seed)
    assert rl_finetune_budget["adaptation_episodes"] == dpo_budget["adaptation_episodes"]
    rl_finetune_policy.save(output_root / "checkpoints/rl_finetune_policy_kb_b.pt")
    plot_series(rl_finetune_curve, output_root / "figures/rl_finetune_policy_kb_b_training.png",
                "Mean episode return", "rl_finetune_policy training on kb_b")

    # ── Arm 3: Full RL retrain from scratch on KB-B train questions ──────────
    retrain_policy = make_agent(env_b_train, config)
    retrain_final_return, retrain_budget = train_and_record("full_retrain_policy", retrain_policy, env_b_train, "kb_b")

    for name, agent, final_return, budget in (
            ("dpo_policy", dpo_policy, "", dpo_budget),
            ("rl_finetune_policy", rl_finetune_policy, rl_finetune_curve[-1], rl_finetune_budget),
            ("full_retrain_policy", retrain_policy, retrain_final_return, retrain_budget)):
        all_rows.append(result_row(name, "kb_b", "train", evaluate_policy(agent, env_b_train)[0],
                                   final_return, budget))
        metrics, kb_b_infos[name] = evaluate_policy(agent, envs[("kb_b", "test")])
        all_rows.append(result_row(name, "kb_b", "test", metrics, budget=budget))

    # ── Baselines ─────────────────────────────────────────────────────────────
    baselines = {
        "answer_directly": AnswerDirectly(),
        "search_once": SearchThenAnswer(1),
        "search_all": SearchThenAnswer(config["max_searches"]),
        "random": RandomPolicy(seed),
    }
    for name, policy in baselines.items():
        for key in (("kb_a", "test"), ("kb_b", "test")):
            all_rows.append(result_row(name, *key, evaluate_policy(policy, envs[key], explore=True)[0]))

    # ── Baseline-relative approval and recovery decisions (held-out facts) ───
    rows = {(r["policy"], r["knowledge_base"], r["split"]): r for r in all_rows}
    approval_baseline = rows[(APPROVAL_BASELINE, "kb_b", "test")]
    old_policy_a = rows[("old_policy", "kb_a", "test")]
    approval_margin = config.get("approval_margin", 0.0)

    print("\n--- Held-out (test) evaluation ---")
    print(f"  {'old_policy / kb_a':<30} Acc={old_policy_a['accuracy']:.3f} Return={old_policy_a['average_reward']:.3f}"
          f"  (kb_a train Acc={rows[('old_policy', 'kb_a', 'train')]['accuracy']:.3f})")
    for name in TRAINED_POLICIES:
        r = rows[(name, "kb_b", "test")]
        print(f"  {name + ' / kb_b':<30} Acc={r['accuracy']:.3f} Return={r['average_reward']:.3f} "
              f"Searches={r['average_searches']:.2f} GiveUp={r['give_up_rate']:.3f} "
              f"Drifted={r['drifted_accuracy']:.3f} Stable={r['stable_accuracy']:.3f}")

    decisions = {name: approval_decision(rows[(name, "kb_b", "test")], approval_baseline, approval_margin)
                 for name in ADAPTED_POLICIES}
    recovery = {name: recovery_decision(rows[(name, "kb_b", "test")], old_policy_a) for name in TRAINED_POLICIES}
    print(f"\n  Approval baseline ({APPROVAL_BASELINE}/kb_b/test): accuracy={approval_baseline['accuracy']:.3f} "
          f"retrieval_cost={approval_baseline['retrieval_cost']:.3f} margin={approval_margin:.3f}")
    for name, decision in decisions.items():
        print(f"  {name:<22} -> {'APPROVED' if decision['approved'] else 'DECLINED'}")
    print(f"  Recovery baseline (old_policy/kb_a/test): accuracy={old_policy_a['accuracy']:.3f}")
    for name, decision in recovery.items():
        print(f"  {name:<22} recovery={'RECOVERED' if decision['recovered'] else 'NOT_RECOVERED'}")

    for row in all_rows:
        row["approval_baseline_accuracy"] = approval_baseline["accuracy"]
        row["approval_baseline_retrieval_cost"] = approval_baseline["retrieval_cost"]
        row["recovery_baseline_accuracy"] = old_policy_a["accuracy"]
        on_b = (row["knowledge_base"], row["split"]) == ("kb_b", "test")
        row["approval_status"] = (("APPROVED" if decisions[row["policy"]]["approved"] else "DECLINED")
                                  if on_b and row["policy"] in decisions else "N/A")
        row["recovery_status"] = (("RECOVERED" if recovery[row["policy"]]["recovered"] else "NOT_RECOVERED")
                                  if on_b and row["policy"] in decisions else "N/A")

    with open(summary_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(all_rows[0].keys()))
        writer.writeheader()
        writer.writerows(all_rows)

    # Paired per-question KB-B test records for seed-level paired analysis.
    query_rows = []
    for index, query in enumerate(queries[("kb_b", "test")]):
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
    with open(summary_path.parent / "split.json", "w", encoding="utf-8") as handle:
        json.dump({"train_facts": sorted(train_facts), "test_facts": sorted(test_facts)}, handle, indent=2)
    metadata = {
        "observation": {
            "observe_query_embedding": config["observe_query_embedding"],
            "features": list(FEATURES),
            "dimension": int(env_b_train.observation_space.shape[0]),
        },
        "adaptation_method": "dpo",
        "dpo": {"pair_construction": "best enumerated trajectory (chosen) vs frozen-policy greedy trajectory "
                                     "(rejected), one per question, only if chosen return is higher and the "
                                     "answer outcome (final action, correct) differs",
                "questions": len(env_b_train.queries),
                "preference_pairs": len(pairs), "questions_with_pairs": len({p.query_id for p in pairs}),
                "trajectories_per_question": len(stopping_strategies(config["max_searches"])),
                "loss_curve": dpo_stats["loss_curve"], **dpo_budget},
        "rl_finetune": {"epochs": matched_epochs, **rl_finetune_budget},
        "full_retrain": retrain_budget,
        "old_policy": old_budget,
    }
    with open(summary_path.parent / "metadata.json", "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)
    print(f"Wrote {len(all_rows)} result rows to {summary_path}")
    print(f"Wrote {len(query_rows)} paired KB-B test query rows to {query_results_path}")
    return all_rows


def write_preference_pairs(path, pairs):
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["query_id", "chosen", "chosen_return", "chosen_outcome",
                         "rejected", "rejected_return", "rejected_outcome"])
        for pair in pairs:
            outcome = lambda t: f"{t.outcome[0]}_{'right' if t.outcome[1] else 'wrong'}"
            writer.writerow([pair.query_id, pair.chosen.label, pair.chosen.episode_return, outcome(pair.chosen),
                             pair.rejected.label, pair.rejected.episode_return, outcome(pair.rejected)])


def main():
    with open(ROOT / "configs/default.yaml", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    run_experiment(config)


if __name__ == "__main__":
    main()
