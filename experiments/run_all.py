import csv, json, os, sys
from pathlib import Path
import yaml



















sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data.documents import generate_document_queries, load_knowledge_base
from src.environment.rl_rag_env import RLRAGEnv
from src.agents.baselines import AlwaysDirect, AlwaysRetrieve, RandomPolicy
from src.agents.rl_agent import RLAgent
from src.evaluation.metrics import summarize
from src.evaluation.plots import plot_series

ROOT = Path(__file__).resolve().parents[1]


def make_env(queries, snapshots, config, seed=None):
    return RLRAGEnv(
        queries, snapshots, {},
        seed=seed if seed is not None else config["seed"],
        retrieval_cost=config["retrieval_cost"],
        top_k=config["top_k"],
    )


def run_policy(policy, env, episodes, deterministic=False):
    history = []
    for _ in range(episodes):
        obs, _ = env.reset()
        infos = []
        for _ in range(len(env.queries)):
            action = policy.act(obs, explore=False) if deterministic else policy.act(obs)
            obs, _, _, _, info = env.step(action)
            infos.append(info)
        history.append(summarize(infos))
    return history


def evaluate_policy(policy, env, episodes=3):
    return run_policy(policy, env, episodes, deterministic=True)


def train_policy(policy, env, episodes):
    rewards = []
    for _ in range(episodes):
        infos, _ = policy.train_episode(env)
        rewards.append(summarize(infos)["average_reward"])
    return sum(rewards) / len(rewards)


def result_row(policy_name, knowledge_base, metrics, training_average_reward=""):
    return {"policy": policy_name, "knowledge_base": knowledge_base,
            "training_average_reward": training_average_reward, **metrics}


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
    os.makedirs(output_root / "metrics", exist_ok=True)
    os.makedirs(output_root / "checkpoints", exist_ok=True)
    os.makedirs(output_root / "figures", exist_ok=True)

    # --- Load both KBs and generate aligned queries together ---
    kb_a_chunks = load_knowledge_base(ROOT / config["corpus_dir"], "kb_a")
    kb_b_chunks = load_knowledge_base(ROOT / config["corpus_dir"], "kb_b")
    queries_a, queries_b = generate_document_queries(
        kb_a_chunks, kb_b_chunks, seed=config["seed"]
    )

    snapshots_a = {0: kb_a_chunks}
    snapshots_b = {0: kb_b_chunks}

    all_rows = []
    import torch

    # ── Arm 1: Old policy trained on KB-A ────────────────────────────────────
    old_policy = RLAgent(11, config["learning_rate"], config["seed"])
    train_a_env = make_env(queries_a, snapshots_a, config)
    training_average_reward = train_policy(old_policy, train_a_env, config["train_episodes"])
    torch.save(old_policy.policy.state_dict(), output_root / "checkpoints/rl_old_policy_kb_a.pt")

    for policy_name, kb_label, policy, queries, snapshots in [
        ("old_policy", "kb_a", old_policy, queries_a, snapshots_a),
        ("old_policy", "kb_b", old_policy, queries_b, snapshots_b),
    ]:
        env = make_env(queries, snapshots, config)
        history = evaluate_policy(policy, env)
        all_rows.append(result_row(
            policy_name, kb_label, history[-1],
            training_average_reward if kb_label == "kb_a" else "",
        ))
        plot_series(
            [item["average_reward"] for item in history],
            output_root / f"figures/{policy_name}_{kb_label}_reward.png",
            "Reward", f"{policy_name} on {kb_label}",
        )

    # ── Arm 2: Adapted policy — starts from old weights, trains on KB-B ──────
    adapted_policy = RLAgent(11, config["learning_rate"], config["seed"])
    adapted_policy.policy.load_state_dict(old_policy.policy.state_dict())
    train_b_env = make_env(queries_b, snapshots_b, config)
    adapted_training_reward = train_policy(adapted_policy, train_b_env, config["train_episodes"])
    torch.save(adapted_policy.policy.state_dict(), output_root / "checkpoints/rl_adapted_policy_kb_b.pt")
    eval_b_env = make_env(queries_b, snapshots_b, config)
    adapted_history = evaluate_policy(adapted_policy, eval_b_env)
    all_rows.append(result_row("adapted_policy", "kb_b", adapted_history[-1], adapted_training_reward))
    plot_series(
        [item["average_reward"] for item in adapted_history],
        output_root / "figures/adapted_policy_kb_b_reward.png",
        "Reward", "adapted_policy on kb_b",
    )

    # ── Arm 3: Full retrain from scratch on KB-B ─────────────────────────────
    retrain_policy = RLAgent(11, config["learning_rate"], config["seed"])
    retrain_b_env = make_env(queries_b, snapshots_b, config)
    retrain_training_reward = train_policy(retrain_policy, retrain_b_env, config["train_episodes"])
    torch.save(retrain_policy.policy.state_dict(), output_root / "checkpoints/rl_full_retrain_policy_kb_b.pt")
    eval_retrain_env = make_env(queries_b, snapshots_b, config)
    retrain_history = evaluate_policy(retrain_policy, eval_retrain_env)
    all_rows.append(result_row("full_retrain_policy", "kb_b", retrain_history[-1], retrain_training_reward))
    plot_series(
        [item["average_reward"] for item in retrain_history],
        output_root / "figures/full_retrain_policy_kb_b_reward.png",
        "Reward", "full_retrain_policy on kb_b",
    )

    # ── Baselines ─────────────────────────────────────────────────────────────
    for policy_name, policy in {
        "always_direct": AlwaysDirect(),
        "always_retrieve": AlwaysRetrieve(),
        "random": RandomPolicy(config["seed"]),
    }.items():
        for kb_label, queries, snapshots in [
            ("kb_a", queries_a, snapshots_a),
            ("kb_b", queries_b, snapshots_b),
        ]:
            env = make_env(queries, snapshots, config)
            history = run_policy(policy, env, 3)
            all_rows.append(result_row(policy_name, kb_label, history[-1]))
            plot_series(
                [item["average_reward"] for item in history],
                output_root / f"figures/{policy_name}_{kb_label}_reward.png",
                "Reward", f"{policy_name} on {kb_label}",
            )

    # ── Baseline-relative approval and recovery decisions ────────────────────
    kb_b_rows = {r["policy"]: r for r in all_rows if r["knowledge_base"] == "kb_b"
                 and r["policy"] in {"old_policy", "adapted_policy", "full_retrain_policy"}}
    always_retrieve_b = next(r for r in all_rows
                             if r["policy"] == "always_retrieve" and r["knowledge_base"] == "kb_b")
    old_policy_a = next(r for r in all_rows
                        if r["policy"] == "old_policy" and r["knowledge_base"] == "kb_a")
    approval_margin = config.get("approval_margin", 0.0)

    print("\n--- Automated Evaluation on KB-B (3-way) ---")
    for label in ["old_policy", "adapted_policy", "full_retrain_policy"]:
        r = kb_b_rows.get(label, {})
        da = r.get("drifted_accuracy")
        sa = r.get("stable_accuracy")
        da_str = f"{da:.3f}" if da is not None else "N/A"
        sa_str = f"{sa:.3f}" if sa is not None else "N/A"
        print(f"  {label:<26} Acc={r.get('accuracy', 0):.3f}  "
              f"Reward={r.get('average_reward', 0):.3f}  "
              f"Drifted={da_str}  Stable={sa_str}")

    old_b  = kb_b_rows.get("old_policy", {})
    adap_b = kb_b_rows.get("adapted_policy", {})
    ret_b  = kb_b_rows.get("full_retrain_policy", {})

    decisions = {
        "adapted_policy": approval_decision(adap_b, always_retrieve_b, approval_margin),
        "full_retrain_policy": approval_decision(ret_b, always_retrieve_b, approval_margin),
    }
    print(f"\n  Approval baseline (always_retrieve/kb_b): accuracy={always_retrieve_b['accuracy']:.3f} "
          f"retrieval_cost={always_retrieve_b['retrieval_cost']:.3f} margin={approval_margin:.3f}")
    for label, decision in decisions.items():
        row = kb_b_rows[label]
        print(f"  {label:<26} accuracy={row['accuracy']:.3f} "
              f"cost={row['retrieval_cost']:.3f} -> "
              f"{'APPROVED' if decision['approved'] else 'DECLINED'}")

    recovery = {
        label: recovery_decision(row, old_policy_a)
        for label, row in kb_b_rows.items()
    }
    print(f"  Recovery baseline (old_policy/kb_a): accuracy={old_policy_a['accuracy']:.3f}")
    for label, decision in recovery.items():
        print(f"  {label:<26} recovery="
              f"{'RECOVERED' if decision['recovered'] else 'NOT_RECOVERED'}")

    for row in all_rows:
        row["approval_baseline_accuracy"] = always_retrieve_b["accuracy"]
        row["approval_baseline_retrieval_cost"] = always_retrieve_b["retrieval_cost"]
        row["recovery_baseline_accuracy"] = old_policy_a["accuracy"]
        if row["policy"] in decisions and row["knowledge_base"] == "kb_b":
            row["approval_status"] = "APPROVED" if decisions[row["policy"]]["approved"] else "DECLINED"
            row["recovery_status"] = "RECOVERED" if recovery[row["policy"]]["recovered"] else "NOT_RECOVERED"
        else:
            row["approval_status"] = "N/A"
            row["recovery_status"] = "N/A"

    with open(summary_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(all_rows[0].keys()))
        writer.writeheader()
        writer.writerows(all_rows)
    with open(summary_path.parent / "config.json", "w", encoding="utf-8") as handle:
        json.dump(config, handle, indent=2)
    print(f"Wrote {len(all_rows)} result rows to {summary_path}")
    return all_rows


def main():
    with open(ROOT / "configs/default.yaml", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    run_experiment(config)


if __name__ == "__main__":
    main()
