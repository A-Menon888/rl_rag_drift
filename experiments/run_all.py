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


def main():
    with open(ROOT / "configs/default.yaml", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    os.makedirs(ROOT / "results/metrics", exist_ok=True)
    os.makedirs(ROOT / "results/checkpoints", exist_ok=True)
    os.makedirs(ROOT / "results/figures", exist_ok=True)

    # --- Load both KBs and generate aligned queries together ---
    kb_a_chunks = load_knowledge_base(ROOT / config["corpus_dir"], "kb_a")
    kb_b_chunks = load_knowledge_base(ROOT / config["corpus_dir"], "kb_b")
    queries_a, queries_b = generate_document_queries(kb_a_chunks, kb_b_chunks)

    snapshots_a = {0: kb_a_chunks}
    snapshots_b = {0: kb_b_chunks}

    all_rows = []
    import torch

    # ── Arm 1: Old policy trained on KB-A ────────────────────────────────────
    old_policy = RLAgent(11, config["learning_rate"], config["seed"])
    train_a_env = make_env(queries_a, snapshots_a, config)
    training_average_reward = train_policy(old_policy, train_a_env, config["train_episodes"])
    torch.save(old_policy.policy.state_dict(), ROOT / "results/checkpoints/rl_old_policy_kb_a.pt")

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
            ROOT / f"results/figures/{policy_name}_{kb_label}_reward.png",
            "Reward", f"{policy_name} on {kb_label}",
        )

    # ── Arm 2: Adapted policy — starts from old weights, trains on KB-B ──────
    adapted_policy = RLAgent(11, config["learning_rate"], config["seed"])
    adapted_policy.policy.load_state_dict(old_policy.policy.state_dict())
    train_b_env = make_env(queries_b, snapshots_b, config)
    adapted_training_reward = train_policy(adapted_policy, train_b_env, config["train_episodes"])
    torch.save(adapted_policy.policy.state_dict(), ROOT / "results/checkpoints/rl_adapted_policy_kb_b.pt")
    eval_b_env = make_env(queries_b, snapshots_b, config)
    adapted_history = evaluate_policy(adapted_policy, eval_b_env)
    all_rows.append(result_row("adapted_policy", "kb_b", adapted_history[-1], adapted_training_reward))
    plot_series(
        [item["average_reward"] for item in adapted_history],
        ROOT / "results/figures/adapted_policy_kb_b_reward.png",
        "Reward", "adapted_policy on kb_b",
    )

    # ── Arm 3: Full retrain from scratch on KB-B ─────────────────────────────
    retrain_policy = RLAgent(11, config["learning_rate"], config["seed"])
    retrain_b_env = make_env(queries_b, snapshots_b, config)
    retrain_training_reward = train_policy(retrain_policy, retrain_b_env, config["train_episodes"])
    torch.save(retrain_policy.policy.state_dict(), ROOT / "results/checkpoints/rl_full_retrain_policy_kb_b.pt")
    eval_retrain_env = make_env(queries_b, snapshots_b, config)
    retrain_history = evaluate_policy(retrain_policy, eval_retrain_env)
    all_rows.append(result_row("full_retrain_policy", "kb_b", retrain_history[-1], retrain_training_reward))
    plot_series(
        [item["average_reward"] for item in retrain_history],
        ROOT / "results/figures/full_retrain_policy_kb_b_reward.png",
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
                ROOT / f"results/figures/{policy_name}_{kb_label}_reward.png",
                "Reward", f"{policy_name} on {kb_label}",
            )

    # ── Automated approval: 3-way KB-B comparison ────────────────────────────
    kb_b_rows = {r["policy"]: r for r in all_rows if r["knowledge_base"] == "kb_b"
                 and r["policy"] in {"old_policy", "adapted_policy", "full_retrain_policy"}}

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

    def approval_score(row, baseline):
        return ((row.get("accuracy", 0) - baseline.get("accuracy", 0)) * 10.0
                + (row.get("average_reward", 0) - baseline.get("average_reward", 0)) * 1.0
                - (row.get("retrieval_cost", 0) - baseline.get("retrieval_cost", 0)) * 1.0)

    adap_score   = approval_score(adap_b, old_b)
    retrain_score = approval_score(ret_b, old_b)
    print(f"\n  Adapted   vs old_policy: score={adap_score:.3f}  "
          f"-> {'APPROVED' if adap_score > 0 else 'DECLINED'}")
    print(f"  Retrained vs old_policy: score={retrain_score:.3f}  "
          f"-> {'APPROVED' if retrain_score > 0 else 'DECLINED'}\n")

    for row in all_rows:
        if row["policy"] == "adapted_policy" and row["knowledge_base"] == "kb_b":
            row["approval_status"] = "APPROVED" if adap_score > 0 else "DECLINED"
        elif row["policy"] == "full_retrain_policy" and row["knowledge_base"] == "kb_b":
            row["approval_status"] = "APPROVED" if retrain_score > 0 else "DECLINED"
        else:
            row["approval_status"] = "N/A"

    with open(ROOT / "results/metrics/summary.csv", "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(all_rows[0].keys()))
        writer.writeheader()
        writer.writerows(all_rows)
    with open(ROOT / "results/metrics/config.json", "w", encoding="utf-8") as handle:
        json.dump(config, handle, indent=2)
    print(f"Wrote {len(all_rows)} result rows to results/metrics/summary.csv")


if __name__ == "__main__":
    main()
