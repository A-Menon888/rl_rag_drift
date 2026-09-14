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

def make_world(config, knowledge_base):
    documents = load_knowledge_base(ROOT / config["corpus_dir"], knowledge_base)
    return generate_document_queries(documents), {0: documents}, {}

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
    return {"policy": policy_name, "knowledge_base": knowledge_base, "training_average_reward": training_average_reward, **metrics}

def main():
    with open(ROOT / "configs/default.yaml", encoding="utf-8") as handle: config = yaml.safe_load(handle)
    os.makedirs(ROOT / "results/metrics", exist_ok=True); os.makedirs(ROOT / "results/checkpoints", exist_ok=True)
    all_rows = []
    worlds = {}
    for knowledge_base in config["knowledge_bases"]:
        queries, snapshots, events = make_world(config, knowledge_base)
        worlds[knowledge_base] = (queries, snapshots, events)

    old_policy = RLAgent(11, config["learning_rate"], config["seed"])
    train_a_env = RLRAGEnv(*worlds["kb_a"], seed=config["seed"], retrieval_cost=config["retrieval_cost"], top_k=config["top_k"])
    training_average_reward = train_policy(old_policy, train_a_env, config["train_episodes"])
    import torch
    torch.save(old_policy.policy.state_dict(), ROOT / "results/checkpoints/rl_old_policy_kb_a.pt")

    for policy_name, knowledge_base, policy in [
        ("old_policy", "kb_a", old_policy),
        ("old_policy", "kb_b", old_policy),
    ]:
        queries, snapshots, events = worlds[knowledge_base]
        env = RLRAGEnv(queries, snapshots, events, seed=config["seed"], retrieval_cost=config["retrieval_cost"], top_k=config["top_k"])
        history = evaluate_policy(policy, env)
        all_rows.append(result_row(policy_name, knowledge_base, history[-1], training_average_reward if knowledge_base == "kb_a" else ""))
        plot_series([item["average_reward"] for item in history], ROOT / f"results/figures/{policy_name}_{knowledge_base}_reward.png", "Reward", f"{policy_name} on {knowledge_base}")

    adapted_policy = RLAgent(11, config["learning_rate"], config["seed"])
    adapted_policy.policy.load_state_dict(old_policy.policy.state_dict())
    train_b_env = RLRAGEnv(*worlds["kb_b"], seed=config["seed"], retrieval_cost=config["retrieval_cost"], top_k=config["top_k"])
    adapted_training_reward = train_policy(adapted_policy, train_b_env, config["train_episodes"])
    torch.save(adapted_policy.policy.state_dict(), ROOT / "results/checkpoints/rl_adapted_policy_kb_b.pt")
    eval_b_env = RLRAGEnv(*worlds["kb_b"], seed=config["seed"], retrieval_cost=config["retrieval_cost"], top_k=config["top_k"])
    adapted_history = evaluate_policy(adapted_policy, eval_b_env)
    all_rows.append(result_row("adapted_policy", "kb_b", adapted_history[-1], adapted_training_reward))
    plot_series([item["average_reward"] for item in adapted_history], ROOT / "results/figures/adapted_policy_kb_b_reward.png", "Reward", "adapted_policy on kb_b")

    for policy_name, policy in {"always_direct": AlwaysDirect(), "always_retrieve": AlwaysRetrieve(), "random": RandomPolicy(config["seed"])}.items():
        for knowledge_base in config["knowledge_bases"]:
            queries, snapshots, events = worlds[knowledge_base]
            env = RLRAGEnv(queries, snapshots, events, seed=config["seed"], retrieval_cost=config["retrieval_cost"], top_k=config["top_k"])
            history = run_policy(policy, env, 3)
            all_rows.append(result_row(policy_name, knowledge_base, history[-1]))
            plot_series([item["average_reward"] for item in history], ROOT / f"results/figures/{policy_name}_{knowledge_base}_reward.png", "Reward", f"{policy_name} on {knowledge_base}")
    with open(ROOT / "results/metrics/summary.csv", "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=all_rows[0].keys()); writer.writeheader(); writer.writerows(all_rows)
    with open(ROOT / "results/metrics/config.json", "w", encoding="utf-8") as handle: json.dump(config, handle, indent=2)
    print(f"Wrote {len(all_rows)} result rows to results/metrics/summary.csv")

if __name__ == "__main__": main()
