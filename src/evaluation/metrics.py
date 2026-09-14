import numpy as np

def summarize(infos, window=30, threshold=0.9):
    if not infos: return {"accuracy": 0.0, "average_reward": 0.0, "retrieval_rate": 0.0, "retrieval_cost": 0.0}
    rewards = np.array([item["reward"] for item in infos]); correct = np.array([item["correct"] for item in infos])
    return {"accuracy": float(correct.mean()), "average_reward": float(rewards.mean()), "retrieval_rate": float(np.mean([item["action"] for item in infos])), "retrieval_cost": float(np.mean([item["retrieval_cost"] for item in infos]))}
