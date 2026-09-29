import random
import numpy as np
import torch
from torch import nn


class PolicyNetwork(nn.Module):
    def __init__(self, state_size, actions=3, hidden=64):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(state_size, hidden), nn.Tanh(), nn.Linear(hidden, actions))

    def forward(self, state):
        return self.net(state)


class RLAgent:
    """Monte Carlo REINFORCE over single-question episodes.

    Returns are undiscounted rewards-to-go within one episode (search costs plus
    the terminal reward); they never mix rewards from different questions.
    Each update uses a batch of episodes, with returns normalized across the
    batch's steps as the baseline, plus an entropy bonus.
    """

    def __init__(self, state_size, learning_rate=0.001, seed=7, entropy_coef=0.01, actions=3):
        random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
        self.policy = PolicyNetwork(state_size, actions)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=learning_rate)
        self.entropy_coef = entropy_coef

    def _distribution(self, observation, action_mask):
        logits = self.policy(torch.as_tensor(observation, dtype=torch.float32))
        mask = torch.as_tensor(np.asarray(action_mask, dtype=bool))
        return torch.distributions.Categorical(logits=logits.masked_fill(~mask, float("-inf")))

    def act(self, observation, info, explore=True):
        with torch.no_grad():
            distribution = self._distribution(observation, info["action_mask"])
        return int(distribution.sample() if explore else torch.argmax(distribution.probs))

    def run_episode(self, env, query_index):
        """Sample one episode, keeping the graph for the policy-gradient update."""
        observation, info = env.reset(options={"query_index": query_index})
        log_probs, entropies, rewards = [], [], []
        terminated = False
        while not terminated:
            distribution = self._distribution(observation, info["action_mask"])
            action = distribution.sample()
            observation, reward, terminated, _, info = env.step(int(action))
            log_probs.append(distribution.log_prob(action))
            entropies.append(distribution.entropy())
            rewards.append(reward)
        return log_probs, entropies, rewards, info

    def update(self, episodes):
        log_probs, entropies, returns = [], [], []
        for episode_log_probs, episode_entropies, rewards, _ in episodes:
            log_probs += episode_log_probs
            entropies += episode_entropies
            returns += list(np.cumsum(rewards[::-1])[::-1])
        returns = torch.tensor(returns, dtype=torch.float32)
        advantages = (returns - returns.mean()) / (returns.std() + 1e-8) if len(returns) > 1 else returns
        loss = -(torch.stack(log_probs) * advantages).mean() - self.entropy_coef * torch.stack(entropies).mean()
        self.optimizer.zero_grad(); loss.backward(); self.optimizer.step()
        return float(loss.item())

    def train_batch(self, env, query_indices):
        """Run one episode per question index, update once, return final infos."""
        episodes = [self.run_episode(env, index) for index in query_indices]
        self.update(episodes)
        return [episode[3] for episode in episodes]
