import random
import numpy as np
import torch
from torch import nn

class PolicyNetwork(nn.Module):
    def __init__(self, state_size=388, actions=2):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(state_size, 32), nn.Tanh(), nn.Linear(32, actions))
    def forward(self, state): return self.net(state)

class RLAgent:
    def __init__(self, state_size=388, learning_rate=0.01, seed=7):
        random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
        self.policy = PolicyNetwork(state_size)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=learning_rate)

    def act(self, observation, explore=True):
        state = torch.as_tensor(observation, dtype=torch.float32).unsqueeze(0)
        distribution = torch.distributions.Categorical(logits=self.policy(state))
        return int(distribution.sample().item() if explore else torch.argmax(distribution.logits).item())

    def update(self, log_probs, rewards, entropies=None, gamma=0.99, entropy_coef=0.05):
        returns, running = [], 0.0
        for reward in reversed(rewards):
            running = reward + gamma * running
            returns.append(running)
        returns = torch.tensor(list(reversed(returns)), dtype=torch.float32)
        if len(returns) > 1: returns = (returns - returns.mean()) / (returns.std() + 1e-8)
        loss = -(torch.stack(log_probs) * returns).sum()
        if entropies:
            loss = loss - entropy_coef * torch.stack(entropies).sum()
        self.optimizer.zero_grad(); loss.backward(); self.optimizer.step()
        return float(loss.item())

    def train_episode(self, env):
        observation, _ = env.reset(); log_probs=[]; rewards=[]; infos=[]; entropies=[]
        for _ in range(len(env.queries)):
            state = torch.as_tensor(observation, dtype=torch.float32).unsqueeze(0)
            distribution = torch.distributions.Categorical(logits=self.policy(state))
            action = distribution.sample()
            observation, reward, terminated, truncated, info = env.step(action.item())
            log_probs.append(distribution.log_prob(action).squeeze())
            entropies.append(distribution.entropy().squeeze())
            rewards.append(reward); infos.append(info)
            if terminated or truncated: break
        loss = self.update(log_probs, rewards, entropies=entropies)
        return infos, loss
