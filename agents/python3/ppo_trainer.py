# ppo_trainer.py
import torch
from torch.optim import Adam

class PPOTrainer:
    def __init__(self, model, lr=1e-3, gamma=0.99, clip_eps=0.2):
        self.model = model
        self.optimizer = Adam(model.parameters(), lr=lr)
        self.gamma = gamma
        self.clip_eps = clip_eps

    def compute_returns(self, rewards):
        returns = []
        R = 0
        for r in reversed(rewards):
            R = r + self.gamma * R
            returns.insert(0, R)
        return torch.tensor(returns, dtype=torch.float32)

    def update(self, memory):
        states = torch.cat(memory.states)
        actions = torch.cat(memory.actions)
        old_logprobs = torch.cat(memory.logprobs)
        returns = self.compute_returns(memory.rewards)

        dists, values = self.model(states)
        new_logprobs = dists.log_prob(actions)
        adv = returns - values.squeeze()

        ratio = torch.exp(new_logprobs - old_logprobs)
        surr1 = ratio * adv
        surr2 = torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps) * adv
        loss = -torch.min(surr1, surr2).mean() + 0.5 * (returns - values.squeeze()).pow(2).mean()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()
