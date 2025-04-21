import torch
import torch.nn as nn
from torch.distributions import Categorical

class PPOAgent(nn.Module):
    def __init__(self, obs_dim, action_dim):
        super().__init__()
        self.actor = nn.Sequential(
            nn.Linear(obs_dim, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim),
            nn.Softmax(dim=-1)
        )
        self.critic = nn.Sequential(
            nn.Linear(obs_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        value = self.critic(x)
        probs = self.actor(x)
        dist = Categorical(probs)
        return dist, value

    def act(self, obs):
        with torch.no_grad():
            dist, _ = self(obs)
            action = dist.sample()
            logprob = dist.log_prob(action)
        return action.item(), logprob