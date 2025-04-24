# ppo_agent.py
import torch
import torch.nn as nn
from torch.distributions import Categorical
from torch.nn import TransformerEncoder, TransformerEncoderLayer

class EntityEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, embed_dim):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, embed_dim)
        )

    def forward(self, x):
        # x: [batch, num_entities, input_dim]
        return self.encoder(x)  # [batch, num_entities, embed_dim]


class PPOAgent(nn.Module):
    def __init__(self, entity_input_dim, hidden_dim, embed_dim, num_heads, action_dim):
        super().__init__()
        self.entity_encoder = EntityEncoder(entity_input_dim, hidden_dim, embed_dim)
        encoder_layer = TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, batch_first=True)
        self.transformer = TransformerEncoder(encoder_layer, num_layers=1)

        self.actor = nn.Sequential(
            nn.Linear(embed_dim, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim)
        )
        self.critic = nn.Sequential(
            nn.Linear(embed_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, entity_features, entity_mask=None):
        # entity_features: [batch, num_entities, input_dim]
        x = self.entity_encoder(entity_features)  # [batch, num_entities, embed_dim]

        if entity_mask is not None:
            # entity_mask: [batch, num_entities], True for padding
            x = self.transformer(x, src_key_padding_mask=entity_mask)
        else:
            x = self.transformer(x)

        x = x.mean(dim=1)  # average pooling over entities
        logits = self.actor(x)
        value = self.critic(x)
        dist = Categorical(logits=logits)
        return dist, value

    def act(self, entity_features, entity_mask=None):
        with torch.no_grad():
            dist, _ = self.forward(entity_features, entity_mask)
            action = dist.sample()
            logprob = dist.log_prob(action)
        return action, logprob