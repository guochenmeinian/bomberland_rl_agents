# model/ppo_model.py (with attention)
import torch
import torch.nn as nn
import torch.nn.functional as F

class PPOModel(nn.Module):
    def __init__(self, self_state_dim=10, map_channels=8, map_size=5, hidden_dim=128, action_dim=6, num_units=3):
        super(PPOModel, self).__init__()
        self.num_units = num_units
        self.hidden_dim = hidden_dim

        # CNN to process local map
        self.cnn = nn.Sequential(
            nn.Conv2d(map_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Flatten()
        )
        
        cnn_output_size = 64 * map_size * map_size

        # Encode self state
        self.self_fc = nn.Sequential(
            nn.Linear(self_state_dim, hidden_dim),
            nn.ReLU()
        )

        # Combine self and map features for each unit
        self.unit_encoder = nn.Sequential(
            nn.Linear(cnn_output_size + hidden_dim, hidden_dim),
            nn.ReLU()
        )

        # Multi-head self-attention among units
        self.attn = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=4, batch_first=True)

        # Post-attention shared layer
        self.post_attn_fc = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )

        # Per-unit policy heads
        self.policy_heads = nn.ModuleList([
            nn.Linear(hidden_dim, action_dim) for _ in range(num_units)
        ])

        # Global state value head (pool all unit outputs)
        self.value_head = nn.Linear(hidden_dim, 1)

    def forward(self, self_states, local_maps):
        # self_states: (B, num_units, self_state_dim)
        # local_maps:  (B, num_units, C, 5, 5)
        B = self_states.size(0)

        unit_embeddings = []
        for i in range(self.num_units):
            map_feat = self.cnn(local_maps[:, i])              # (B, cnn_out)
            self_feat = self.self_fc(self_states[:, i])        # (B, hidden)
            unit_feat = torch.cat([map_feat, self_feat], dim=-1)
            encoded = self.unit_encoder(unit_feat)             # (B, hidden)
            unit_embeddings.append(encoded)

        unit_tensor = torch.stack(unit_embeddings, dim=1)  # (B, num_units, hidden)
        attn_out, _ = self.attn(unit_tensor, unit_tensor, unit_tensor)  # (B, num_units, hidden)
        attn_out = self.post_attn_fc(attn_out)  # (B, num_units, hidden)

        # Policy for each unit
        logits = [self.policy_heads[i](attn_out[:, i]) for i in range(self.num_units)]

        # Value: mean pool then output
        pooled = attn_out.mean(dim=1)  # (B, hidden)
        value = self.value_head(pooled).squeeze(-1)  # (B,)

        return logits, value

