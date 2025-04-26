# models/ppo_model.py (with attention)
import torch
import torch.nn as nn
import torch.nn.functional as F

class PPOModel(nn.Module):
    def __init__(self, self_state_dim=10, map_channels=8, map_size=15, hidden_dim=128, action_dim=6, num_units=3):
        super().__init__()

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

    def forward(self, self_states, full_map):
        """
        self_states: (B, num_units, self_state_dim)
        full_map:    (B, C=8, H=15, W=15)
        """
        B = self_states.size(0)
        map_feat = self.cnn(full_map)  # (B, cnn_output_size)，**不是 per-unit 的！**

        unit_embeddings = []
        for i in range(self.num_units):
            self_feat = self.self_fc(self_states[:, i])  # (B, hidden_dim)
            

            combined_feat = torch.cat([map_feat, self_feat], dim=-1)  # (B, cnn_output_size + hidden)
            
            encoded = self.unit_encoder(combined_feat)  # (B, hidden)
            unit_embeddings.append(encoded)

        unit_tensor = torch.stack(unit_embeddings, dim=1)  # (B, num_units, hidden)


        attn_out, _ = self.attn(unit_tensor, unit_tensor, unit_tensor)
        attn_out = self.post_attn_fc(attn_out)

        logits = [self.policy_heads[i](attn_out[:, i]) for i in range(self.num_units)]


        pooled = attn_out.mean(dim=1)  # (B, hidden)
        value = self.value_head(pooled).squeeze(-1)  # (B,)

        return logits, value


