# models/ppo_model.py (with attention)
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # shape (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return x

class PPOModel(nn.Module):
    def __init__(self, self_state_dim=10, map_channels=8, map_size=15, hidden_dim=128, action_dim=6, lstm_hidden_dim=128, num_units=3):
        super().__init__()

        self.num_units = num_units
        self.hidden_dim = hidden_dim
        self.lstm_hidden_dim = lstm_hidden_dim

        # CNN to process local map
        self.cnn = nn.Sequential(
            nn.Conv2d(map_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Flatten()
        ) # (B, C=8, H=15, W=15) -> (B, 64 * 15 * 15)
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

       # Positional encoding (sinusoidal)
        self.pos_encoding = SinusoidalPositionalEncoding(hidden_dim)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=4, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)

        # Value head (pooled)
        self.value_head = nn.Linear(hidden_dim, 1)

        # Policy heads (per unit)
        self.policy_heads = nn.ModuleList([
            nn.Linear(hidden_dim, action_dim) for _ in range(num_units)
        ])

    def forward(self, self_states, full_map):
        """
        Args:
            self_states: (B, num_units, self_state_dim)
            full_map:    (B, C=8, H=15, W=15)
            lstm_states: ignored
        Returns:
            logits_list: [(B, action_dim)] * num_units
            values: (B,)
            next_lstm_states: None (for compatibility)
        """
        B = self_states.size(0)
        map_feat = self.cnn(full_map)  # (B, cnn_output_size)

        unit_embeddings = []
        for i in range(self.num_units):
            self_feat = self.self_fc(self_states[:, i])  # (B, hidden_dim)
            combined_feat = torch.cat([map_feat, self_feat], dim=-1)
            encoded = self.unit_encoder(combined_feat)  # (B, hidden_dim)
            unit_embeddings.append(encoded)

        unit_tensor = torch.stack(unit_embeddings, dim=1)  # (B, num_units, hidden_dim)

        # Add sinusoidal positional encoding
        x = self.pos_encoding(unit_tensor)  # (B, num_units, hidden_dim)

        # Transformer encoder
        x = self.transformer_encoder(x)  # (B, num_units, hidden_dim)

        # Value head (mean pool)
        pooled = x.mean(dim=1)
        value = self.value_head(pooled).squeeze(-1)

        # Policy heads
        logits = [self.policy_heads[i](x[:, i]) for i in range(self.num_units)]

        return logits, value


