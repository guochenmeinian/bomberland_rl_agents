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
        pe = pe.unsqueeze(0)  # shape: (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]  # 保持 batch size 和 sequence length 对齐


class PPOModel(nn.Module):
    def __init__(self, self_state_dim=10, map_channels=8, map_size=15,
                 hidden_dim=128, action_dim=6, num_units=3):
        super().__init__()

        self.num_units = num_units
        self.hidden_dim = hidden_dim

        # CNN: (B, 8, 15, 15) -> (B, cnn_output_size)
        self.cnn = nn.Sequential(
            nn.Conv2d(map_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Flatten()
        )
        cnn_output_size = 64 * map_size * map_size

        # Linear 替代 Sequential（以便支持 3D 输入）
        self.self_fc = nn.Linear(self_state_dim, hidden_dim)

        # Unit encoder: map + self_feat → hidden
        self.unit_encoder = nn.Sequential(
            nn.Linear(cnn_output_size + hidden_dim, hidden_dim),
            nn.ReLU()
        )

        # Positional encoding for transformer
        self.pos_encoding = SinusoidalPositionalEncoding(hidden_dim)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=4, batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)

        # Value head（池化后送入）
        self.value_head = nn.Linear(hidden_dim, 1)

        # 每个 unit 一个 policy head（输出动作分布）
        self.policy_heads = nn.ModuleList([
            nn.Linear(hidden_dim, action_dim) for _ in range(num_units)
        ])

    def forward(self, self_states, full_map):
        """
        Args:
            self_states: (B, N, D)
            full_map: (B, C, H, W)
        Returns:
            logits: (B, N, A)
            value: (B,)
        """
        B, N, D = self_states.shape

        # CNN 处理 map
        map_feat = self.cnn(full_map)  # (B, cnn_output_size)
        map_feat_exp = map_feat.unsqueeze(1).expand(-1, N, -1)  # (B, N, cnn_output_size)

        # 自身状态 → (B*N, D) → fc → (B, N, hidden)
        self_states_flat = self_states.view(B * N, D)
        self_feat = self.self_fc(self_states_flat).view(B, N, self.hidden_dim)  # (B, N, H)

        # 拼接 map + self feature
        combined = torch.cat([map_feat_exp, self_feat], dim=-1)  # (B, N, cnn + hidden)
        unit_embeds = self.unit_encoder(combined)  # (B, N, hidden)

        # Transformer attention
        encoded = self.pos_encoding(unit_embeds)
        encoded = self.transformer_encoder(encoded)  # (B, N, hidden)

        # Value head：mean pool over N units
        pooled = encoded.mean(dim=1)
        value = self.value_head(pooled).squeeze(-1)  # (B,)

        # 每个 unit 输出 logits（动作分布）
        logits = torch.stack([head(encoded[:, i]) for i, head in enumerate(self.policy_heads)], dim=1)  # (B, N, A)

        return logits, value
