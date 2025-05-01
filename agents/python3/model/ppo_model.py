import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np

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


# [每条序列]: (T, N, D) unit状态序列
#            ↓
# [Unit Encoder]: 变成 (T, N, H)
#            ↓
# [flatten]: reshape 成 (T, N, H) → (T, N, H) or (T*N, H)
#            ↓
# [Positional Encoding]: 为时间 + unit 添加位置编码
#            ↓
# [Transformer Block]: 可加 causal mask（可选）
#            ↓
# [Head（Policy & Value）]
class PPOModel(nn.Module):
    def __init__(self, self_state_dim=10, map_channels=8, map_size=15,
                 hidden_dim=128, action_dim=6, num_units=3, max_seq_len=650):
        super().__init__()

        self.num_units = num_units
        self.hidden_dim = hidden_dim

        # CNN: (B, 10, 15, 15) → (B, 64, 7, 7) → (B, 3136)
        self.cnn = nn.Sequential(
            nn.Conv2d(map_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 15×15 → 7×7
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Flatten()      # (64 × 7 × 7 = 3136)
        )
        cnn_output_size = 3136
        
        # 自身状态 -> hidden
        self.self_fc = nn.Linear(self_state_dim, hidden_dim)

        # Unit encoder: map + self_feat → hidden
        self.unit_encoder = nn.Sequential(
            nn.Linear(cnn_output_size + hidden_dim, hidden_dim),
            nn.ReLU()
        )

        # 用于时序 transformer，支持 max_seq_len*T*N
        self.pos_encoding = SinusoidalPositionalEncoding(hidden_dim, max_len=max_seq_len * num_units)

        # Entity-wise Attention 模块（每帧对 N 个单位做 Attention）
        self.entity_attention = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=4, batch_first=True),
            num_layers=1
        )

        # Value head（池化后送入）
        self.value_head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 1)
        )

        # ✅ 共享 policy head
        self.policy_head = nn.Sequential(
            # nn.LayerNorm(hidden_dim),
            # nn.Dropout(0.1),
            nn.Linear(hidden_dim, action_dim)
        )

    def forward(self, self_states, full_map):
        """
        Args:
            self_states: (B, T, N, D)
            full_map:   (B, T, C, H, W)
        Returns:
            logits: (B, T, N, A)
            values: (B, T)
        """

        # 🆕 类型安全处理：确保是 torch.Tensor
        if isinstance(self_states, np.ndarray):
            self_states = torch.tensor(self_states, dtype=torch.float32)
        if isinstance(full_map, np.ndarray):
            full_map = torch.tensor(full_map, dtype=torch.float32)

        B, T, N, D = self_states.shape
        C, H, W = full_map.shape[2:]

        self_states = self_states.reshape(B * T, N, D)
        full_map = full_map.reshape(B * T, C, H, W)

        # CNN 
        map_feat = self.cnn(full_map) # (B*T, C, H, W) → (B*T, cnn_output)
        map_feat_exp = map_feat.unsqueeze(1).expand(-1, N, -1)  # (B*T, N, cnn_output)

        # 自身状态 → (B*N, D) → fc → (B, N, hidden)
        self_states_flat = self_states.reshape(B * T * N, D)
        self_feat = self.self_fc(self_states_flat).reshape(B * T, N, self.hidden_dim)

        # 拼接 map + self feature
        combined = torch.cat([map_feat_exp, self_feat], dim=-1)
        unit_embeds = self.unit_encoder(combined)  # (B*T, N, H)
        unit_embeds = unit_embeds.view(B, T, N, self.hidden_dim) # (B, T, N, H)

        # 添加位置编码（对 T×N 做线性编码）
        unit_embeds = unit_embeds.view(B, T * N, self.hidden_dim) # (B*T, N, H)
        unit_embeds = self.pos_encoding(unit_embeds.view(B, T * N, self.hidden_dim)) # (B*T, N, H)
        unit_embeds = unit_embeds.view(B, T, N, self.hidden_dim) # (B, T, N, H)

        # Entity-wise Attention per timestep
        attended = []
        for t in range(T):
            x = unit_embeds[:, t, :, :]  # (B, N, H)
            x = self.entity_attention(x)  # (B, N, H)
            attended.append(x.unsqueeze(1))  # (B, 1, N, H)

        encoded = torch.cat(attended, dim=1)  # (B, T, N, H)

        logits = self.policy_head(encoded)  # (B, T, N, A)
        value = self.value_head(encoded).squeeze(-1)  # (B, T, N) → 取均值
        value = value.mean(dim=-1)  # (B, T)

        return logits, value
