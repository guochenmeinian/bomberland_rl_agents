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

        cnn_output_size = 64 * map_size * map_size

        # CNN: (B, 8, 15, 15) -> (B, cnn_output_size)
        self.cnn = nn.Sequential(
            nn.Conv2d(map_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Flatten()
        )
        
        # 自身状态 -> hidden
        self.self_fc = nn.Linear(self_state_dim, hidden_dim)

        # Unit encoder: map + self_feat → hidden
        self.unit_encoder = nn.Sequential(
            nn.Linear(cnn_output_size + hidden_dim, hidden_dim),
            nn.ReLU()
        )

        # 用于时序 transformer，支持 max_seq_len*T*N
        self.pos_encoding = SinusoidalPositionalEncoding(hidden_dim, max_len=max_seq_len * num_units)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=4, batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)

        # Value head（池化后送入）
        self.value_head = nn.Linear(hidden_dim, 1)

        # ✅ 共享 policy head
        self.policy_head = nn.Linear(hidden_dim, action_dim)  

    def forward(self, self_states, full_map):
        """
        Args:
            self_states: (B, T, N, D)
            full_map:   (B, T, C, H, W)
        Returns:
            logits: (B, T, N, A)
            values: (B, T)
        """
        B, T, N, D = self_states.shape
        C, H, W = full_map.shape[2:]

        self_states = self_states.view(B * T, N, D)
        full_map = full_map.view(B * T, C, H, W)

        # CNN (B*T, C, H, W) → (B*T, cnn_output)
        map_feat = self.cnn(full_map)
        map_feat_exp = map_feat.unsqueeze(1).expand(-1, N, -1)  # (B*T, N, cnn_output)

        # 自身状态 → (B*N, D) → fc → (B, N, hidden)
        self_states_flat = self_states.view(B * T * N, D)
        self_feat = self.self_fc(self_states_flat).view(B * T, N, self.hidden_dim)

        # 拼接 map + self feature
        combined = torch.cat([map_feat_exp, self_feat], dim=-1)
        unit_embeds = self.unit_encoder(combined)  # (B*T, N, H)

        # reshape 到 Transformer 格式：(B, T×N, H)
        unit_embeds = unit_embeds.view(B, T * N, self.hidden_dim)

        # 加上时序位置编码（注意：你可以自行添加unit位置编码）
        encoded = self.pos_encoding(unit_embeds)

        # Transformer 编码 (B, T*N, H)
        encoded = self.transformer_encoder(encoded)  # (B, T*N, H)

        # reshape 回来 (B, T, N, H)
        encoded = encoded.view(B, T, N, self.hidden_dim)

        logits = self.policy_head(encoded)  # (B, T, N, A)
        value = self.value_head(encoded).squeeze(-1)  # (B, T, N) → 取均值
        value = value.mean(dim=-1)  # (B, T)

        return logits, value
