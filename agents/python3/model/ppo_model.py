import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np

# [输入]: self_states: (B, N, D)   # 每帧 N 个 unit 的状态
#        full_map:    (B, C, H, W) # 当前帧全局地图
#         ↓
# [Self FC]:  self_states → self feature (B, N, hidden_dim)
# [CNN]:      full_map → global feature (B, cnn_output) → broadcast N 份 → (B, N, cnn_output)
#         ↓
# [Unit Encoder]: 拼接 self feature + map feature → unit embedding (B, N, hidden_dim)
#         ↓
# [Entity-wise Attention]: 对当前帧 N 个单位做交互建模 (TransformerEncoder)
#         ↓
# [Policy Head]: 每个 unit 输出一个动作分布 → logits (B, N, action_dim)
# [Value Head]: 对所有单位求均值作为当前帧全局 value 值 → value (B,)
class PPOModel(nn.Module):
    def __init__(self, self_state_dim=10, map_channels=8, map_size=15,
                 hidden_dim=128, action_dim=6, num_units=3, max_seq_len=650):
        super().__init__()

        self.num_units = num_units
        self.hidden_dim = hidden_dim
        self.embed_dim = 16

        # CNN: (B, C, H, W) → (B, cnn_output)
        self.cnn = nn.Sequential(
            nn.Conv2d(map_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 15×15 → 7×7
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Flatten()      # (64 × 7 × 7 = 3136)
        )
        cnn_output_size = 3136

        self.self_fc = nn.Linear(self_state_dim, hidden_dim)
        self.unit_id_embed = nn.Embedding(num_units, self.embed_dim)

        combined_input_dim = cnn_output_size + hidden_dim + self.embed_dim
        self.unit_encoder = nn.Sequential(
            nn.Linear(combined_input_dim, hidden_dim),
            nn.ReLU()
        )

        self.entity_attention = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=4, batch_first=True),
            num_layers=1
        )

        self.value_head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 1)
        )

        self.policy_head = nn.Sequential(
            nn.Linear(hidden_dim, action_dim)
        )

    def forward(self, self_states, full_map):
        """
        Args:
            self_states: (B, N, D)
            full_map:    (B, C, H, W)
        Returns:
            logits: (B, N, A)
            values: (B,)
        """

        # 类型转换
        if isinstance(self_states, np.ndarray):
            self_states = torch.tensor(self_states, dtype=torch.float32)
        if isinstance(full_map, np.ndarray):
            full_map = torch.tensor(full_map, dtype=torch.float32)

        B, N, D = self_states.shape
        C, H, W = full_map.shape[1:]

        map_feat = self.cnn(full_map)  # (B, cnn_output)
        map_feat = map_feat.unsqueeze(1).expand(-1, N, -1)  # (B, N, cnn_output)

        self_feat = self.self_fc(self_states)  # (B, N, hidden_dim)

        unit_ids = torch.arange(N, device=self_states.device).unsqueeze(0).expand(B, -1)  # (B, N)
        unit_id_feature = self.unit_id_embed(unit_ids)  # (B, N, embed_dim)

        combined = torch.cat([map_feat, self_feat, unit_id_feature], dim=-1)  # (B, N, combined_dim)
        unit_embeds = self.unit_encoder(combined)  # (B, N, hidden_dim)

        attended = self.entity_attention(unit_embeds)  # (B, N, hidden_dim)

        logits = self.policy_head(attended)  # (B, N, A)

        value = self.value_head(attended).squeeze(-1)  # (B, N)
        value = value.mean(dim=-1)  # (B,)

        return logits, value
