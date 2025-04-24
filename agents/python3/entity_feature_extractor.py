# entity_feature_extractor.py
import torch
import numpy as np

def extract_entity_features(state, agent_id="a"):
    entity_features = []
    unit_ids = state["agents"][agent_id]["unit_ids"]
    for uid in unit_ids:
        if uid not in state["unit_state"]:
            continue
        unit = state["unit_state"][uid]
        x, y = unit["coordinates"]
        hp = unit["hp"]
        bombs = unit["inventory"]["bombs"]
        blast = unit["blast_diameter"]
        invul = unit["invulnerable"]
        stunned = unit["stunned"]
        tick = state["tick"]

        # 构建每个 unit 的 feature 向量（你也可以加入周围实体）
        f = [
            x / 15, y / 15,         # normalized coordinates
            hp / 3.0,
            bombs / 3.0,
            blast / 5.0,
            invul / 5.0,
            stunned / 5.0,
            tick / 300.0
        ]
        entity_features.append(f)

    entity_tensor = torch.tensor([entity_features], dtype=torch.float32)  # [1, num_units, feat_dim]
    return entity_tensor