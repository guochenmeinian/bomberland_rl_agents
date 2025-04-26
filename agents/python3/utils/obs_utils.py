# obs_utils.py
import torch
import numpy as np

SELF_STATE_DIM = 10  # 单位状态维度


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

# Helper to filter out dead units
def filter_alive_units(agent_id, unit_ids, unit_state):
    alive_ids = []
    for uid in unit_ids:
        unit = unit_state.get(uid)
        if unit is not None and unit.get("hp", 0) > 0:
            alive_ids.append(uid)
    return alive_ids

def get_alive_mask(unit_ids, alive_unit_ids):
    alive_set = set(alive_unit_ids)  # 先把活着的unit id做成set，查找O(1)
    alive_mask = []
    for uid in unit_ids:
        if uid in alive_set:
            alive_mask.append(1)
        else:
            alive_mask.append(0)
    return np.array(alive_mask, dtype=np.float32)


# Pad actions to maintain fixed unit action shape
def padding_actions(action_indices, unit_ids, alive_unit_ids):
    padded_actions = []
    alive_idx = 0

    for unit_id in unit_ids:
        if alive_idx < len(alive_unit_ids) and unit_id == alive_unit_ids[alive_idx]:
            padded_actions.append(action_indices[alive_idx])
            alive_idx += 1
        else:
            padded_actions.append(6)  # 6 = 'do nothing' / null action

    return padded_actions


def state_to_observations(state, agent_id="a"):
    # print(f"state to observations: {state}")
    if isinstance(state, dict) and "payload" in state:
        state = state["payload"]

    map_size = state["world"]["width"]
    # 创建地图表示
    # 通道: 0=我方单位, 1=敌方单位, 2=金属墙, 3=木墙, 4=矿石, 5=炸弹, 6=火焰, 7=道具
    full_map = np.zeros((8, map_size, map_size), dtype=np.float32)
    # 填充实体
    for entity in state["entities"]:
        x, y = entity["x"], entity["y"]
        if entity["type"] == "m":  # 金属墙
            full_map[2, y, x] = 1
        elif entity["type"] == "w":  # 木墙
            full_map[3, y, x] = 1
        elif entity["type"] == "o":  # 矿石
            full_map[4, y, x] = 1
        elif entity["type"] == "b":  # 炸弹
            full_map[5, y, x] = 1
        elif entity["type"] == "x":  # 火焰
            full_map[6, y, x] = 1
        elif entity["type"] in ["a", "bp"]:  # 道具
            full_map[7, y, x] = 1
    
    # 获取当前玩家的单位ID
    unit_ids = state["agents"][agent_id]["unit_ids"]
    enemy_id = "b" if agent_id == "a" else "a"
    enemy_unit_ids = state["agents"][enemy_id]["unit_ids"]
    
    # 填充单位位置
    self_states, alive_unit_ids = [], []
    
    for unit_id in unit_ids:
        unit = state["unit_state"].get(unit_id)
       
        if unit is not None and unit["hp"] > 0:
            alive_unit_ids.append(unit_id)
            x, y = unit["coordinates"]
            full_map[0, y, x] = 1
            
            # 构建单位状态向量
            self_state = np.zeros(SELF_STATE_DIM, dtype=np.float32)
            self_state[0:2] = [x / map_size, y / map_size]  # 标准化坐标
            self_state[2] = unit["hp"] / 3.0  # 标准化生命值
            self_state[3] = unit["inventory"]["bombs"] / 3.0  # 标准化炸弹数量
            self_state[4] = unit["blast_diameter"] / 5.0  # 标准化爆炸范围
            self_state[5] = unit["invulnerable"] / 5.0  # 标准化无敌时间
            self_state[6] = unit["stunned"] / 5.0  # 标准化眩晕时间
            self_state[7] = state["tick"] / 300.0  # 标准化游戏时间
        else:
            self_state = np.zeros(SELF_STATE_DIM, dtype=np.float32)

        self_states.append(self_state)
    # 为敌方单位填充位置
    for unit_id in enemy_unit_ids:
        unit = state["unit_state"].get(unit_id)
        if unit and unit["hp"] > 0:
            x, y = unit["coordinates"]
            full_map[1, y, x] = 1
    
    return np.array(self_states), np.expand_dims(full_map, 0), unit_ids, alive_unit_ids



# Convert the model's action index to the game action format
def action_index_to_game_action(action_indices, state, detonate_targets, agent_id="a"):
    # print(f"action index to game action: {state}")
    if isinstance(state, dict) and "payload" in state:
        state = state["payload"]

    unit_ids = state["agents"][agent_id]["unit_ids"]
    unit_state = state["unit_state"]
    # alive_unit_ids = filter_alive_units(agent_id, unit_ids, unit_state)

    game_actions = []
    
    # action mapping to convert model action index to game action information
    action_mapping = {
        0: {"type": "move", "move": "up"},
        1: {"type": "move", "move": "right"},
        2: {"type": "move", "move": "down"},
        3: {"type": "move", "move": "left"},
        4: {"type": "bomb"},
        5: {"type": "detonate"},
        6: None  # 不动时不发送动作
    }
    
    for i, unit_id in enumerate(unit_ids):
        unit = unit_state.get(unit_id)
        if unit is None or unit.get("hp", 0) <= 0:
            continue

        action_idx = action_indices[i]
        action = action_mapping[action_idx]
        if action is None:
            continue

        action = action.copy()
        action["unit_id"] = unit_id

        if action["type"] == "detonate":
            target = detonate_targets[i]
            if target is not None:
                x, y = target
                action["coordinates"] = [x, y]
            else:
                continue

        game_actions.append({"agent_id": agent_id, "action": action})

    return game_actions


def bombs_positions_and_count(state, unit_ids):
    if isinstance(state, dict) and "payload" in state:
        state = state["payload"]

    bomb_infos = []
    bomb_count = 0
    current_tick = state["tick"]

    for entity in state["entities"]:
        if entity["type"] == "b" and entity.get("unit_id") in unit_ids:
            x = entity["x"]
            y = entity["y"]
            unit_id = entity["unit_id"]
            created_tick = entity["created"]

            if current_tick - created_tick >= 5:
                bomb_infos.append((x, y, unit_id))
                bomb_count += 1
            else:
                continue

    return bomb_infos, bomb_count
