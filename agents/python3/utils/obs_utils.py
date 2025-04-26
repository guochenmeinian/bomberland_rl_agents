# obs_utils.py
import torch
import numpy as np

SELF_STATE_DIM = 10  # 单位状态维度
MAP_SIZE = 5     # 局部地图

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
    self_states, local_maps, alive_unit_ids = [], [], []
    
    for unit_id in alive_unit_ids:
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
            
            # 提取局部地图 (5x5)
            local_x_start = max(0, x - MAP_SIZE // 2)
            local_x_end = min(map_size, x + MAP_SIZE // 2 + 1)
            local_y_start = max(0, y - MAP_SIZE // 2)
            local_y_end = min(map_size, y + MAP_SIZE // 2 + 1)
            
            # 初始化局部地图
            local_map = np.zeros((8, MAP_SIZE, MAP_SIZE), dtype=np.float32)
            
            # 复制地图部分到局部地图
            x_offset = max(0, MAP_SIZE // 2 - x)
            y_offset = max(0, MAP_SIZE // 2 - y)
            local_map[:, 
                      y_offset:y_offset + local_y_end - local_y_start, 
                      x_offset:x_offset + local_x_end - local_x_start] = \
                full_map[:, local_y_start:local_y_end, local_x_start:local_x_end]
            
            self_states.append(self_state)
            local_maps.append(local_map)
        else:
            self_states.append(np.zeros(SELF_STATE_DIM, dtype=np.float32))
            local_maps.append(np.zeros((8, MAP_SIZE, MAP_SIZE), dtype=np.float32))
    
    # 为敌方单位填充位置
    for unit_id in enemy_unit_ids:
        unit = state["unit_state"].get(unit_id)
        if unit and unit["hp"] > 0:
            x, y = unit["coordinates"]
            full_map[1, y, x] = 1
    
    return np.array(self_states), np.array(local_maps), unit_ids, alive_unit_ids


# Convert the model's action index to the game action format
def action_index_to_game_action(action_indices, state, agent_id="a"):
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
    
    for i, unit_id in enumerate(alive_unit_ids):
        # assert len(action_indices) == len(alive_unit_ids), "Mismatch between actions and alive units"

        action_idx = action_indices[i]
        action = action_mapping[action_idx]
        
        # if the action is None, skip this unit's action inf
        if action is None:
            continue
            
       # Copy the origional data, just in case we need it later
        action = action.copy()
        
        # update the unit_id in the action dic
        action["unit_id"] = unit_id

        if action["type"] == "detonate":
            bombs = []
            for entity in state["entities"]:
                if entity["type"] == "b" and entity["unit_id"] == unit_id:
                    current_tick = state.get("tick", 0)
                    if current_tick - entity.get("created", 0)  >= 5:
                        bombs.append(entity)
            if bombs:
                bombs.sort(key=lambda x: x.get("created", 0))
                action["coordinates"] = [bombs[0]["x"], bombs[0]["y"]]
            else:
                # If no bombs are available, skip this action
                continue

        game_actions.append({"agent_id": agent_id, "action": action})
    
    return game_actions
