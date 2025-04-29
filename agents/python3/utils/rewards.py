# reward.py

from config import Config 

def detect_invalid_moves(state, prev_state, action_indices, unit_ids):
    if action_indices is None:
        return 0
    
    invalid_move_count = 0

    for unit_id, action in zip(unit_ids, action_indices):

        if 0 <= action <= 3:
            current_unit = state["unit_state"].get(unit_id)
            prev_unit = prev_state["unit_state"].get(unit_id)

            if prev_unit is None or prev_unit.get("stunned") > 0 or prev_unit.get("hp", 0) <= 0:
                continue
            
            prev_coords = tuple(prev_unit["coordinates"])

            if current_unit is not None:
                curr_coords = tuple(current_unit["coordinates"])
            else:
                curr_coords = prev_coords
            
            if prev_coords == curr_coords:
                invalid_move_count += 1

    return invalid_move_count


# Calculate the reward based on the state change
def calculate_reward(state, prev_state, action_indices, episode, agent_id):

    if isinstance(state, dict) and "payload" in state:
        state = state["payload"]

    if isinstance(prev_state, dict) and "payload" in prev_state:
        prev_state = prev_state["payload"]

    reward = 0
    # print(f"the prev state: {prev_state}")
    unit_ids = state["agents"][agent_id]["unit_ids"]
    enemy_id = "b" if agent_id == "a" else "a"
    enemy_unit_ids = state["agents"][enemy_id]["unit_ids"]

    if action_indices is not None:
        bomb_place_count = 0

        for action_idx in action_indices:
            if action_idx == 4:  # 炸弹放下的动作
                bomb_place_count += 1

        if bomb_place_count > 0:
            reward += 0.1 * bomb_place_count  # 小奖励鼓励尝试放炸弹
            if bomb_place_count > 1:
                reward += 0.1 * (bomb_place_count - 1)  # 小奖励鼓励尝试放炸弹
        
        unique_actions = len(set(action_indices))
        progress = min(episode / Config.num_episodes, 1.0)  # 训练进度
        exploration_weight = max(0.3, 1 - progress)
        reward += exploration_weight * 0.1 * unique_actions
    
    invalid_move_count = detect_invalid_moves(state, prev_state, action_indices, unit_ids)
        
    reward -= 0.5 * invalid_move_count  # 惩罚无效动作

    # 检查单位状态变化
    for unit_id in unit_ids:
        # 如果单位在上一个状态中存在但在当前状态中不存在，说明单位死亡
        current_unit = state["unit_state"].get(unit_id)
        prev_unit = prev_state["unit_state"].get(unit_id)


        if prev_unit is not None and current_unit is None:
            reward -= 8.0  # 惩罚单位死亡
            continue
        
        if prev_unit is not None and current_unit is not None:
            hp_diff = prev_unit["hp"] - current_unit["hp"]
            if hp_diff > 0:
                reward -= 1.0 * hp_diff  # 惩罚单位受到伤害
            
            if  current_unit["blast_diameter"] > prev_unit["blast_diameter"]:
                reward += 0.5  # 小奖励鼓励提升爆炸范围
            
            if tuple(current_unit["coordinates"]) != tuple(prev_unit["coordinates"]):
                reward += 0.05
        
        if current_unit is not None and current_unit.get("hp", 0) > 0:
            reward += 0.05  # 小奖励鼓励单位活着tick
        

    # 检查敌方单位状态变化
    for unit_id in enemy_unit_ids:
        current_enemy = state["unit_state"].get(unit_id)
        prev_enemy = prev_state["unit_state"].get(unit_id)

        if prev_enemy is not None and current_enemy is None:
            reward += 15.0  # 奖励击杀敌方单位
            continue

        if prev_enemy is not None and current_enemy is not None:
            hp_diff = prev_enemy["hp"] - current_enemy["hp"]
            if hp_diff > 0:
                reward += 1.0 * hp_diff  # 奖励 敌方单位伤害
            
            if current_enemy["stunned"] > prev_enemy["stunned"]:
                reward += 1.0  # 奖励 敌方单位被眩晕

    # 奖励游戏进行中的存活
    alive_units = 0
    for unit_id in unit_ids:
        if unit_id in state["unit_state"]:
            unit = state["unit_state"][unit_id]
            if unit.get("hp", 0) > 0:
                alive_units += 1

    alive_enemies = 0
    for unit_id in enemy_unit_ids:
        if unit_id in state["unit_state"]:
            enemy_unit = state["unit_state"][unit_id]
            if enemy_unit.get("hp", 0) > 0:
                alive_enemies += 1

    reward += 0.2 * (alive_units - alive_enemies)  

    if alive_units > 0 and alive_enemies == 0:
        reward += 20.0
    
    width = state["world"]["width"]
    height = state["world"]["height"]
    total_tiles = width * height  # 地图总格子数

    game_duration_ticks = state["config"]["game_duration_ticks"]
    fire_spawn_interval_ticks  = state["config"]["fire_spawn_interval_ticks"]

    fire_ticks_needed = (total_tiles - 2) * fire_spawn_interval_ticks
    estimated_burn_complete_tick = game_duration_ticks + fire_ticks_needed

    # 游戏结束时的奖励
    if state["tick"] >= estimated_burn_complete_tick:
        # 计算最终存活单位差
        survival_advantage = alive_units - alive_enemies
        reward += survival_advantage * 15.0
    
    return reward