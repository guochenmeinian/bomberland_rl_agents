# reward.py

# Calculate the reward based on the state change
def calculate_reward(state, prev_state, agent_id="a"):
    if isinstance(state, dict) and "payload" in state:
        state = state["payload"]

    if isinstance(prev_state, dict) and "payload" in prev_state:
        prev_state = prev_state["payload"]

    reward = 0
    # print(f"the prev state: {prev_state}")
    unit_ids = state["agents"][agent_id]["unit_ids"]
    enemy_id = "b" if agent_id == "a" else "a"
    enemy_unit_ids = state["agents"][enemy_id]["unit_ids"]
    
    # 检查单位状态变化
    for unit_id in unit_ids:
        # 如果单位在上一个状态中存在但在当前状态中不存在，说明单位死亡
        current_unit = state["unit_state"].get(unit_id)
        prev_unit = prev_state["unit_state"].get(unit_id)

        if prev_unit is not None and current_unit is None:
            reward -= 5.0  # 惩罚单位死亡
        elif prev_unit and current_unit:
            hp_diff = prev_unit["hp"] - current_unit["hp"]
            if hp_diff > 0:
                reward -= 1.0 * hp_diff  # 奖励伤害

            if current_unit["inventory"]["bombs"] < prev_unit["inventory"]["bombs"]:
                reward += 0.1  # 小奖励鼓励尝试放炸弹
            
            if  current_unit["blast_diameter"] > prev_unit["blast_diameter"]:
                reward += 0.2  # 小奖励鼓励提升爆炸范围

    # 检查敌方单位状态变化
    for unit_id in enemy_unit_ids:
        current_enemy = state["unit_state"].get(unit_id)
        prev_enemy = prev_state["unit_state"].get(unit_id)

        if prev_enemy is not None and current_enemy is None:
            reward += 10.0  # 奖励击杀敌方单位
        elif prev_enemy and current_enemy:
            hp_diff = prev_enemy["hp"] - current_enemy["hp"]
            if hp_diff > 0:
                reward += 0.6 * hp_diff  # 惩罚敌方单位伤害
            
            if current_enemy["stunned"] > prev_enemy["stunned"]:
                reward += 0.2  # 惩罚敌方单位被眩晕

    # 奖励游戏进行中的存活
    alive_units = sum(1 for unit_id in unit_ids if unit_id in state["unit_state"])
    alive_enemies = sum(1 for unit_id in enemy_unit_ids if unit_id in state["unit_state"])
    
    # 如果敌方全部死亡而我方有单位存活，给予额外奖励
    if alive_units > 0 and alive_enemies == 0:
        reward += 20.0
    
    # 游戏结束时的奖励
    if state["tick"] >= state["config"]["game_duration_ticks"] - 1:
        # 计算最终存活单位差
        survival_advantage = alive_units - alive_enemies
        reward += survival_advantage * 5.0
    
    return reward