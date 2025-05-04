from config import Config
import wandb

def detect_invalid_moves(action_indices, unit_ids, alive_prev_units, alive_curr_units):
    if action_indices is None or not unit_ids:
        return 0

    invalid_move_count = 0
    for unit_id, action in zip(unit_ids, action_indices):
        if 0 <= action <= 3:  
         
            if unit_id not in alive_prev_units or alive_prev_units[unit_id].get("stunned", 0) > 0:
                continue
                
            prev_unit = alive_prev_units[unit_id]
            prev_coords = tuple(prev_unit["coordinates"])
            
            # 检查当前状态中单位是否存活
            if unit_id in alive_curr_units:
                curr_coords = tuple(alive_curr_units[unit_id]["coordinates"])
            else:
                curr_coords = prev_coords

            if prev_coords == curr_coords:
                invalid_move_count += 1

    return invalid_move_count


def is_in_bomb_range(unit_pos, bomb_pos, blast_radius):
    if unit_pos is None or bomb_pos is None or blast_radius < 0:
        return False
    
    return (abs(unit_pos[0] - bomb_pos[0]) <= blast_radius and unit_pos[1] == bomb_pos[1]) or \
           (abs(unit_pos[1] - bomb_pos[1]) <= blast_radius and unit_pos[0] == bomb_pos[0])

def calculate_block_destruction_reward(prev_state, state, unit_ids):
    """计算炸毁方块的奖励"""
    if not prev_state or not state or not unit_ids:
        return 0.0
        
    unit_ids_set = set(unit_ids)
    current_tick = state.get("tick", 0)
    reward = 0.0
    exploded_bombs = []
    previous_entities = prev_state.get("entities", [])
    current_entities = state.get("entities", [])

    previous_block_positions = set()
    current_block_positions = set()

    # 收集之前的方块位置和爆炸的炸弹
    for entity in previous_entities:
        entity_type = entity.get("type")

        if (entity_type == "b" and entity.get("expires") == current_tick and entity.get("unit_id") in unit_ids_set):
            exploded_bombs.append({
                "x": entity.get("x", 0),
                "y": entity.get("y", 0),
                "unit_id": entity.get("unit_id", ""),
                "blast_diameter": entity.get("blast_diameter", 3)
            })
        elif entity_type in {"w", "o"}:
            previous_block_positions.add((entity.get("x", 0), entity.get("y", 0)))

    # 收集当前的方块位置
    for entity in current_entities:
        if entity.get("type") in {"w", "o"}:
            current_block_positions.add((entity.get("x", 0), entity.get("y", 0)))

    destroyed_blocks_by_agent = set()

    # 计算被摧毁的方块
    for bomb in exploded_bombs:
        bomb_x = bomb["x"]
        bomb_y = bomb["y"]
        blast_diameter = bomb["blast_diameter"]
        blast_radius = (blast_diameter - 1) // 2

        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

        for direction_x, direction_y in directions:
            for distance in range(1, blast_radius + 1):
                target_x = bomb_x + direction_x * distance
                target_y = bomb_y + direction_y * distance
                target_position = (target_x, target_y)

                if target_position in previous_block_positions:
                    if target_position not in current_block_positions:
                        destroyed_blocks_by_agent.add(target_position)
                    break  # 遇到方块后停止继续沿该方向检查
    
    reward += 0.2 * len(destroyed_blocks_by_agent)
    return reward

def manhattan_distance(pos1, pos2):
    """计算两点之间的曼哈顿距离"""
    return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])

def calculate_danger_penalty(state, prev_state, unit_ids, alive_prev_units, alive_curr_units):
    """计算处于危险区域的惩罚，但跳过刚放置炸弹的位置"""
    if not alive_prev_units or not alive_curr_units:
        return 0.0

    penalty = 0.0
    unit_ids_set = set(unit_ids)
    
    # 获取所有炸弹
    bombs = [e for e in state.get("entities", []) if e.get("type") == "b"]
    
    # 找出本回合新放置的炸弹位置
    current_entities = state.get("entities", [])
    previous_entities = prev_state.get("entities", [])
    
    current_bombs = {(e.get("x", 0), e.get("y", 0)): e for e in current_entities if e.get("type") == "b"}
    previous_bombs = {(e.get("x", 0), e.get("y", 0)): e for e in previous_entities if e.get("type") == "b"}
    
    # 新放置的炸弹位置和对应的放置单位
    newly_placed_bomb_positions = {}
    for pos, bomb in current_bombs.items():
        if pos not in previous_bombs and bomb.get("unit_id") in unit_ids_set:
            newly_placed_bomb_positions[pos] = bomb.get("unit_id")
    
   # 检查每个存活的单位
    for unit_id in alive_curr_units:
        unit = alive_curr_units[unit_id]
        unit_pos = (unit["coordinates"][0], unit["coordinates"][1])
        
        # 只处理前一状态中也存活的单位
        if unit_id in alive_prev_units:
            prev_unit = alive_prev_units[unit_id]
            prev_unit_pos = (prev_unit["coordinates"][0], prev_unit["coordinates"][1])
            
            # 检查是否朝着危险区域移动
            moved_toward_danger = False
            
            for bomb in bombs:
                bomb_pos = (bomb.get("x", 0), bomb.get("y", 0))
                blast_diameter = bomb.get("blast_diameter", 3)
                blast_radius = (blast_diameter - 1) // 2
                
                # 跳过自己刚刚放置的炸弹，避免放炸弹就受到惩罚
                if (bomb.get("x", 0), bomb.get("y", 0)) in newly_placed_bomb_positions and \
                   newly_placed_bomb_positions[(bomb.get("x", 0), bomb.get("y", 0))] == unit_id:
                    continue
                
                prev_in_danger = is_in_bomb_range(prev_unit_pos, bomb_pos, blast_radius)
                curr_in_danger = is_in_bomb_range(unit_pos, bomb_pos, blast_radius)
                
                # 如果之前不在危险区，现在在危险区
                if not prev_in_danger and curr_in_danger:
                    moved_toward_danger = True
                    
                    # 如果是自己放的炸弹，惩罚更重
                    if bomb.get("unit_id") == unit_id:
                        penalty -= 1.5  # 严重惩罚自爆倾向
                    elif bomb.get("unit_id") in unit_ids_set:
                        penalty -= 1.0  # 惩罚冲向队友的炸弹
                    else:
                        penalty -= 1.0  # 惩罚冲向敌人的炸弹
                
                # 单纯待在危险区域也有惩罚 (但跳过刚放炸弹的情况)
                elif curr_in_danger:
                    # 如果是自己刚刚放的炸弹，不惩罚
                    prev_distance = manhattan_distance(prev_unit_pos, bomb_pos)
                    curr_distance = manhattan_distance(unit_pos, bomb_pos)
                    
                    # 如果当前位置比前一位置更远离炸弹，说明在逃离，不惩罚
                    if curr_distance > prev_distance:
                        continue
                    
                    if bomb.get("unit_id") == unit_id:
                        penalty -= 0.5  # 惩罚待在自己之前的炸弹旁边
                    elif bomb.get("unit_id") in unit_ids_set:
                        penalty -= 0.4  # 惩罚待在队友炸弹旁边
                    else:
                        penalty -= 0.3  # 惩罚待在敌人炸弹旁边
            
            # 检查是否从危险区域移动到安全区域
            if not moved_toward_danger and prev_unit_pos != unit_pos:
                # 检查是否从任何炸弹的危险区域移动到了安全区域
                was_in_danger = False
                now_in_danger = False
                
                for bomb in bombs:
                    bomb_pos = (bomb.get("x", 0), bomb.get("y", 0))
                    blast_diameter = bomb.get("blast_diameter", 3)
                    blast_radius = (blast_diameter - 1) // 2
                    
                    # 跳过刚放置的炸弹的考虑
                    if (bomb.get("x", 0), bomb.get("y", 0)) in newly_placed_bomb_positions and \
                       newly_placed_bomb_positions[(bomb.get("x", 0), bomb.get("y", 0))] == unit_id:
                        continue
                    
                    prev_in_danger = is_in_bomb_range(prev_unit_pos, bomb_pos, blast_radius)
                    curr_in_danger = is_in_bomb_range(unit_pos, bomb_pos, blast_radius)
                    
                    was_in_danger = was_in_danger or prev_in_danger
                    now_in_danger = now_in_danger or curr_in_danger
                
                # 只有当确实从危险区域移动到了安全区域才给予奖励
                if was_in_danger and not now_in_danger:
                    penalty += 1  # 奖励远离危险
    
    return penalty

def calculate_smart_bomb_reward(state, prev_state, action_indices, unit_ids, enemy_unit_ids, alive_team_units, alive_enemy_units):
    """计算智能放炸弹的奖励，主要基于当前局面"""
    if action_indices is None or not state or not prev_state or not unit_ids:
        return 0.0
        
    reward = 0.0
    unit_ids_set = set(unit_ids)
    placed_bombs = []
    
    # 找出这一回合放置的炸弹
    current_entities = state.get("entities", [])
    previous_entities = prev_state.get("entities", [])
    
    current_bombs = {(e.get("x", 0), e.get("y", 0)): e for e in current_entities if e.get("type") == "b"}
    previous_bombs = {(e.get("x", 0), e.get("y", 0)): e for e in previous_entities if e.get("type") == "b"}
    
    # 找出新放置的炸弹
    for pos, bomb in current_bombs.items():
        if pos not in previous_bombs and bomb.get("unit_id") in unit_ids_set:
            placed_bombs.append(bomb)
            reward += 0.2  # 奖励可能放置炸弹
    
    # 检查每个放置的炸弹是否能打到敌人
    for bomb in placed_bombs:
        bomb_pos = (bomb.get("x", 0), bomb.get("y", 0))
        blast_diameter = bomb.get("blast_diameter", 3)
        blast_radius = (blast_diameter - 1) // 2
        
        can_hit_enemy = False
        can_destroy_blocks = False
        
        # 检查是否能打到敌人
        for enemy_id in alive_enemy_units:
            enemy = alive_enemy_units[enemy_id]
            enemy_pos = (enemy["coordinates"][0], enemy["coordinates"][1])
            if is_in_bomb_range(enemy_pos, bomb_pos, blast_radius):
                can_hit_enemy = True
                reward += 0.5  # 奖励可能打到敌人的炸弹
                break
        
        # 检查是否能炸掉方块
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        for dir_x, dir_y in directions:
            for dist in range(1, blast_radius + 1):
                check_x = bomb_pos[0] + dir_x * dist
                check_y = bomb_pos[1] + dir_y * dist
                check_pos = (check_x, check_y)
                
                # 检查这个位置是否有可破坏方块
                for entity in current_entities:
                    if entity.get("type") in {"w", "o"} and \
                       (entity.get("x", 0), entity.get("y", 0)) == check_pos:
                        can_destroy_blocks = True
                        break
                
                if can_destroy_blocks:
                    break
            
            if can_destroy_blocks:
                break
        
        if can_destroy_blocks:
            reward += 0.3  # 奖励可能炸掉方块的炸弹
        
        # 如果炸弹既不能打到敌人也不能炸方块
        if not can_hit_enemy and not can_destroy_blocks:
            reward -= 0.2  # 轻微惩罚无效炸弹
            
    # 添加对队友炸弹的判断
    for bomb in placed_bombs:
        bomb_pos = (bomb.get("x", 0), bomb.get("y", 0))
        blast_diameter = bomb.get("blast_diameter", 3)
        blast_radius = (blast_diameter - 1) // 2
        
        # 检查是否可能伤害到队友
        for teammate_id in alive_team_units:
            if teammate_id != bomb.get("unit_id"):  # 不是放炸弹的单位
                teammate = alive_team_units[teammate_id]
                teammate_pos = (teammate["coordinates"][0], teammate["coordinates"][1])
                if is_in_bomb_range(teammate_pos, bomb_pos, blast_radius):
                    reward -= 0.8  # 惩罚可能伤害队友的炸弹
                    break
    
    return reward


def calculate_reward(state, prev_state, action_indices, episode, agent_id):
    """计算总奖励"""
    # 防御性检查
    if not state or not prev_state:
        return 0.0
        
    # 处理原始状态数据
    if isinstance(state, dict) and "payload" in state:
        state = state["payload"]
    if isinstance(prev_state, dict) and "payload" in prev_state:
        prev_state = prev_state["payload"]

    # 初始化奖励
    reward = 0.0
    
    # 获取单位ID
    unit_ids = state["agents"][agent_id]["unit_ids"]
    enemy_id = "b" if agent_id == "a" else "a"
    enemy_unit_ids = state["agents"][enemy_id]["unit_ids"]

    # 获取真正存活的单位（前一状态）
    alive_prev_units = {}
    for unit_id in unit_ids:
        prev_unit = prev_state["unit_state"].get(unit_id)
        if prev_unit and prev_unit.get("hp", 0) > 0:
            alive_prev_units[unit_id] = prev_unit
            
    # 获取真正存活的单位（当前状态）
    alive_curr_units = {}
    for unit_id in unit_ids:
        curr_unit = state["unit_state"].get(unit_id)
        if curr_unit and curr_unit.get("hp", 0) > 0:
            alive_curr_units[unit_id] = curr_unit
            
    # 获取真正存活的敌人单位（前一状态）
    alive_prev_enemies = {}
    for enemy_id in enemy_unit_ids:
        prev_enemy = prev_state["unit_state"].get(enemy_id)
        if prev_enemy and prev_enemy.get("hp", 0) > 0:
            alive_prev_enemies[enemy_id] = prev_enemy
            
    # 获取真正存活的敌人单位（当前状态）
    alive_curr_enemies = {}
    for enemy_id in enemy_unit_ids:
        curr_enemy = state["unit_state"].get(enemy_id)
        if curr_enemy and curr_enemy.get("hp", 0) > 0:
            alive_curr_enemies[enemy_id] = curr_enemy

    # 检测无效移动并惩罚
    invalid_move_count = detect_invalid_moves(action_indices, unit_ids, alive_prev_units, alive_curr_units)
    reward -= 0.5 * invalid_move_count

    # 使用智能炸弹奖励代替简单放炸弹奖励
    smart_bomb_reward = calculate_smart_bomb_reward(state, prev_state, action_indices, unit_ids, enemy_unit_ids, alive_curr_units, alive_curr_enemies)
    reward += smart_bomb_reward
    
    # 考虑行为多样性
    if action_indices is not None:
        unique_actions = len(set(action_indices))
        progress = min(episode / max(1, Config.num_episodes), 1.0)  # 防止除零
        exploration_weight = max(0.3, 1.0 - progress)
        reward += exploration_weight * 0.05 * unique_actions  # 降低了行动多样性奖励
    
    

    # 处理单位状态变化的奖励/惩罚
    for unit_id in unit_ids:

        if unit_id in alive_prev_units:
            prev_unit = alive_prev_units[unit_id]

            if unit_id not in alive_curr_units:
                reward -= 8.0
                continue
            
            curr_unit = alive_curr_units[unit_id]
            hp_diff = prev_unit.get("hp", 0) - curr_unit.get("hp", 0)
            if hp_diff > 0:
                reward -= 1.2 * hp_diff 
            if curr_unit.get("blast_diameter", 0) > prev_unit.get("blast_diameter", 0):
                reward += 0.5
            if tuple(curr_unit.get("coordinates", [0, 0]))!= tuple(prev_unit.get("coordinates", [0, 0])):
                reward += 0.05 
        
        if unit_id in alive_curr_units:
            reward += 0.02


    for enemy_id in enemy_unit_ids:
        if enemy_id in alive_prev_enemies:
            prev_enemy = alive_prev_enemies[enemy_id]
            
            if enemy_id not in alive_curr_enemies:
                reward += 8.0  
                continue
                
            curr_enemy = alive_curr_enemies[enemy_id]
            
            hp_diff = prev_enemy.get("hp", 0) - curr_enemy.get("hp", 0)
            if hp_diff > 0:
                reward += 1.5 * hp_diff  # 增加伤害敌人奖励
            if curr_enemy.get("stunned", 0) > prev_enemy.get("stunned", 0):
                reward += 0.5



    # 添加危险区域惩罚
    danger_penalty = calculate_danger_penalty(state, prev_state, unit_ids, alive_prev_units, alive_curr_units)
    reward += danger_penalty

    # 炸毁方块奖励
    block_reward = calculate_block_destruction_reward(prev_state, state, unit_ids)
    reward += block_reward



    # 存活单位奖励
    alive_units = len(alive_curr_units)
    alive_enemies = len(alive_curr_enemies)
    reward += 0.3 * (alive_units - alive_enemies)

    # 游戏结束奖励
    if alive_units > 0 and alive_enemies == 0:
        reward += 15.0  # 增加了赢得游戏的奖励
    elif alive_units == 0 and alive_enemies > 0:
        reward -= 15.0  # 增加了输掉游戏的惩罚
    elif alive_units == 0 and alive_enemies == 0:
        reward -= 10.0

    # 游戏即将结束时的奖励
    width = state["world"].get("width", 0)
    height = state["world"].get("height", 0)
    total_tiles = width * height
    # 防止配置缺失导致的错误
    fire_spawn_interval = state.get("config", {}).get("fire_spawn_interval_ticks", 1)
    game_duration = state.get("config", {}).get("game_duration_ticks", 0)
    
    if fire_spawn_interval > 0:  # 防止除零
        fire_ticks_needed = (total_tiles // 2) * fire_spawn_interval
        estimated_end_tick = game_duration + fire_ticks_needed

        if state.get("tick", 0) >= estimated_end_tick:
            survival_advantage = alive_units - alive_enemies
            reward += 5.0 * survival_advantage  # 增加了生存到最后的奖励

    # 确保奖励不会过大
    reward = max(-100.0, min(100.0, reward))
    
    return reward