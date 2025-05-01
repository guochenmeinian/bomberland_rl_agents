from config import Config

def detect_invalid_moves(state, prev_state, action_indices, unit_ids):
    if action_indices is None:
        return 0

    invalid_move_count = 0
    for unit_id, action in zip(unit_ids, action_indices):
        if 0 <= action <= 3:
            current_unit = state["unit_state"].get(unit_id)
            prev_unit = prev_state["unit_state"].get(unit_id)

            if prev_unit is None or prev_unit.get("stunned", 0) > 0 or prev_unit.get("hp", 0) <= 0:
                continue

            prev_coords = tuple(prev_unit["coordinates"])
            
            if current_unit is not None:
                curr_coords = tuple(current_unit["coordinates"])
            else:
                curr_coords = prev_coords

            if prev_coords == curr_coords:
                invalid_move_count += 1

    return invalid_move_count

def calculate_block_destruction_reward(prev_state, state, unit_ids):
    unit_ids = set(unit_ids)
    current_tick = state["tick"]
    reward = 0.0
    exploded_bombs = []
    previous_entities = prev_state.get("entities", [])
    current_entities = state.get("entities", [])

    previous_block_positions = set()
    current_block_positions = set()

    for entity in previous_entities:
        entity_type = entity.get("type")

        if (entity_type == "b" and entity.get("expires") == current_tick and entity.get("unit_id") in unit_ids):
            exploded_bombs.append({
                "x": entity["x"],
                "y": entity["y"],
                "unit_id": entity["unit_id"],
                "blast_diameter": entity.get("blast_diameter", 3)
            })
        elif entity_type in {"w", "o"}:
            previous_block_positions.add((entity["x"], entity["y"]))

    for entity in current_entities:
        if entity.get("type") in {"w", "o"}:
            current_block_positions.add((entity["x"], entity["y"]))

    
    destroyed_blocks_by_agent = set()

    for bomb in exploded_bombs:
        bomb_x = bomb["x"]
        bomb_y = bomb["y"]
        blast_diameter = bomb["blast_diameter"]
        blast_radius = (blast_diameter - 1) // 2

        directions = [(-1, 0),  (1, 0),  (0, -1),  (0, 1)]

        for direction_x, direction_y in directions:
            for distance in range(1, blast_radius + 1):
                target_x = bomb_x + direction_x * distance
                target_y = bomb_y + direction_y * distance
                target_position = (target_x, target_y)

                if target_position in previous_block_positions:
                    if target_position not in current_block_positions:
                        destroyed_blocks_by_agent.add(target_position)
                    break
    
    reward += 0.2 * len(destroyed_blocks_by_agent)
    return reward


def calculate_reward(state, prev_state, action_indices, episode, agent_id):
    if isinstance(state, dict) and "payload" in state:
        state = state["payload"]
    if isinstance(prev_state, dict) and "payload" in prev_state:
        prev_state = prev_state["payload"]

    reward = 0.0
    unit_ids = state["agents"][agent_id]["unit_ids"]
    enemy_id = "b" if agent_id == "a" else "a"
    enemy_unit_ids = state["agents"][enemy_id]["unit_ids"]

   
    if action_indices is not None:
        bomb_place_count = 0

        for a in action_indices:
            if a == 4:
                bomb_place_count += 1

        if bomb_place_count > 0:
            reward += 0.1 * bomb_place_count
            if bomb_place_count > 1:
                reward += 0.05 * (bomb_place_count - 1)

        unique_actions = len(set(action_indices))
        progress = min(episode / Config.num_episodes, 1.0)
        exploration_weight = max(0.3, 1.0 - progress)
        reward += exploration_weight * 0.1 * unique_actions

    invalid_move_count = detect_invalid_moves(state, prev_state, action_indices, unit_ids)
    reward -= 0.5 * invalid_move_count


    for unit_id in unit_ids:
        curr = state["unit_state"].get(unit_id)
        prev = prev_state["unit_state"].get(unit_id)

        if prev and not curr:
            reward -= 5.0
            continue

        if prev and curr:
            hp_diff = prev["hp"] - curr["hp"]
            if hp_diff > 0:
                reward -= 0.8 * hp_diff
            if curr["blast_diameter"] > prev["blast_diameter"]:
                reward += 0.5
            if tuple(curr["coordinates"]) != tuple(prev["coordinates"]):
                reward += 0.05

        if curr and curr.get("hp", 0) > 0:
            reward += 0.01


    for unit_id in enemy_unit_ids:
        curr = state["unit_state"].get(unit_id)
        prev = prev_state["unit_state"].get(unit_id)

        if prev and not curr:
            reward += 5.0
            continue

        if prev and curr:
            hp_diff = prev["hp"] - curr["hp"]
            if hp_diff > 0:
                reward += 0.8 * hp_diff
            if curr["stunned"] > prev["stunned"]:
                reward += 0.5

    reward += calculate_block_destruction_reward(prev_state, state, unit_ids)


    alive_units = sum(1 for uid in unit_ids if state["unit_state"].get(uid, {}).get("hp", 0) > 0)
    alive_enemies = sum(1 for uid in enemy_unit_ids if state["unit_state"].get(uid, {}).get("hp", 0) > 0)
    reward += 0.2 * (alive_units - alive_enemies)

  
    if alive_units > 0 and alive_enemies == 0:
        reward += 10.0
    elif alive_units == 0 and alive_enemies > 0:
        reward -= 10.0


    width = state["world"]["width"]
    height = state["world"]["height"]
    total_tiles = width * height
    fire_ticks_needed = (total_tiles // 2) * state["config"]["fire_spawn_interval_ticks"]
    estimated_end_tick = state["config"]["game_duration_ticks"] + fire_ticks_needed

    if state["tick"] >= estimated_end_tick:
        survival_advantage = alive_units - alive_enemies
        reward += 3.0 * survival_advantage

    return reward
