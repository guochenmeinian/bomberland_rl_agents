import random
from typing import Dict

def generate_initial_state(width=6, height=6) -> Dict:
    agent_a_ids = ["c", "e", "g"]
    agent_b_ids = ["d", "f", "h"]
    all_units = agent_a_ids + agent_b_ids

    state = {
        "game_id": "train",
        "agents": {
            "a": {"agent_id": "a", "unit_ids": agent_a_ids},
            "b": {"agent_id": "b", "unit_ids": agent_b_ids}
        },
        "unit_state": {},
        "entities": [],
        "world": {"width": width, "height": height},
        "tick": 0,
        "config": {
            "tick_rate_hz": 10,
            "game_duration_ticks": 300,
            "fire_spawn_interval_ticks": 2
        }
    }

    used_coords = set()
    for idx, uid in enumerate(all_units):
        while True:
            x, y = random.randint(0, width - 1), random.randint(0, height - 1)
            if (x, y) not in used_coords:
                used_coords.add((x, y))
                break
        state["unit_state"][uid] = {
            "coordinates": [x, y],
            "hp": 3,
            "inventory": {"bombs": 3},
            "blast_diameter": 3,
            "unit_id": uid,
            "agent_id": "a" if uid in agent_a_ids else "b",
            "invulnerable": 0,
            "stunned": 0
        }
    
    # add obstacles
    for _ in range(8):
        while True:
            x, y = random.randint(0, width - 1), random.randint(0, height - 1)
            if (x, y) not in used_coords:
                used_coords.add((x, y))
                state["entities"].append({"created": 0, "x": x, "y": y, "type": "m"})  # 木箱
                break

    for _ in range(8):
        while True:
            x, y = random.randint(0, width - 1), random.randint(0, height - 1)
            if (x, y) not in used_coords:
                used_coords.add((x, y))
                state["entities"].append({"created": 0, "x": x, "y": y, "type": "w", "hp": 1})  # 可破坏墙体
                break

    return state

def extract_obs(state: Dict):
    obs = []
    for unit in state["unit_state"].values():
        if unit["agent_id"] == "a":
            obs.extend(unit["coordinates"])
    return obs  # or return torch.tensor(obs, dtype=torch.float32)

def extract_obs_b(state: Dict):
    obs = []
    for unit in state["unit_state"].values():
        if unit["agent_id"] == "b":
            obs.extend(unit["coordinates"])
    return obs
