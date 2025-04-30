import random
from typing import Dict

import os
import random
import datetime
from typing import Dict, Optional

def generate_initial_state(width=6, height=6, world_seed: Optional[int] = None, prng_seed: Optional[int] = None, use_forward_model=False) -> Dict:
    agent_a_ids = ["c", "e", "g"]
    agent_b_ids = ["d", "f", "h"]
    all_units = agent_a_ids + agent_b_ids

    # Apply random seeds if provided
    if world_seed is not None:
        random.seed(world_seed)

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

    if use_forward_model:
        # Add seeds if using forward model
        state["world_seed"] = world_seed if world_seed is not None else random.randint(0, 1e6)
        state["prng_seed"] = prng_seed if prng_seed is not None else random.randint(0, 1e6)
    else:
        # Manually construct the map
        used_coords = set()
        for uid in all_units:
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

        for _ in range(8):  # 木箱
            while True:
                x, y = random.randint(0, width - 1), random.randint(0, height - 1)
                if (x, y) not in used_coords:
                    used_coords.add((x, y))
                    state["entities"].append({"created": 0, "x": x, "y": y, "type": "m"})
                    break

        for _ in range(8):  # 可破坏墙体
            while True:
                x, y = random.randint(0, width - 1), random.randint(0, height - 1)
                if (x, y) not in used_coords:
                    used_coords.add((x, y))
                    state["entities"].append({"created": 0, "x": x, "y": y, "type": "w", "hp": 1})
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


def log_error(error_message):
    os.makedirs("logs", exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

    log_path = f"logs/error_{timestamp}.log"
    with open(log_path, "w") as f:
        f.write(error_message + "\n\n")

def extract_overlapping_sequences(sequence, window_size=10, stride=1):
    """
    将完整 episode 的 step 序列滑动切片为多个固定长度序列
    """
    result = []
    for i in range(0, len(sequence) - window_size + 1, stride):
        result.append(sequence[i : i + window_size])
    return result