from typing import List, Dict

def calculate_reward(events: List[Dict]) -> float:
    reward = 0
    for event in events:
        if event["type"] == "unit_killed":
            if event["unit_id"] in {"d", "f", "h"}:
                reward += 5  # 击杀对方
            elif event["unit_id"] in {"c", "e", "g"}:
                reward -= 2  # 自己死了
        elif event["type"] == "game_won":
            reward += 10
        elif event["type"] == "game_lost":
            reward -= 5
    return reward if reward != 0 else -0.01 # 避免reward为0
