from typing import List, Dict

def calculate_reward(events: List[Dict]) -> float:
    reward = 0
    for event in events:
        if event["type"] == "unit_killed" and event["unit_id"] in ["d", "f", "h"]:
            reward += 5  # 击杀对方 unit
        if event["type"] == "unit_killed" and event["unit_id"] in ["c", "e", "g"]:
            reward -= 2  # 自己 unit 被杀
        if event["type"] == "game_won":
            reward += 10
        if event["type"] == "game_lost":
            reward -= 5
    return reward or -0.01  # 避免 reward=0 无信号
