import asyncio
import torch
import os
from gym import Gym
from env_utils import generate_initial_state
from ppo_agent import PPOAgent
from memory import Memory
from rewards import calculate_reward
from save_model import save_model, load_or_create_model
from plot import plot_curve

reward_history = []

ACTIONS = ["up", "down", "left", "right", "stop", "bomb"]

def action_from_id(idx):
    action = ACTIONS[idx]
    if action in ["up", "down", "left", "right"]:
        return {"type": "move", "move": action}
    elif action == "bomb":
        return {"type": "bomb"}
    elif action == "stop":
        # ✅ "stop" = do nothing = send nothing (omit action)
        return None
    else:
        raise ValueError(f"Invalid action: {action}")

def build_unit_actions(state, model_a, model_b, entities):
    
    parsed = parse_entities(entities)

    obs_a, obs_b = [], []
    unit_ids_a, unit_ids_b = [], []

    for uid, unit in state["unit_state"].items():
        if unit["agent_id"] == "a":
            obs_a.extend(unit["coordinates"])
            unit_ids_a.append(uid)
        elif unit["agent_id"] == "b":
            obs_b.extend(unit["coordinates"])
            unit_ids_b.append(uid)

    if len(obs_a) != 6 or len(obs_b) != 6:
        raise ValueError("Agent observations must each have 3 units → 6 coordinates")

    obs_a = torch.tensor(obs_a, dtype=torch.float32).unsqueeze(0)
    obs_b = torch.tensor(obs_b, dtype=torch.float32).unsqueeze(0)

    dist_a, _ = model_a(obs_a)
    dist_b, _ = model_b(obs_b)

    actions_a = dist_a.sample().tolist()
    actions_b = dist_b.sample().tolist()

    def format_action(agent_id, unit_id, act_id):
        act = action_from_id(act_id)
        if act is None:
            return None  # stop → skip
        act["unit_id"] = unit_id
        return {
            "agent_id": agent_id,
            "action": act
        }

    final_actions = [
        format_action("a", uid, act_id) for uid, act_id in zip(unit_ids_a, actions_a)
    ] + [
        format_action("b", uid, act_id) for uid, act_id in zip(unit_ids_b, actions_b)
    ]

    # 去除 stop = None 的项
    final_actions = [a for a in final_actions if a is not None]

    return final_actions, obs_a, dist_a


def parse_entities(entities):
    metal_blocks = set()
    wooden_blocks = {}
    bombs = []
    powerups = []

    for entity in entities:
        x, y = entity["x"], entity["y"]
        ent_type = entity["type"]
        coord = (x, y)

        if ent_type == "m":
            metal_blocks.add(coord)
        elif ent_type in {"w", "o"}:
            wooden_blocks[coord] = entity["hp"]
        elif ent_type in {"a", "bp", "fp"}:
            powerups.append(entity)
        elif ent_type == "b":
            bombs.append(entity)

    return {
        "metal_blocks": metal_blocks,
        "wooden_blocks": wooden_blocks,
        "bombs": bombs,
        "powerups": powerups
    }


async def main():
    gym = Gym(os.getenv("FWD_MODEL_CONNECTION_STRING", "ws://fwd-server:6969/?role=admin"))
    await gym.connect()

    model = load_or_create_model(obs_dim=6, action_dim=6)
    old_model = PPOAgent(obs_dim=6, action_dim=6)
    old_model.load_state_dict(model.state_dict())
    old_model.eval()

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    for episode in range(10000):
        if episode % 100 == 0:
            save_model(model)
            old_model.load_state_dict(model.state_dict())
            print("[Self-Play] Updated old_model")

        memory = Memory()
        state = generate_initial_state()
        env = gym.make(f"ep-{episode}", state)
        await env.reset()

        done = False
        while not done:
            entities = state.get("entities", [])
            actions, obs_tensor, dist = build_unit_actions(state, model, old_model, entities)
            action_tensor = dist.sample()
            logprob = dist.log_prob(action_tensor)

            # ✅ send in correct format: list of {agent_id, unit_id, action}
            next_state, done, info = await env.step(actions)
            reward = calculate_reward(info)
            memory.store(obs_tensor, action_tensor, reward, done, logprob)

            if episode % 100 == 0:
                reward_history.append(sum(memory.rewards))
                plot_curve(reward_history)

            state = next_state

        # PPO update
        states = torch.cat(memory.states, dim=0)
        actions_tensor = torch.cat(memory.actions)
        old_logprobs = torch.cat(memory.logprobs)
        returns = torch.tensor(memory.rewards, dtype=torch.float32)

        dists, values = model(states)
        new_logprobs = dists.log_prob(actions_tensor)
        adv = returns - values.squeeze()

        surr1 = torch.exp(new_logprobs - old_logprobs) * adv
        surr2 = torch.clamp(torch.exp(new_logprobs - old_logprobs), 0.8, 1.2) * adv
        loss = -torch.min(surr1, surr2).mean() + 0.5 * (returns - values.squeeze()).pow(2).mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print(f"[EP {episode}] loss = {loss.item():.4f}")

    await gym.close()

if __name__ == "__main__":
    asyncio.run(main())
