# train.py
import asyncio
import torch
import os
from gym import Gym
from env_utils import generate_initial_state
from entity_feature_extractor import extract_entity_features
from ppo_agent import PPOAgent
from ppo_trainer import PPOTrainer
from memory import Memory
from rewards import calculate_reward
from save_model import save_model, load_or_create_model
from plot import plot_curve

reward_history = []
ACTIONS = ["up", "down", "left", "right", "bomb", "detonate"]

def action_from_id(idx, state, unit_id):
    action = ACTIONS[idx]
    if action in ["up", "down", "left", "right"]:
        return {"type": "move", "move": action, "unit_id": unit_id}
    elif action == "bomb":
        return {"type": "bomb", "unit_id": unit_id}
    elif action == "detonate":
        bomb_coords = get_bomb_to_detonate(state, unit_id)
        if bomb_coords:
            return {"type": "detonate", "coordinates": bomb_coords, "unit_id": unit_id}
    return None


def get_bomb_to_detonate(state, unit_id):
    for e in state.get("entities", []):
        if e.get("type") == "b" and e.get("unit_id") == unit_id:
            return [e.get("x"), e.get("y")]
    return None

def format_actions(state, unit_ids, actions, agent_id):
    final = []
    for uid, act_id in zip(unit_ids, actions):
        act = action_from_id(act_id, state, uid)
        if act:
            final.append({"agent_id": agent_id, "action": act})
    return final


async def main():
    gym = Gym(os.getenv("FWD_MODEL_CONNECTION_STRING", "ws://fwd-server:6969/?role=admin"))
    await gym.connect()

    model_a = PPOAgent(entity_input_dim=8, hidden_dim=64, embed_dim=64, num_heads=4, action_dim=len(ACTIONS))
    model_b = PPOAgent(entity_input_dim=8, hidden_dim=64, embed_dim=64, num_heads=4, action_dim=len(ACTIONS))
    trainer_a = PPOTrainer(model_a)
    trainer_b = PPOTrainer(model_b)

    if os.path.exists("checkpoints/ppo.pt"):
        model_a.load_state_dict(torch.load("checkpoints/ppo.pt"))
        print("[Checkpoint] Loaded agent A")

    model_a.train()
    model_b.load_state_dict(model_a.state_dict())
    model_b.eval()

    for episode in range(10000):
        if episode % 100 == 0:
            save_model(model_a)
            model_b.load_state_dict(model_a.state_dict())
            print("[Checkpoint] Model saved and agent B updated")

        memory_a, memory_b = Memory(), Memory()
        state = generate_initial_state()
        env = gym.make(f"ep-{episode}", state)
        await env.reset()

        done = False
        while not done:
            entity_tensor_a = extract_entity_features(state, agent_id="a")
            entity_tensor_b = extract_entity_features(state, agent_id="b")

            dist_a, _ = model_a(entity_tensor_a)
            dist_b, _ = model_b(entity_tensor_b)

            actions_a = dist_a.sample()
            logprob_a = dist_a.log_prob(actions_a)
            actions_b = dist_b.sample()
            logprob_b = dist_b.log_prob(actions_b)

            unit_ids_a = state["agents"]["a"]["unit_ids"]
            unit_ids_b = state["agents"]["b"]["unit_ids"]

            game_actions = format_actions(state, unit_ids_a, actions_a.tolist(), "a") + \
                           format_actions(state, unit_ids_b, actions_b.tolist(), "b")

            next_state, done, info = await env.step(game_actions)

            reward_a = calculate_reward(info)
            reward_b = -reward_a  # 对称对抗

            memory_a.store(entity_tensor_a, actions_a, reward_a, done, logprob_a)
            memory_b.store(entity_tensor_b, actions_b, reward_b, done, logprob_b)

            state = next_state

        loss_a = trainer_a.update(memory_a)
        print(f"[EP {episode}] Agent A Loss: {loss_a:.4f}")

    await gym.close()

if __name__ == "__main__":
    asyncio.run(main())
