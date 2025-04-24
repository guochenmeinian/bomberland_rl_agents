import torch
import os
from ppo_agent import PPOAgent

def save_model(model, path="checkpoints/ppo.pt"):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(model.state_dict(), path)
    print(f"[Checkpoint] Model saved to {path}")

def load_or_create_model(obs_dim=210, action_dim=6, path="checkpoints/ppo.pt"):
    model = PPOAgent(obs_dim, action_dim)
    if os.path.exists(path):
        model.load_state_dict(torch.load(path))
        print(f"[Checkpoint] Loaded model from {path}")
    else:
        print(f"[Checkpoint] Initialized new model.")
    return model
