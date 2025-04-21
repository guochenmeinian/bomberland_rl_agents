import torch
import os
from ppo_agent import PPOAgent  # 直接引入模型类

CHECKPOINT_PATH = "checkpoints/ppo.pt"

def save_model(model):
    os.makedirs("checkpoints", exist_ok=True)
    torch.save(model.state_dict(), CHECKPOINT_PATH)
    print(f"[Checkpoint] Model saved to {CHECKPOINT_PATH}")

def load_or_create_model(obs_dim=6, action_dim=5):
    model = PPOAgent(obs_dim, action_dim)
    if os.path.exists(CHECKPOINT_PATH):
        model.load_state_dict(torch.load(CHECKPOINT_PATH))
        print(f"[Checkpoint] Loaded from {CHECKPOINT_PATH}")
    else:
        print(f"[Checkpoint] No existing model found, created new PPOAgent.")
    return model
