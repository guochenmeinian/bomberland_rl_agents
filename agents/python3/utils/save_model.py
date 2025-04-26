import os
import glob
import torch

# utils/checkpoint_utils.py
import os
import glob
import torch

def save_checkpoint(agent, episode, keep_last_n=5, save_dir="models"):
    """
    保存模型到 save_dir/ppo_checkpoint_ep{episode}.pt
    并且自动只保留最近 keep_last_n 个checkpoint
    """
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"ppo_checkpoint_ep{episode}.pt")
    
    torch.save({
        "model": agent.model.state_dict(),
        "optimizer": agent.optimizer.state_dict(),
        "episode": episode
    }, save_path)
    
    print(f"[Checkpoint] Saved: {save_path}")
    
    # 清理多余的旧 checkpoint
    checkpoints = sorted(glob.glob(os.path.join(save_dir, "ppo_checkpoint_ep*.pt")), key=os.path.getmtime)
    if len(checkpoints) > keep_last_n:
        for ckpt in checkpoints[:-keep_last_n]:
            os.remove(ckpt)
            print(f"[Checkpoint] Removed old checkpoint: {ckpt}")

def load_latest_checkpoint(agent, save_dir):
    """
    从 save_dir 加载最新的 checkpoint
    如果不存在任何 checkpoint，返回 0
    """
    checkpoints = sorted(glob.glob(os.path.join(save_dir, "ppo_checkpoint_ep*.pt")), key=os.path.getmtime)
    
    if not checkpoints:
        print("[Checkpoint] No checkpoint found. Start from scratch.")
        return 0  # 没找到，返回 episode 0

    latest_ckpt = checkpoints[-1]
    checkpoint = torch.load(latest_ckpt)

    agent.model.load_state_dict(checkpoint["model"])
    agent.optimizer.load_state_dict(checkpoint["optimizer"])
    episode = checkpoint["episode"]

    print(f"[Checkpoint] Loaded {latest_ckpt} (episode {episode})")
    return episode

def find_latest_checkpoint(save_dir="models"):
    os.makedirs(save_dir, exist_ok=True)
    checkpoints = glob.glob(os.path.join(save_dir, "ppo_checkpoint_ep*.pt"))
    if not checkpoints:
        return None
    checkpoints = sorted(checkpoints, key=lambda x: int(x.split("_ep")[-1].split(".pt")[0]))
    return checkpoints[-1]

def load_checkpoint(agent, path):
    checkpoint = torch.load(path)
    agent.model.load_state_dict(checkpoint["model"])
    agent.optimizer.load_state_dict(checkpoint["optimizer"])
    print(f"[Checkpoint] 加载模型: {path} (Episode {checkpoint['episode']})")
    return checkpoint["episode"]
