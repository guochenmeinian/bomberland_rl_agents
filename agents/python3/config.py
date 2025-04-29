import torch
import os

class Config:
    # 环境参数
    fwd_model_uri = os.environ.get("FWD_MODEL_CONNECTION_STRING", "ws://127.0.0.1:6969/?role=admin")
    checkpoint_dir = "checkpoints"  # 目录名
    keep_last_n_checkpoint = 100  # 保留最近的15个checkpoint
    log_frequency = 10  # 每10个episode打印一次
    save_frequency = 50  # 每50个episode保存一次
    update_target_frequency = 15  # 每15个episode同步一次
    eval_frequency = 100  # 每100次训练后评估一次

    # 多环境设置
    num_envs = 5  # 🛠️ 开5个环境并行训练

    # 速度benchmark
    benchmark_batch_size = 20  # 🛠️ 每20个episode打印一次benchmark

    # 训练参数
    num_episodes = 10000
    max_steps_per_episode = 650

    # PPO参数
    self_state_dim = 10
    map_channels = 8
    map_size = 15
    action_dim = 7
    num_units = 3

    gamma = 0.99
    lam = 0.95
    clip_eps = 0.2
    lr = 3e-4

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
