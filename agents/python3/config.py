# config.py
import os
import torch

class Config:
    
    # server
    fwd_model_uri = os.environ.get("FWD_MODEL_CONNECTION_STRING", "ws://127.0.0.1:6969/?role=admin")

    # environment
    user = "cg3972"                     # team member username
    model_name = "basic-ppo"            # model name
    checkpoint_dir = "checkpoints"      # save dir
    keep_last_n_checkpoint = 10         # save n most recent checkpoints
    log_frequency = 10                  # log reward per n episodes
    save_frequency = 50                # save model per n episodes
    update_target_frequency = 50        # update target_model per n episodes
    eval_frequency = 25                 # eval win rate per n episodes

    # multi_env
    num_envs = 5                        # (not used)

    # benchmark
    benchmark_batch_size = 20           # log benchmark

    # training
    sequence_length = 20                # number of data per sequence
    num_episodes = 10000                # training epochs
    max_steps_per_episode = 650         # maximum rounds for each game
    update_every = 5                    # (not used)
    batch_size = 64                     # sample that amount of sequences each time when updating model
    epochs = 4                          # sample that many times when updating model
    full_threshold = 256                # 
    mid_threshold = 128

    # PPO
    self_state_dim = 10                 # self state size (fixed)
    map_channels = 8                    # map channels (fixed)
    map_size = 15                       # map height and width (fixed)
    action_dim = 7                      # action space: [left, right, up, down, bomb, detonate] (fixed)
    num_units = 3                       # number of units per team
    kl_beta = 0.01                      
    kl_target = 0.02
    kl_update_rate = 1.5
    gamma = 0.99
    lam = 0.95                          # (not used)
    clip_eps = 0.1
    lr = 3e-4                           # learning rate
    min_kl_beta = 1e-4
    max_kl_beta = 10.0

    # others
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
