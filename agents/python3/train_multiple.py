import asyncio
import datetime
import numpy as np
import torch
import os
import traceback
import wandb

from agent.ppo_agent import PPOAgent
from utils.obs_utils import *
from utils.rewards import calculate_reward
from utils.save_model import save_checkpoint, load_latest_checkpoint, find_latest_checkpoint
from config import Config
from dotenv import load_dotenv

from utils.vector_env_manager import VectorGymManager  # 🆕 用并行环境

load_dotenv()

def log_error(error_message):
    with open("error.log", "a") as f:
        f.write(error_message + "\n\n")

async def run_training():
    now = datetime.datetime.now().strftime("%Y%m%d-%H%M")
    run_name = f"ppo-lr{Config.lr}-g{Config.gamma}-c{Config.clip_eps}-{now}"

    wandb.login(key=os.getenv("WANDB_API_KEY"))
    cfg = Config()
    wandb.init(
        project="bomberland",
        name=run_name,
        config={key: getattr(cfg, key) for key in dir(cfg) if not key.startswith("__") and not callable(getattr(cfg, key))}
    )

    agent = PPOAgent(Config)
    target_agent = PPOAgent(Config)

    latest_ckpt = find_latest_checkpoint()
    if latest_ckpt:
        start_episode = load_latest_checkpoint(agent, latest_ckpt)
        target_agent.model.load_state_dict(agent.model.state_dict())
    else:
        print("[Checkpoint] 没找到已有模型，从头训练")
        start_episode = 0
        target_agent.model.load_state_dict(agent.model.state_dict())

    # 🆕 启动并行Vector环境
    gym_manager = VectorGymManager(Config.fwd_model_uri, num_envs=Config.num_envs)
    await gym_manager.connect_all()
    await gym_manager.reset_all()
    await asyncio.sleep(0.5)

    lstm_states_a = [None for _ in range(Config.num_envs)]  # 每个环境自己的agent
    lstm_states_b = [None for _ in range(Config.num_envs)]  # 每个环境的target agent
    episode_buffers = [[] for _ in range(Config.num_envs)]
    total_rewards = [0 for _ in range(Config.num_envs)]
    current_decays = [1.0 for _ in range(Config.num_envs)]
    done_envs = [False for _ in range(Config.num_envs)]

    decay_rate = 0.  # 你的初始设定

    for episode in range(start_episode, Config.num_episodes):
        print(f"\n开始 Episode {episode+1}/{Config.num_episodes}")

        try:
            all_actions = []

            for env_idx in range(Config.num_envs):
                if done_envs[env_idx]:  # 如果这个env已经done了，填空动作
                    all_actions.append([])
                    continue

                current_state = gym_manager.current_states[env_idx]

                try:
                    # agent_a
                    self_states_a, full_map_a, agent_units_ids_a, agent_alive_units_ids_a = state_to_observations(current_state, agent_id="a")
                    alive_mask_a = get_alive_mask(agent_units_ids_a, agent_alive_units_ids_a)
                    current_bomb_infos_a, current_bomb_count_a = bombs_positions_and_count(current_state, agent_units_ids_a)

                    action_indices_a, log_probs_a, value_a, detonate_targets_a, lstm_states_a[env_idx] = agent.select_actions(
                        self_states_a, full_map_a, alive_mask_a, current_bomb_infos_a, current_bomb_count_a, agent_units_ids_a, current_state, lstm_states_a[env_idx]
                    )
                    action_indices_a = action_indices_a[0]
                    log_probs_a = log_probs_a[0]

                    # agent_b
                    self_states_b, full_map_b, agent_units_ids_b, agent_alive_units_ids_b = state_to_observations(current_state, agent_id="b")
                    alive_mask_b = get_alive_mask(agent_units_ids_b, agent_alive_units_ids_b)
                    current_bomb_infos_b, current_bomb_count_b = bombs_positions_and_count(current_state, agent_units_ids_b)

                    with torch.no_grad():
                        action_indices_b, _, _, detonate_targets_b, lstm_states_b[env_idx] = target_agent.select_actions(
                            self_states_b, full_map_b, alive_mask_b, current_bomb_infos_b, current_bomb_count_b, agent_units_ids_b, current_state, lstm_states_b[env_idx]
                        )
                    action_indices_b = action_indices_b[0]

                    # 整合actions
                    actions_a = action_index_to_game_action(action_indices_a, current_state, detonate_targets_a, agent_id="a")
                    actions_b = action_index_to_game_action(action_indices_b, current_state, detonate_targets_b, agent_id="b")
                    combined_actions = actions_a + actions_b
                    all_actions.append(combined_actions)

                    # 记录上一状态
                    gym_manager.current_states[env_idx]["_prev_state"] = current_state.copy()
                    gym_manager.current_states[env_idx]["_meta"] = {
                        "self_states_a": self_states_a,
                        "full_map_a": full_map_a,
                        "action_indices_a": action_indices_a,
                        "log_probs_a": log_probs_a,
                        "value_a": value_a
                    }

                except Exception as step_error:
                    print(f"[Step 错误] Episode {episode+1} Env {env_idx}: {step_error}\n{traceback.format_exc()}")
                    done_envs[env_idx] = True
                    all_actions.append([])  # 空动作避免崩溃

            # 🧹 多环境一起执行 step
            next_states, dones, infos = await gym_manager.step_all(all_actions)

            for env_idx, (next_state, done) in enumerate(zip(next_states, dones)):
                if done_envs[env_idx]:
                    continue

                prev_state = gym_manager.current_states[env_idx]["_prev_state"]
                meta = gym_manager.current_states[env_idx]["_meta"]

                if next_state is None:
                    print(f"警告: Env {env_idx} step 返回了 None")
                    done_envs[env_idx] = True
                    continue

                # 结束检测
                alive_units = filter_alive_units("a", next_state["agents"]["a"]["unit_ids"], next_state["unit_state"])
                alive_enemies = filter_alive_units("b", next_state["agents"]["b"]["unit_ids"], next_state["unit_state"])
                if len(alive_units) == 0 or len(alive_enemies) == 0:
                    done = True

                reward = calculate_reward(next_state, prev_state, meta["action_indices_a"], agent_id="a")
                reward *= current_decays[env_idx]
                current_decays[env_idx] *= decay_rate
                total_rewards[env_idx] += reward

                episode_buffers[env_idx].append((
                    meta["self_states_a"],
                    meta["full_map_a"],
                    meta["action_indices_a"],
                    meta["log_probs_a"],
                    reward,
                    meta["value_a"],
                    done
                ))

                # 更新current_state
                gym_manager.current_states[env_idx] = next_state

                if done:
                    # 单局结束，更新agent
                    agent.update_from_buffer(episode_buffers[env_idx], episode)
                    episode_buffers[env_idx] = []
                    print(f"✅ Env {env_idx} 单局完成，总奖励: {total_rewards[env_idx]:.2f}")
                    total_rewards[env_idx] = 0
                    current_decays[env_idx] = 1.0
                    # 重置环境
                    gym_manager.current_states[env_idx] = await gym_manager.envs[env_idx].reset_game()
                    await asyncio.sleep(0.2)
                    gym_manager.envs[env_idx].make("bomberland-env", gym_manager.current_states[env_idx]["payload"])

            if (episode + 1) % Config.eval_frequency == 0:
                print(f"\n[评估] 开始 Evaluation at Episode {episode+1}")
                await evaluate(agent, target_agent)

            if (episode + 1) % Config.save_frequency == 0:
                save_checkpoint(agent, episode+1)

            if (episode + 1) % Config.update_target_frequency == 0:
                target_agent.model.load_state_dict(agent.model.state_dict())
                print(f"[Sync] target_agent 同步于 Episode {episode+1}")

        except Exception as e:
            msg = f"[总体错误] Episode {episode+1} 出错: {e}\n{traceback.format_exc()}"
            print(msg)
            log_error(msg)

    await gym_manager.close_all()
    wandb.finish()
    print("训练完成 ✅")

