import asyncio
import datetime
import numpy as np
import torch
import os
import traceback
import time
import wandb

from agent.ppo_agent import PPOAgent
from utils.vector_env_manager import VectorGymManager
from utils.obs_utils import *
from utils.rewards import calculate_reward
from utils.save_model import save_checkpoint, load_latest_checkpoint, find_latest_checkpoint
from config import Config

def log_error(error_message):
    os.makedirs("logs", exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    log_path = f"logs/error_{timestamp}.log"
    with open(log_path, "w") as f:
        f.write(error_message + "\n\n")

async def run_training():
    now = datetime.datetime.now().strftime("%Y%m%d-%H%M")
    run_name = f"ppo-lr{Config.lr}-g{Config.gamma}-c{Config.clip_eps}-{now}"

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

    gym_manager = VectorGymManager(Config.fwd_model_uri, num_envs=Config.num_envs)
    await gym_manager.connect_all()
    await gym_manager.reset_all()
    await asyncio.sleep(0.5)

    lstm_states_a = [None for _ in range(Config.num_envs)]
    lstm_states_b = [None for _ in range(Config.num_envs)]
    episode_buffers = [[] for _ in range(Config.num_envs)]
    sequence_buffers = [[] for _ in range(Config.num_envs)]  # 🆕 临时收集sequence的小buffer
    total_rewards = [0 for _ in range(Config.num_envs)]
    current_decays = [1.0 for _ in range(Config.num_envs)]
    done_envs = [False for _ in range(Config.num_envs)]

    decay_rate = 0.999
    batch_start_time = time.time()

    episode_count = start_episode
    sequence_length = Config.sequence_length  # 🆕 每段多少步组成一个小sequence，比如20步

    while episode_count < Config.num_episodes:
        print(f"\n开始 Episode {episode_count+1}/{Config.num_episodes}")

        all_actions = []

        for env_idx in range(Config.num_envs):
            if done_envs[env_idx]:
                all_actions.append([])
                continue

            current_state = gym_manager.current_states[env_idx]

            try:
                self_states_a, full_map_a, agent_units_ids_a, agent_alive_units_ids_a = state_to_observations(current_state, agent_id="a")
                alive_mask_a = get_alive_mask(agent_units_ids_a, agent_alive_units_ids_a)
                current_bomb_infos_a, current_bomb_count_a = bombs_positions_and_count(current_state, agent_units_ids_a)

                action_indices_a, log_probs_a, value_a, detonate_targets_a, lstm_states_a[env_idx] = agent.select_actions(
                    self_states_a, full_map_a, alive_mask_a, current_bomb_infos_a, current_bomb_count_a, agent_units_ids_a, current_state, lstm_states_a[env_idx]
                )
                action_indices_a = action_indices_a[0]
                log_probs_a = log_probs_a[0]

                self_states_b, full_map_b, agent_units_ids_b, agent_alive_units_ids_b = state_to_observations(current_state, agent_id="b")
                alive_mask_b = get_alive_mask(agent_units_ids_b, agent_alive_units_ids_b)
                current_bomb_infos_b, current_bomb_count_b = bombs_positions_and_count(current_state, agent_units_ids_b)

                with torch.no_grad():
                    action_indices_b, _, _, detonate_targets_b, lstm_states_b[env_idx] = target_agent.select_actions(
                        self_states_b, full_map_b, alive_mask_b, current_bomb_infos_b, current_bomb_count_b, agent_units_ids_b, current_state, lstm_states_b[env_idx]
                    )
                action_indices_b = action_indices_b[0]

                actions_a = action_index_to_game_action(action_indices_a, current_state, detonate_targets_a, agent_id="a")
                actions_b = action_index_to_game_action(action_indices_b, current_state, detonate_targets_b, agent_id="b")
                combined_actions = actions_a + actions_b
                all_actions.append(combined_actions)

                gym_manager.current_states[env_idx]["_prev_state"] = current_state.copy()
                gym_manager.current_states[env_idx]["_meta"] = {
                    "self_states_a": self_states_a,
                    "full_map_a": full_map_a,
                    "action_indices_a": action_indices_a,
                    "log_probs_a": log_probs_a,
                    "value_a": value_a
                }

            except Exception as e:
                print(f"[Step 错误] Env {env_idx}: {e}\n{traceback.format_exc()}")
                done_envs[env_idx] = True
                all_actions.append([])

        next_states, dones, infos = await gym_manager.step_all(all_actions)

        for env_idx, (next_state, done) in enumerate(zip(next_states, dones)):
            if done_envs[env_idx]:
                continue

            prev_state = gym_manager.current_states[env_idx]["_prev_state"]
            meta = gym_manager.current_states[env_idx]["_meta"]

            if next_state is None:
                print(f"⚠️ Env {env_idx} step 返回 None")
                done_envs[env_idx] = True
                continue

            alive_units = filter_alive_units("a", next_state["agents"]["a"]["unit_ids"], next_state["unit_state"])
            alive_enemies = filter_alive_units("b", next_state["agents"]["b"]["unit_ids"], next_state["unit_state"])
            if len(alive_units) == 0 or len(alive_enemies) == 0:
                done = True

            reward = calculate_reward(next_state, prev_state, meta["action_indices_a"], episode_count, agent_id="a")
            reward *= current_decays[env_idx]
            current_decays[env_idx] *= decay_rate
            total_rewards[env_idx] += reward

            sequence_buffers[env_idx].append((
                meta["self_states_a"],
                meta["full_map_a"],
                meta["action_indices_a"],
                meta["log_probs_a"],
                reward,
                meta["value_a"],
                done
            ))

            # 🛠️ 如果积累够一段sequence_length步，就塞入episode_buffer
            if len(sequence_buffers[env_idx]) >= sequence_length:
                episode_buffers[env_idx].append(sequence_buffers[env_idx])
                sequence_buffers[env_idx] = []

            gym_manager.current_states[env_idx] = next_state

            if done:
                if sequence_buffers[env_idx]:   # 🔵 有剩余再存
                    episode_buffers[env_idx].append(sequence_buffers[env_idx])
                    sequence_buffers[env_idx] = []
                agent.update_from_buffer(episode_buffers[env_idx], episode_count)
                print(f"✅ Env {env_idx} 完成 Episode {episode_count}, 总奖励: {total_rewards[env_idx]:.2f}")
                total_rewards[env_idx] = 0
                episode_buffers[env_idx] = []
                current_decays[env_idx] = 1.0
                episode_count += 1

                if episode_count % Config.benchmark_batch_size == 0:
                    batch_elapsed = time.time() - batch_start_time
                    avg_time_per_ep = batch_elapsed / Config.benchmark_batch_size

                    print(f"\n🚀 Completed {Config.benchmark_batch_size} episodes in {batch_elapsed:.2f} seconds (Avg {avg_time_per_ep:.2f} sec/episode)")

                    # 🟢 wandb log
                    wandb.log({
                        "benchmark/batch_elapsed_time": batch_elapsed,
                        "benchmark/avg_episode_time": avg_time_per_ep,
                        "benchmark/episode": episode_count,
                    }, step=episode_count
                    )

                    batch_start_time = time.time()

                if (episode_count) % Config.save_frequency == 0:
                    save_checkpoint(agent, episode_count, Config.keep_last_n_checkpoint)

                if (episode_count) % Config.update_target_frequency == 0:
                    target_agent.model.load_state_dict(agent.model.state_dict())
                    print(f"[Sync] target_agent 同步于 Episode {episode_count}")
                
                if episode_count % Config.eval_frequency == 0:
                    print(f"\n[评估] Evaluation at Episode {episode_count}")
                    await evaluate(agent, target_agent, episode_count)


                gym_manager.current_states[env_idx] = await gym_manager.envs[env_idx].reset_game()
                await asyncio.sleep(0.2)
                await gym_manager.envs[env_idx].make("bomberland-env", gym_manager.current_states[env_idx]["payload"])

                # 🛠️ Reset LSTM
                lstm_states_a[env_idx] = None
                lstm_states_b[env_idx] = None

    await gym_manager.close_all()
    print("训练完成 ✅")

if __name__ == "__main__":
    asyncio.run(run_training())
