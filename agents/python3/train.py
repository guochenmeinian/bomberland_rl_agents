# train.py
import asyncio
import datetime
import numpy as np
import torch
import os
import sys
import traceback
import wandb

from agent.ppo_agent import PPOAgent
from env.safe_gym import SafeGym
from utils.obs_utils import *
from utils.rewards import calculate_reward
from utils.save_model import save_checkpoint, load_latest_checkpoint, find_latest_checkpoint
from config import Config 
import time

def log_error(error_message):
    os.makedirs("logs", exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

    log_path = f"logs/error_{timestamp}.log"
    with open(log_path, "w") as f:
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

    episode_rewards = []

    initial_decay = 1.0
    decay_rate = 0.999
    
    lstm_states_a = None  # 自己的agent
    lstm_states_b = None  # target_agent（敌方智能体）

    batch_start_time = time.time()  # 🕐 benchmark：每 batch 开始计时
   
    for episode in range(start_episode, Config.num_episodes):
        print(f"\n开始 Episode {episode+1}/{Config.num_episodes}")
        gym = SafeGym(Config.fwd_model_uri)
        
        try:
            try:
                await gym.connect()
            except Exception as e:
                msg = f"[连接错误] Episode {episode+1} gym.connect() 失败: {e}\n{traceback.format_exc()}"
                print(msg)
                log_error(msg)
                raise e

            try:
                current_state = await gym.reset_game()
            except Exception as e:
                msg = f"[重置错误] Episode {episode+1} gym.reset_game() 失败: {e}\n{traceback.format_exc()}"
                print(msg)
                log_error(msg)
                raise e

            gym.make("bomberland-env", current_state["payload"])
            # await asyncio.sleep(0.5)

            episode_buffer = []
            total_reward = 0
            current_decay = initial_decay
            current_sequence = [] # for lstm timesteps

            for step in range(Config.max_steps_per_episode):
                try:
                    # agent_a
                    self_states_a, full_map_a, agent_units_ids_a, agent_alive_units_ids_a = state_to_observations(current_state, agent_id="a")
                    alive_mask_a = get_alive_mask(agent_units_ids_a, agent_alive_units_ids_a)
                    current_bomb_infos_a, current_bomb_count_a = bombs_positions_and_count(current_state, agent_units_ids_a)

                    action_indices_a, log_probs_a, value_a, detonate_targets_a, lstm_states_a = agent.select_actions(
                        self_states_a, full_map_a, alive_mask_a, current_bomb_infos_a, current_bomb_count_a, agent_units_ids_a, current_state, lstm_states_a
                    )
                    action_indices_a = action_indices_a[0]
                    log_probs_a = log_probs_a[0]

                    # agent_b
                    self_states_b, full_map_b, agent_units_ids_b, agent_alive_units_ids_b = state_to_observations(current_state, agent_id="b")
                    alive_mask_b = get_alive_mask(agent_units_ids_b, agent_alive_units_ids_b)
                    current_bomb_infos_b, current_bomb_count_b = bombs_positions_and_count(current_state, agent_units_ids_b)

                    with torch.no_grad():
                        action_indices_b, _, _, detonate_targets_b, lstm_states_b = target_agent.select_actions(
                                self_states_b, full_map_b, alive_mask_b, current_bomb_infos_b, current_bomb_count_b, agent_units_ids_b, current_state, lstm_states_b
                        )
                    action_indices_b = action_indices_b[0]

                    actions_a = action_index_to_game_action(action_indices_a, current_state, detonate_targets_a, agent_id="a")
                    actions_b = action_index_to_game_action(action_indices_b, current_state, detonate_targets_b, agent_id="b")
                    combined_actions = actions_a + actions_b

                    prev_state = current_state.copy()

                    try:
                        next_state, done, info = await gym.step(combined_actions)
                        # await asyncio.sleep(0.2)
                    except Exception as e:
                        msg = f"[执行动作错误] Episode {episode+1} Step {step+1} gym.step() 失败: {e}\n{traceback.format_exc()}"
                        print(msg)
                        log_error(msg)
                        raise e
                except Exception as step_error:
                    msg = f"[Step 错误] Episode {episode+1} Step {step+1}: {step_error}\n{traceback.format_exc()}"
                    print(msg)
                    log_error(msg)
                    break

                if next_state is None:
                    print("警告: step 返回了 None，跳出当前 episode")
                    break

                # if next_state["tick"] >= next_state["config"]["game_duration_ticks"]:
                #     done = True

                alive_units = filter_alive_units("a", next_state["agents"]["a"]["unit_ids"], next_state["unit_state"])
                alive_enemies = filter_alive_units("b", next_state["agents"]["b"]["unit_ids"], next_state["unit_state"])

                if len(alive_units) == 0 or len(alive_enemies) == 0:
                    print("警告: 有一个队伍没有活着的单位，跳出当前 episode")
                    done = True

                reward = calculate_reward(next_state, prev_state, action_indices_a, episode, agent_id="a")

                reward *= current_decay
                current_decay *= decay_rate

                total_reward += reward

                # 平时凑够 Config.sequence_length 步存一次；如果提前done，且current_sequence不为空，也存一次
                if len(current_sequence) == Config.sequence_length or (done and current_sequence):
                    episode_buffer.append(current_sequence)
                    current_sequence = []

                current_state = next_state

                if step % 10 == 0:
                    print(f"Step {step+1}, Reward: {reward:.2f}, Total: {total_reward:.2f}")

                if done:
                    break

            agent.update_from_buffer(episode_buffer, episode)
            episode_rewards.append(total_reward)

            # 🔵 每 eval_frequency 轮做一次评估
            if (episode + 1) % Config.eval_frequency == 0:
                print(f"\n[评估] 开始 Evaluation at Episode {episode+1}")
                await evaluate(agent, target_agent, episode)

            if (episode + 1) % Config.save_frequency == 0:
                save_checkpoint(agent, episode+1, Config.keep_last_n_checkpoint)

            if (episode + 1) % Config.update_target_frequency == 0:
                target_agent.model.load_state_dict(agent.model.state_dict())
                print(f"[Sync] target_agent 同步于 Episode {episode+1}")

        except Exception as e:
            msg = f"[总体错误] Episode {episode+1} 失败: {e}\n{traceback.format_exc()}"
            print(msg)
            log_error(msg)
        finally:
            try:
                await gym.close()
            except Exception as close_error:
                msg = f"[关闭错误] Episode {episode+1} gym.close() 出错: {close_error}\n{traceback.format_exc()}"
                print(msg)
                log_error(msg)
        
        # 🔵 每 batch_size 个 episode 打印一次时间
        if (episode + 1) % Config.benchmark_batch_size == 0:
            batch_elapsed = time.time() - batch_start_time
            avg_time_per_ep = batch_elapsed / Config.benchmark_batch_size

            print(f"\n🚀 Completed {Config.benchmark_batch_size} episodes in {batch_elapsed:.2f} seconds (Avg {avg_time_per_ep:.2f} sec/episode)")

            # 🟢 wandb log
            wandb.log({
                "benchmark/batch_elapsed_time": batch_elapsed,
                "benchmark/avg_episode_time": avg_time_per_ep,
                "benchmark/episode": episode
            }, step=episode)

            batch_start_time = time.time()

    wandb.finish()
    print("训练完成")



async def evaluate(agent, target_agent, episode, num_episodes=5):
    total_rewards = []
    win_count = 0

    for _ in range(num_episodes):
        gym = SafeGym(Config.fwd_model_uri)
        await gym.connect()
        current_state = await gym.reset_game()
        gym.make("bomberland-env", current_state["payload"])
        # await asyncio.sleep(0.5)

        lstm_states_a = None
        lstm_states_b = None

        total_reward = 0
        done = False
        while not done:
            self_states_a, full_map_a, agent_units_ids_a, agent_alive_units_ids_a = state_to_observations(current_state, agent_id="a")
            alive_mask_a = get_alive_mask(agent_units_ids_a, agent_alive_units_ids_a)
            current_bomb_infos_a, current_bomb_count_a = bombs_positions_and_count(current_state, agent_units_ids_a)
            

            action_indices_a, _, _, detonate_targets_a, lstm_states_a = agent.select_actions(
                self_states_a, full_map_a, alive_mask_a, current_bomb_infos_a, current_bomb_count_a, agent_units_ids_a, current_state, lstm_states_a
            )
            action_indices_a = action_indices_a[0]

            self_states_b, full_map_b, agent_units_ids_b, agent_alive_units_ids_b = state_to_observations(current_state, agent_id="b")
            alive_mask_b = get_alive_mask(agent_units_ids_b, agent_alive_units_ids_b)
            current_bomb_infos_b, current_bomb_count_b = bombs_positions_and_count(current_state, agent_units_ids_b)

            action_indices_b, _, _, detonate_targets_b, lstm_states_b = target_agent.select_actions(
                self_states_b, full_map_b, alive_mask_b, current_bomb_infos_b, current_bomb_count_b, agent_units_ids_b, current_state, lstm_states_b
            )
            action_indices_b = action_indices_b[0]

            actions_a = action_index_to_game_action(action_indices_a, current_state, detonate_targets_a, agent_id="a")
            actions_b = action_index_to_game_action(action_indices_b, current_state, detonate_targets_b, agent_id="b")
            combined_actions = actions_a + actions_b

            next_state, done, info = await gym.step(combined_actions)
            # await asyncio.sleep(0.2)

            reward = calculate_reward(next_state, current_state, action_indices_a, episode, agent_id="a")
            total_reward += reward
            current_state = next_state

            if done:
                if len(filter_alive_units("a", next_state["agents"]["a"]["unit_ids"], next_state["unit_state"])) > 0:
                    win_count += 1

        total_rewards.append(total_reward)
        await gym.close()

    avg_reward = np.mean(total_rewards)
    win_rate = win_count / num_episodes

    print(f"[评估结果] Win rate: {win_rate:.2f}, Avg Reward: {avg_reward:.2f}")

    wandb.log({
        "eval_win_rate": win_rate,
        "eval_avg_reward": avg_reward
    })


if __name__ == "__main__":
    asyncio.run(run_training())
