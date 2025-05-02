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
from utils.env_utils import log_error, extract_overlapping_sequences
from config import Config 
import time

async def run_training():

    now = datetime.datetime.now().strftime("%Y%m%d-%H%M")
    run_name = f"{Config.user}-ppo-lr{Config.lr}-g{Config.gamma}-c{Config.clip_eps}-{now}"

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
        print("[Error] No model checkpoint found, start training from scratch...")
        start_episode = 0
        target_agent.model.load_state_dict(agent.model.state_dict())

    episode_rewards = []

    initial_decay = 1.0
    decay_rate = 0.999
    win_count = 0
    
    batch_start_time = time.time()
   
    for episode in range(start_episode, Config.num_episodes):
    # for episode in range(start_episode, start_episode + 1): # for testing
        print(f"\Start Episode {episode+1}/{Config.num_episodes}")
        gym = SafeGym(Config.fwd_model_uri)
        
        try:
            try:
                await gym.connect()
            except Exception as e:
                msg = f"[Error] Episode {episode+1} gym.connect() Failed: {e}\n{traceback.format_exc()}"
                print(msg)
                log_error(msg)
                raise e

            try:
                current_state = await gym.reset_game()
            except Exception as e:
                msg = f"[Error] Episode {episode+1} gym.reset_game() Failed: {e}\n{traceback.format_exc()}"
                print(msg)
                log_error(msg)
                raise e

            gym.make("bomberland-env", current_state["payload"])
            # await asyncio.sleep(0.5)

            total_reward = 0
            current_decay = initial_decay
            episode_steps = []  # keep track of all the steps within one episode run
            episode_buffer = [] # keep track of all sequence steps within one episode run

            for step in range(Config.max_steps_per_episode):
                try:
                    # agent_a
                    self_states_a, full_map_a, agent_units_ids_a, agent_alive_units_ids_a  = state_to_observations(current_state, agent_id="a")
                    alive_mask_a = get_alive_mask(agent_units_ids_a, agent_alive_units_ids_a)
                    current_bomb_infos_a, current_bomb_count_a = bombs_positions_and_count(current_state, agent_units_ids_a)

                    action_indices_a, log_probs_a, value_a, detonate_targets_a, old_logits_a = agent.select_actions(
                        self_states_a, full_map_a, alive_mask_a, current_bomb_infos_a, current_bomb_count_a, agent_units_ids_a, current_state
                    )
                    action_indices_a = action_indices_a[0]
                    log_probs_a = log_probs_a[0]

                    # agent_b
                    self_states_b, full_map_b, agent_units_ids_b, agent_alive_units_ids_b = state_to_observations(current_state, agent_id="b")
                    alive_mask_b = get_alive_mask(agent_units_ids_b, agent_alive_units_ids_b)
                    current_bomb_infos_b, current_bomb_count_b = bombs_positions_and_count(current_state, agent_units_ids_b)

                    # freeze agent_b to not update the params
                    with torch.no_grad():
                        action_indices_b, _, _, detonate_targets_b, _ = target_agent.select_actions(
                                self_states_b, full_map_b, alive_mask_b, current_bomb_infos_b, current_bomb_count_b, agent_units_ids_b, current_state
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
                        msg = f"[Error] Episode {episode+1} Step {step+1} gym.step() Failed: {e}\n{traceback.format_exc()}"
                        print(msg)
                        log_error(msg)
                        raise e
                except Exception as step_error:
                    msg = f"[Error] Episode {episode+1} Step {step+1}: {step_error}\n{traceback.format_exc()}"
                    print(msg)
                    log_error(msg)
                    break

                if next_state is None:
                    print("[Warning]: step returns None, skip current episode")
                    break

                # if next_state["tick"] >= next_state["config"]["game_duration_ticks"]:
                #     done = True

                alive_units = filter_alive_units("a", next_state["agents"]["a"]["unit_ids"], next_state["unit_state"])
                alive_enemies = filter_alive_units("b", next_state["agents"]["b"]["unit_ids"], next_state["unit_state"])

                # game already ended, early stop to move to next round
                if len(alive_units) == 0 or len(alive_enemies) == 0:
                    print("[Warning]: Game already ended, moving to next round...")
                    done = True

                reward = calculate_reward(next_state, prev_state, action_indices_a, episode, agent_id="a")

                reward *= current_decay
                current_decay *= decay_rate

                total_reward += reward

                episode_steps.append((
                    self_states_a,
                    full_map_a,
                    action_indices_a,
                    log_probs_a,
                    reward,
                    value_a,
                    done,
                    old_logits_a
                ))

                current_state = next_state

                if step % Config.log_frequency == 0:
                    print(f"Step {step+1}, Reward: {reward:.2f}, Total Reward: {total_reward:.2f}")

                if done:
                    if len(filter_alive_units("a", next_state["agents"]["a"]["unit_ids"], next_state["unit_state"])) > 0:
                        win_count += 1
                    break
            
            if len(episode_steps) >= Config.sequence_length:
                overlapping_sequences = extract_overlapping_sequences(
                    episode_steps, 
                    window_size=Config.sequence_length, 
                    stride=1 # fully consecutive sequences (e.g. ([1,20], [2,21], [3,22], ...))
                )
                episode_buffer.extend(overlapping_sequences)
            else: # edge case (e.g. [220-227])
                episode_buffer.append(episode_steps)

            print("current buffer size:", len(episode_steps))
            print("current buffer size after sliding window:", len(episode_buffer))

            episode_rewards.append(total_reward)

            # ✅ eval win rate
            if (episode + 1) % Config.eval_frequency == 0:
                # print(f"\n[Eval] Start Evaluation at Episode {episode+1}")
                # await evaluate(agent, target_agent, episode)
                wandb.log({
                    "eval/eval_win_rate": win_count / Config.eval_frequency,
                }, step=episode)
                win_count = 0

            # ✅ save model
            if (episode + 1) % Config.save_frequency == 0:
                save_checkpoint(agent, episode+1, Config.keep_last_n_checkpoint)

            if (episode + 1) % Config.update_target_frequency == 0:
                target_agent.model.load_state_dict(agent.model.state_dict())
                print(f"[Sync] target_agent synced on Episode {episode+1}")

            buffer_len = len(episode_buffer)

            # buffer update based on threshold
            if buffer_len >= Config.full_threshold:
                print(f"✅ Full PPO update: buffer={buffer_len}, batch={batch_size}, epoch={epochs}")
                batch_size = Config.batch_size
                epochs = Config.epochs
            elif buffer_len >= Config.mid_threshold:
                batch_size = max(32, buffer_len // 3)
                epochs = 3
                print(f"🟡 Mid PPO update: buffer={buffer_len}, batch={batch_size}, epoch={epochs}")
            else:
                batch_size = max(8, buffer_len // 2)
                epochs = 2
                print(f"🔴 Edge PPO update: buffer={buffer_len}, batch={batch_size}, epoch={epochs}")

            agent.update_from_buffer(episode_buffer, episode, epochs=epochs, batch_size=batch_size)
            episode_buffer.clear()

        except Exception as e:
            msg = f"[Error] Episode {episode+1} Failed: {e}\n{traceback.format_exc()}"
            print(msg)
            log_error(msg)
        finally:
            try:
                await gym.close()
            except Exception as close_error:
                msg = f"[Error] Episode {episode+1} gym.close() error: {close_error}\n{traceback.format_exc()}"
                print(msg)
                log_error(msg)


        num_episode_buffer = len(episode_buffer)
        wandb.log({
            "benchmark/num_episode_buffer": num_episode_buffer,
            "benchmark/episode": episode
        }, step=episode)
        
        # ✅ log time and benchmark
        if (episode + 1) % Config.benchmark_batch_size == 0:
            batch_elapsed = time.time() - batch_start_time
            avg_time_per_ep = batch_elapsed / Config.benchmark_batch_size

            print(f"\n🚀 Completed {Config.benchmark_batch_size} episodes in {batch_elapsed:.2f} seconds (Avg {avg_time_per_ep:.2f} sec/episode)")

            avg_reward = np.mean(episode_rewards[-Config.benchmark_batch_size:])

            # 🟢 wandb log
            wandb.log({
                "benchmark/batch_elapsed_time": batch_elapsed,
                "benchmark/avg_episode_time": avg_time_per_ep,
                "benchmark/avg_reward": avg_reward,
                "benchmark/episode": episode
            }, step=episode)

            batch_start_time = time.time()

    wandb.finish()
    print("Training completed.")



if __name__ == "__main__":
    asyncio.run(run_training())
