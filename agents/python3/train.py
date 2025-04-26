# train.py
import asyncio
import wandb
import datetime
import numpy as np
import torch
import os

from agent.ppo_agent import PPOAgent
from env.safe_gym import SafeGym
from utils.obs_utils import *
from utils.rewards import calculate_reward
from utils.save_model import save_model, load_latest_checkpoint
from config import Config

async def run_training():
    now = datetime.datetime.now().strftime("%Y%m%d-%H%M")
    run_name = f"ppo-lr{Config.lr}-g{Config.gamma}-c{Config.clip_eps}-{now}"

    wandb.init(
        project="bomberland-ppo",
        name=run_name,
        config=vars(Config)
    )

    agent = PPOAgent(Config)
    target_agent = PPOAgent(Config)
    
    # 加载最新的 checkpoint
    latest_ckpt = find_latest_checkpoint()
    if latest_ckpt:
        start_episode = load_latest_checkpoint(agent, latest_ckpt)
        target_agent.model.load_state_dict(agent.model.state_dict())
    else:
        print("[Checkpoint] 没找到已有模型，从头训练")
        start_episode = 0
        target_agent.model.load_state_dict(agent.model.state_dict())

    episode_rewards = []

    for episode in range(start_episode, Config.num_episodes):
        print(f"\n开始 Episode {episode+1}/{Config.num_episodes}")
        gym = SafeGym(Config.fwd_model_uri)

        try:
            await gym.connect()
            current_state = await gym.reset_game()
            gym.make("bomberland-env", current_state["payload"])
            await asyncio.sleep(0.5)

            episode_buffer = []
            total_reward = 0

            for step in range(Config.max_steps_per_episode):
                # agent_a
                self_states_a, local_maps_a, agent_units_ids_a, agent_alive_units_ids_a = state_to_observations(current_state, agent_id="a")
                alive_mask_a = get_alive_mask(agent_units_ids_a, agent_alive_units_ids_a)
                action_indices_a, log_probs_a, value_a = agent.select_actions(self_states_a, local_maps_a, alive_mask_a)
                action_indices_a = action_indices_a[0]
                log_probs_a = log_probs_a[0]
                value_a = value_a[0]

                # agent_b
                self_states_b, local_maps_b, agent_units_ids_b, agent_alive_units_ids_b = state_to_observations(current_state, agent_id="b")
                alive_mask_b = get_alive_mask(agent_units_ids_b, agent_alive_units_ids_b)
                with torch.no_grad():
                    action_indices_b, _, _ = target_agent.select_actions(self_states_b, local_maps_b, alive_mask_b)
                action_indices_b = action_indices_b[0]

                # 合并动作
                actions_a = action_index_to_game_action(action_indices_a, current_state, agent_id="a")
                actions_b = action_index_to_game_action(action_indices_b, current_state, agent_id="b")
                combined_actions = actions_a + actions_b

                prev_state = current_state.copy()
                next_state, done, info = await gym.step(combined_actions)

                if next_state is None:
                    print("警告: step 返回了 None，跳出当前 episode")
                    break

                if next_state["tick"] >= next_state["config"]["game_duration_ticks"]:
                    done = True

                reward = calculate_reward(next_state, prev_state, agent_id="a")
                total_reward += reward

                episode_buffer.append((self_states_a, local_maps_a, action_indices_a, log_probs_a, reward, value_a, done))
                current_state = next_state

                if step % 10 == 0:
                    wandb.log({
                        "step_reward": reward,
                        "total_reward": total_reward,
                        "step": step,
                        "episode": episode
                    })
                    print(f"Step {step+1}, Reward: {reward:.2f}, Total: {total_reward:.2f}")

                if done:
                    break

            # 更新 agent_a 策略
            agent.update_from_buffer(episode_buffer)
            episode_rewards.append(total_reward)

            wandb.log({
                "episode_reward": total_reward,
                "episode": episode
            })

            # 保存 checkpoint
            if (episode + 1) % Config.save_frequency == 0:
                save_model(agent, episode+1)

            # 同步 target_agent
            if (episode + 1) % Config.update_target_frequency == 0:
                target_agent.model.load_state_dict(agent.model.state_dict())
                print(f"[Sync] target_agent 同步于 Episode {episode+1}")

        except Exception as e:
            print(f"Episode {episode+1} 错误: {e}")
        finally:
            await gym.close()

    wandb.finish()
    print("训练完成")

if __name__ == "__main__":
    asyncio.run(run_training())
