# train.py
import asyncio
import os
import time
import json
import numpy as np
import torch
from gym import Gym
import websockets
import random
from ppo_trainer import PPOAgent
from collections import deque
from safe_gym import SafeGym
from rewards import calculate_reward
from obs_utils import state_to_observations, action_index_to_game_action, filter_alive_units, get_alive_mask

# 环境配置
FWD_MODEL_URI = os.environ.get("FWD_MODEL_CONNECTION_STRING", "ws://127.0.0.1:6969/?role=admin")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 超参数设置
NUM_EPISODES = 4
MAX_STEPS_PER_EPISODE = 300
SAVE_FREQUENCY = 50  # 每隔多少轮保存模型
LOG_FREQUENCY = 10   # 每隔多少轮记录日志

# PPO 参数
SELF_STATE_DIM = 10  # 单位状态维度
MAP_CHANNELS = 8     # 局部地图通道数
MAP_SIZE = 5         # 局部地图大小
ACTION_DIM = 7       # 动作空间大小（上下左右，放炸弹，不动）
NUM_UNITS = 3        # 每个玩家的单位数量


async def train():
    # 初始化 PPO 智能体
    agent = PPOAgent(
        self_state_dim=SELF_STATE_DIM,
        map_channels=MAP_CHANNELS,
        map_size=MAP_SIZE,
        action_dim=ACTION_DIM,
        num_units=NUM_UNITS,
        device=DEVICE
    )
    
    # 加载检查点（如果存在）
    # 如果之前训练过了，可以直接加载检查点，继续训练
    # 这样可以避免从头开始训练，节省时间
    checkpoint_path = "models/ppo_checkpoint.pt"
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        agent.model.load_state_dict(checkpoint["model"])
        agent.optimizer.load_state_dict(checkpoint["optimizer"])
        start_episode = checkpoint["episode"]
        print(f"加载检查点: episode {start_episode}")
    else:
        start_episode = 0
        os.makedirs("models", exist_ok=True)
    
    # 记录每一局的reward，在后面可以计算平均reward
    episode_rewards = []
    
    # 训练循环
    for episode in range(start_episode, NUM_EPISODES):
        print(f"\n开始 Episode {episode+1}/{NUM_EPISODES}")
        
        # 对每个 episode 创建一个新的 SafeGym 实例
        gym = SafeGym(FWD_MODEL_URI)
        
        total_reward = 0  # 记录从当前这局游戏开始到结束的全部奖励
        episode_buffer = deque()

        try:
            # 连接环境
            await gym.connect()

            current_state = await gym.reset_game()
            # current_state = {'type': 'game_state', 'payload': {'game_id': 'caa3a00e-d9e1-4409-8a51-d235070ab8f6', 'agents': {'a': {'agent_id': 'a', 'unit_ids': ['c', 'e', 'g']}, 'b': {'agent_id': 'b', 'unit_ids': ['d', 'f', 'h']}}, 'unit_state': {'c': {'coordinates': [5, 12], 'hp': 3, 'inventory': {'bombs': 9999}, 'blast_diameter': 3, 'unit_id': 'c', 'agent_id': 'a', 'invulnerable': 0, 'stunned': 0}, 'd': {'coordinates': [9, 12], 'hp': 3, 'inventory': {'bombs': 9999}, 'blast_diameter': 3, 'unit_id': 'd', 'agent_id': 'b', 'invulnerable': 0, 'stunned': 0}, 'e': {'coordinates': [8, 7], 'hp': 3, 'inventory': {'bombs': 9999}, 'blast_diameter': 3, 'unit_id': 'e', 'agent_id': 'a', 'invulnerable': 0, 'stunned': 0}, 'f': {'coordinates': [6, 7], 'hp': 3, 'inventory': {'bombs': 9999}, 'blast_diameter': 3, 'unit_id': 'f', 'agent_id': 'b', 'invulnerable': 0, 'stunned': 0}, 'g': {'coordinates': [14, 12], 'hp': 3, 'inventory': {'bombs': 9999}, 'blast_diameter': 3, 'unit_id': 'g', 'agent_id': 'a', 'invulnerable': 0, 'stunned': 0}, 'h': {'coordinates': [0, 12], 'hp': 3, 'inventory': {'bombs': 9999}, 'blast_diameter': 3, 'unit_id': 'h', 'agent_id': 'b', 'invulnerable': 0, 'stunned': 0}}, 'entities': [{'created': 0, 'x': 13, 'y': 4, 'type': 'm'}, {'created': 0, 'x': 1, 'y': 4, 'type': 'm'}, {'created': 0, 'x': 3, 'y': 6, 'type': 'm'}, {'created': 0, 'x': 11, 'y': 6, 'type': 'm'}, {'created': 0, 'x': 5, 'y': 4, 'type': 'm'}, {'created': 0, 'x': 9, 'y': 4, 'type': 'm'}, {'created': 0, 'x': 8, 'y': 13, 'type': 'm'}, {'created': 0, 'x': 6, 'y': 13, 'type': 'm'}, {'created': 0, 'x': 10, 'y': 11, 'type': 'm'}, {'created': 0, 'x': 4, 'y': 11, 'type': 'm'}, {'created': 0, 'x': 0, 'y': 7, 'type': 'm'}, {'created': 0, 'x': 14, 'y': 7, 'type': 'm'}, {'created': 0, 'x': 0, 'y': 2, 'type': 'm'}, {'created': 0, 'x': 14, 'y': 2, 'type': 'm'}, {'created': 0, 'x': 14, 'y': 10, 'type': 'm'}, {'created': 0, 'x': 0, 'y': 10, 'type': 'm'}, {'created': 0, 'x': 2, 'y': 2, 'type': 'm'}, {'created': 0, 'x': 12, 'y': 2, 'type': 'm'}, {'created': 0, 'x': 12, 'y': 12, 'type': 'm'}, {'created': 0, 'x': 2, 'y': 12, 'type': 'm'}, {'created': 0, 'x': 11, 'y': 9, 'type': 'm'}, {'created': 0, 'x': 3, 'y': 9, 'type': 'm'}, {'created': 0, 'x': 8, 'y': 5, 'type': 'm'}, {'created': 0, 'x': 6, 'y': 5, 'type': 'm'}, {'created': 0, 'x': 2, 'y': 9, 'type': 'm'}, {'created': 0, 'x': 12, 'y': 9, 'type': 'm'}, {'created': 0, 'x': 11, 'y': 11, 'type': 'm'}, {'created': 0, 'x': 3, 'y': 11, 'type': 'm'}, {'created': 0, 'x': 4, 'y': 2, 'type': 'w', 'hp': 1}, {'created': 0, 'x': 10, 'y': 2, 'type': 'w', 'hp': 1}, {'created': 0, 'x': 4, 'y': 10, 'type': 'w', 'hp': 1}, {'created': 0, 'x': 10, 'y': 10, 'type': 'w', 'hp': 1}, {'created': 0, 'x': 0, 'y': 9, 'type': 'w', 'hp': 1}, {'created': 0, 'x': 14, 'y': 9, 'type': 'w', 'hp': 1}, {'created': 0, 'x': 10, 'y': 4, 'type': 'w', 'hp': 1}, {'created': 0, 'x': 4, 'y': 4, 'type': 'w', 'hp': 1}, {'created': 0, 'x': 5, 'y': 10, 'type': 'w', 'hp': 1}, {'created': 0, 'x': 9, 'y': 10, 'type': 'w', 'hp': 1}, {'created': 0, 'x': 3, 'y': 10, 'type': 'w', 'hp': 1}, {'created': 0, 'x': 11, 'y': 10, 'type': 'w', 'hp': 1}, {'created': 0, 'x': 8, 'y': 6, 'type': 'w', 'hp': 1}, {'created': 0, 'x': 6, 'y': 6, 'type': 'w', 'hp': 1}, {'created': 0, 'x': 5, 'y': 2, 'type': 'w', 'hp': 1}, {'created': 0, 'x': 9, 'y': 2, 'type': 'w', 'hp': 1}, {'created': 0, 'x': 9, 'y': 0, 'type': 'w', 'hp': 1}, {'created': 0, 'x': 5, 'y': 0, 'type': 'w', 'hp': 1}, {'created': 0, 'x': 9, 'y': 8, 'type': 'w', 'hp': 1}, {'created': 0, 'x': 5, 'y': 8, 'type': 'w', 'hp': 1}, {'created': 0, 'x': 3, 'y': 2, 'type': 'w', 'hp': 1}, {'created': 0, 'x': 11, 'y': 2, 'type': 'w', 'hp': 1}, {'created': 0, 'x': 8, 'y': 4, 'type': 'w', 'hp': 1}, {'created': 0, 'x': 6, 'y': 4, 'type': 'w', 'hp': 1}, {'created': 0, 'x': 4, 'y': 6, 'type': 'w', 'hp': 1}, {'created': 0, 'x': 10, 'y': 6, 'type': 'w', 'hp': 1}, {'created': 0, 'x': 6, 'y': 14, 'type': 'w', 'hp': 1}, {'created': 0, 'x': 8, 'y': 14, 'type': 'w', 'hp': 1}, {'created': 0, 'x': 13, 'y': 8, 'type': 'w', 'hp': 1}, {'created': 0, 'x': 1, 'y': 8, 'type': 'w', 'hp': 1}, {'created': 0, 'x': 10, 'y': 14, 'type': 'w', 'hp': 1}, {'created': 0, 'x': 4, 'y': 14, 'type': 'w', 'hp': 1}, {'created': 0, 'x': 1, 'y': 9, 'type': 'w', 'hp': 1}, {'created': 0, 'x': 13, 'y': 9, 'type': 'w', 'hp': 1}, {'created': 0, 'x': 13, 'y': 5, 'type': 'w', 'hp': 1}, {'created': 0, 'x': 1, 'y': 5, 'type': 'w', 'hp': 1}, {'created': 0, 'x': 5, 'y': 6, 'type': 'w', 'hp': 1}, {'created': 0, 'x': 9, 'y': 6, 'type': 'w', 'hp': 1}, {'created': 0, 'x': 11, 'y': 12, 'type': 'w', 'hp': 1}, {'created': 0, 'x': 3, 'y': 12, 'type': 'w', 'hp': 1}, {'created': 0, 'x': 4, 'y': 9, 'type': 'w', 'hp': 1}, {'created': 0, 'x': 10, 'y': 9, 'type': 'w', 'hp': 1}, {'created': 0, 'x': 9, 'y': 11, 'type': 'w', 'hp': 1}, {'created': 0, 'x': 5, 'y': 11, 'type': 'w', 'hp': 1}, {'created': 0, 'x': 9, 'y': 3, 'type': 'o', 'hp': 3}, {'created': 0, 'x': 5, 'y': 3, 'type': 'o', 'hp': 3}, {'created': 0, 'x': 12, 'y': 1, 'type': 'o', 'hp': 3}, {'created': 0, 'x': 2, 'y': 1, 'type': 'o', 'hp': 3}, {'created': 0, 'x': 14, 'y': 4, 'type': 'o', 'hp': 3}, {'created': 0, 'x': 0, 'y': 4, 'type': 'o', 'hp': 3}, {'created': 0, 'x': 2, 'y': 6, 'type': 'o', 'hp': 3}, {'created': 0, 'x': 12, 'y': 6, 'type': 'o', 'hp': 3}, {'created': 0, 'x': 12, 'y': 7, 'type': 'o', 'hp': 3}, {'created': 0, 'x': 2, 'y': 7, 'type': 'o', 'hp': 3}, {'created': 0, 'x': 3, 'y': 4, 'type': 'o', 'hp': 3}, {'created': 0, 'x': 11, 'y': 4, 'type': 'o', 'hp': 3}, {'created': 0, 'x': 2, 'y': 8, 'type': 'o', 'hp': 3}, {'created': 0, 'x': 12, 'y': 8, 'type': 'o', 'hp': 3}, {'created': 0, 'x': 4, 'y': 13, 'type': 'o', 'hp': 3}, {'created': 0, 'x': 10, 'y': 13, 'type': 'o', 'hp': 3}, {'created': 0, 'x': 5, 'y': 1, 'type': 'o', 'hp': 3}, {'created': 0, 'x': 9, 'y': 1, 'type': 'o', 'hp': 3}, {'created': 0, 'x': 11, 'y': 8, 'type': 'o', 'hp': 3}, {'created': 0, 'x': 3, 'y': 8, 'type': 'o', 'hp': 3}, {'created': 0, 'x': 8, 'y': 10, 'type': 'o', 'hp': 3}, {'created': 0, 'x': 6, 'y': 10, 'type': 'o', 'hp': 3}, {'created': 0, 'x': 4, 'y': 7, 'type': 'o', 'hp': 3}, {'created': 0, 'x': 10, 'y': 7, 'type': 'o', 'hp': 3}, {'created': 0, 'x': 5, 'y': 9, 'type': 'o', 'hp': 3}, {'created': 0, 'x': 9, 'y': 9, 'type': 'o', 'hp': 3}, {'created': 0, 'x': 5, 'y': 13, 'type': 'o', 'hp': 3}, {'created': 0, 'x': 9, 'y': 13, 'type': 'o', 'hp': 3}, {'created': 0, 'x': 0, 'y': 8, 'type': 'o', 'hp': 3}, {'created': 0, 'x': 14, 'y': 8, 'type': 'o', 'hp': 3}, {'created': 0, 'x': 12, 'y': 0, 'type': 'o', 'hp': 3}, {'created': 0, 'x': 2, 'y': 0, 'type': 'o', 'hp': 3}, {'created': 0, 'x': 1, 'y': 0, 'type': 'o', 'hp': 3}, {'created': 0, 'x': 13, 'y': 0, 'type': 'o', 'hp': 3}, {'created': 0, 'x': 11, 'y': 14, 'type': 'o', 'hp': 3}, {'created': 0, 'x': 3, 'y': 14, 'type': 'o', 'hp': 3}, {'created': 0, 'x': 10, 'y': 12, 'type': 'o', 'hp': 3}, {'created': 0, 'x': 4, 'y': 12, 'type': 'o', 'hp': 3}, {'created': 0, 'x': 14, 'y': 0, 'type': 'o', 'hp': 3}, {'created': 0, 'x': 0, 'y': 0, 'type': 'o', 'hp': 3}, {'created': 0, 'x': 2, 'y': 5, 'type': 'o', 'hp': 3}, {'created': 0, 'x': 12, 'y': 5, 'type': 'o', 'hp': 3}, {'created': 0, 'x': 13, 'y': 1, 'type': 'o', 'hp': 3}, {'created': 0, 'x': 1, 'y': 1, 'type': 'o', 'hp': 3}, {'created': 0, 'x': 3, 'y': 0, 'type': 'o', 'hp': 3}, {'created': 0, 'x': 11, 'y': 0, 'type': 'o', 'hp': 3}, {'created': 0, 'x': 11, 'y': 3, 'type': 'o', 'hp': 3}, {'created': 0, 'x': 3, 'y': 3, 'type': 'o', 'hp': 3}, {'created': 0, 'x': 6, 'y': 3, 'type': 'o', 'hp': 3}, {'created': 0, 'x': 8, 'y': 3, 'type': 'o', 'hp': 3}, {'created': 0, 'x': 5, 'y': 7, 'type': 'o', 'hp': 3}, {'created': 0, 'x': 9, 'y': 7, 'type': 'o', 'hp': 3}, {'created': 0, 'x': 12, 'y': 13, 'type': 'o', 'hp': 3}, {'created': 0, 'x': 2, 'y': 13, 'type': 'o', 'hp': 3}, {'created': 0, 'x': 2, 'y': 14, 'type': 'o', 'hp': 3}, {'created': 0, 'x': 12, 'y': 14, 'type': 'o', 'hp': 3}, {'created': 0, 'x': 12, 'y': 4, 'type': 'o', 'hp': 3}, {'created': 0, 'x': 2, 'y': 4, 'type': 'o', 'hp': 3}, {'created': 0, 'x': 0, 'y': 11, 'type': 'o', 'hp': 3}, {'created': 0, 'x': 14, 'y': 11, 'type': 'o', 'hp': 3}, {'created': 0, 'x': 12, 'y': 3, 'type': 'o', 'hp': 3}, {'created': 0, 'x': 2, 'y': 3, 'type': 'o', 'hp': 3}, {'created': 0, 'x': 0, 'y': 3, 'type': 'o', 'hp': 3}, {'created': 0, 'x': 14, 'y': 3, 'type': 'o', 'hp': 3}, {'created': 0, 'x': 8, 'y': 1, 'type': 'o', 'hp': 3}, {'created': 0, 'x': 6, 'y': 1, 'type': 'o', 'hp': 3}, {'created': 0, 'x': 10, 'y': 3, 'type': 'o', 'hp': 3}, {'created': 0, 'x': 4, 'y': 3, 'type': 'o', 'hp': 3}, {'created': 0, 'x': 12, 'y': 11, 'type': 'o', 'hp': 3}, {'created': 0, 'x': 2, 'y': 11, 'type': 'o', 'hp': 3}, {'created': 0, 'x': 6, 'y': 0, 'type': 'o', 'hp': 3}, {'created': 0, 'x': 8, 'y': 0, 'type': 'o', 'hp': 3}, {'created': 0, 'x': 1, 'y': 11, 'type': 'o', 'hp': 3}, {'created': 0, 'x': 13, 'y': 11, 'type': 'o', 'hp': 3}, {'created': 0, 'x': 10, 'y': 0, 'type': 'o', 'hp': 3}, {'created': 0, 'x': 4, 'y': 0, 'type': 'o', 'hp': 3}, {'created': 0, 'x': 0, 'y': 1, 'type': 'o', 'hp': 3}, {'created': 0, 'x': 14, 'y': 1, 'type': 'o', 'hp': 3}, {'created': 0, 'x': 2, 'y': 10, 'type': 'o', 'hp': 3}, {'created': 0, 'x': 12, 'y': 10, 'type': 'o', 'hp': 3}, {'created': 0, 'x': 10, 'y': 1, 'type': 'o', 'hp': 3}, {'created': 0, 'x': 4, 'y': 1, 'type': 'o', 'hp': 3}], 'world': {'width': 15, 'height': 15}, 'tick': 0, 'config': {'tick_rate_hz': 10, 'game_duration_ticks': 200, 'fire_spawn_interval_ticks': 2}, 'connection': {'id': 5, 'role': 'admin', 'agent_id': None}}}
            # 4. 用它创建 Gym 环境（只是用来初始化包裹器）
            gym.make("bomberland-env", current_state["payload"])

            # 我们需要等待游戏引擎初始化完成
            await asyncio.sleep(0.5)
            
            # print(f"已保存检查点: episode {episode+1}")
            # print(f"当前的状态是: {current_state}")
    
            # 执行一个完整的游戏回合
           
            for step in range(MAX_STEPS_PER_EPISODE):
                
                # 将状态转换为观察向量
                self_states, local_maps, agent_units_ids, agent_alive_units_ids = state_to_observations(current_state, agent_id="a")

                alive_mask = get_alive_mask(agent_units_ids, agent_alive_units_ids)

                action_indices, log_probs, value = agent.select_actions(self_states, local_maps, alive_mask)

                # Update the values
                action_indices = action_indices[0]
                log_probs = log_probs[0]
                value = value[0]
                
                # 转换为游戏动作
                game_actions = action_index_to_game_action(action_indices, current_state, agent_id="a")
                
                # 记录执行前的状态
                prev_state = current_state.copy()
                
                # 执行动作
                next_state, done, info = await gym.step(game_actions)
                 
                # 如果返回None，则可能是发生了错误
                if next_state is None:
                    print("警告: step 返回了 None，跳出当前 episode")
                    break
                
                if next_state["tick"] >= next_state["config"]["game_duration_ticks"]:
                    done = True

                alive_units = filter_alive_units("a", next_state["agents"]["a"]["unit_ids"], next_state["unit_state"])
                alive_enemies = filter_alive_units("b", next_state["agents"]["b"]["unit_ids"], next_state["unit_state"])

                if len(alive_units) == 0 or len(alive_enemies) == 0:
                    done = True

                # 计算奖励
                reward = calculate_reward(next_state, prev_state, agent_id="a")
                total_reward += reward
                
                # 记录到缓冲区
                episode_buffer.append((self_states, local_maps, action_indices, log_probs, reward, value, done))
                
                # 更新当前状态
                current_state = next_state
                
                # 打印进度
                if step % 10 == 0:
                    print(f"Step {step+1}/{MAX_STEPS_PER_EPISODE}, Reward: {reward:.2f}, Total: {total_reward:.2f}")
                
                # 检查游戏是否结束
                if done:
                    print(f"当前 Episode {episode+1}， 游戏结束于 step {step+1}, 总奖励: {total_reward:.2f}")
                    break
            
            # 处理经验并更新策略
            agent.update_from_buffer(episode_buffer)
            
            # 记录奖励
            episode_rewards.append(total_reward)
            
            # 日志记录
            if (episode + 1) % LOG_FREQUENCY == 0:
                avg_reward = np.mean(episode_rewards[-LOG_FREQUENCY:])
                print(f"Episodes {episode-LOG_FREQUENCY+2}-{episode+1}: 平均奖励 = {avg_reward:.2f}")
            
            # 保存检查点
            if (episode + 1) % SAVE_FREQUENCY == 0:
                torch.save({
                    "model": agent.model.state_dict(),
                    "optimizer": agent.optimizer.state_dict(),
                    "episode": episode + 1
                }, checkpoint_path)
                print(f"已保存检查点: episode {episode+1}")
        
        except Exception as e:
            print(f"Episode {episode+1} 过程中发生错误: {e}")
            import traceback
            traceback.print_exc()
        
        finally:
            # 每个 episode 结束后关闭环境连接
            try:
                await gym.close()
                print(f"Episode {episode+1} 环境已关闭")
            except Exception as e:
                print(f"关闭环境时发生错误: {e}")
    
    print("训练完成，所有环境已关闭")

if __name__ == "__main__":
    asyncio.run(train())
