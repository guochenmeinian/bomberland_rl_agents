# rl_agent.py (supports multi-unit attention PPO)
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torch.nn.functional as F
from ppo_model import PPOModel
from collections import deque

class PPOAgent:
    def __init__(self, self_state_dim=10, map_channels=8, map_size=5, action_dim=7, num_units=3,
                 gamma=0.99, lam=0.95, clip_eps=0.2, lr=3e-4, device='cpu'):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.num_units = num_units
        self.model = PPOModel(self_state_dim, map_channels, map_size, action_dim=action_dim, num_units=num_units).to(device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)

        self.gamma = gamma
        self.lam = lam
        self.clip_eps = clip_eps

        self.memory = []

    def select_actions(self, self_states, local_maps, alive_mask):
        # 检查输入维度
        if isinstance(self_states, np.ndarray) and len(self_states.shape) == 2:  # (num_units, self_state_dim)
            self_states = np.expand_dims(self_states, 0)  # 添加批次维度 -> (1, num_units, self_state_dim)
        
        if isinstance(local_maps, np.ndarray) and len(local_maps.shape) == 4:  # (num_units, C, 5, 5)
            local_maps = np.expand_dims(local_maps, 0)  # 添加批次维度 -> (1, num_units, C, 5, 5)
        
        # The batch size in this project should be 1, so B = 1
        # print(f"The input dim: self_states={self_states.shape}, local_maps={local_maps.shape}")
        self_states = torch.tensor(self_states, dtype=torch.float32).to(self.device)    # (B, num_units, self_state_dim)
        local_maps = torch.tensor(local_maps, dtype=torch.float32).to(self.device)      # (B, num_units, C, 5, 5)

        with torch.no_grad():
            logits_list, value = self.model(self_states, local_maps)
        
        # Since each agent has 3 units, we need to store them separately
        actions = []
        log_probs = []

        for i, logits in enumerate(logits_list):
            if alive_mask[i] == 0:
                action = torch.tensor(6, device=self.device)
                log_prob = torch.tensor(0.0, device=self.device)  # log(1) = 0
            else:
                probs = torch.softmax(logits, dim=-1)
                dist = torch.distributions.Categorical(probs)
                # Sample an action from the distribution
                action = dist.sample()

            actions.append(action)
            log_probs.append(dist.log_prob(action))

        actions = torch.stack(actions, dim=1)      # (B, num_units)
        log_probs = torch.stack(log_probs, dim=1)  # (B, num_units)

        return actions.cpu().numpy(), log_probs.cpu().numpy(), value.cpu().numpy()

    def store_transition(self, transition):
        self.memory.append(transition)

    def compute_gae(self, rewards, values, dones):
        """
        Compute Generalized Advantage Estimation
        
        Parameters:
        - rewards: list of rewards for each step
        - values: list of value estimates for each state (including last state)
        - dones: list of done flags for each step
        
        Returns:
        - advantages: list of advantage estimates for each step
        - returns: list of return estimates for each step
        """
        advantages = []
        gae = 0
        
        # 确保values数组维度正确
        if isinstance(values, np.ndarray) and len(values.shape) > 1 and values.shape[1] > 1:
            # 如果values是二维数组且第二维大于1，取第一个维度
            values = values.squeeze()
        
        # 确保values长度比rewards长1（包含最后状态的值）
        if len(values) != len(rewards) + 1:
            # 如果不是，我们复制最后一个值
            last_value = values[-1]
            values = np.append(values, last_value)
        
        for step in reversed(range(len(rewards))):
            # 计算时序差异误差 (δ)
            # δ = r_t + γV(s_{t+1}) - V(s_t)
            delta = rewards[step] + self.gamma * values[step + 1] * (1 - dones[step]) - values[step]
            
            # 计算GAE
            # A_t = δ_t + γλA_{t+1}
            gae = delta + self.gamma * self.lam * (1 - dones[step]) * gae
            advantages.insert(0, gae)
        
        # 计算回报 (advantages + values)
        returns = np.array(advantages) + values[:-1]
        
        return advantages, returns

    def update(self, epochs=4, batch_size=64):
        states, maps, actions, old_log_probs, returns, advantages = zip(*self.memory)

        states = torch.tensor(states, dtype=torch.float32).to(self.device)           # (B, num_units, self_state_dim)
        maps = torch.tensor(maps, dtype=torch.float32).to(self.device)              # (B, num_units, C, 5, 5)
        actions = torch.tensor(actions).to(self.device)                              # (B, num_units)
        old_log_probs = torch.tensor(old_log_probs, dtype=torch.float32).to(self.device)  # (B, num_units)
        returns = torch.tensor(returns, dtype=torch.float32).to(self.device)         # (B,)
        advantages = torch.tensor(advantages, dtype=torch.float32).to(self.device)   # (B,)

        dataset = torch.utils.data.TensorDataset(states, maps, actions, old_log_probs, returns, advantages)
        loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

        for _ in range(epochs):
            for batch in loader:
                b_s, b_m, b_a, b_old_lp, b_ret, b_adv = batch
                logits_list, values = self.model(b_s, b_m)

                total_policy_loss = 0
                for i, logits in enumerate(logits_list):
                    dist = torch.distributions.Categorical(logits=logits)
                    log_probs = dist.log_prob(b_a[:, i])
                    ratio = torch.exp(log_probs - b_old_lp[:, i])
                    surr1 = ratio * b_adv
                    surr2 = torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps) * b_adv
                    policy_loss = -torch.min(surr1, surr2).mean()
                    total_policy_loss += policy_loss

                value_loss = F.mse_loss(values, b_ret)
                loss = total_policy_loss + 0.5 * value_loss

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

        self.memory.clear()
    
    # 在PPOAgent类中添加此方法，放在update方法之后

    def update_from_buffer(self, episode_buffer):
        """
        Process experience buffer for PPO updates
        
        Parameters:
        - episode_buffer: List of experience tuples
        """
        # 如果buffer为空，直接返回
        if not episode_buffer:
            print("警告: 空的episode buffer, 跳过更新")
            return
            
        # 提取数据
        states = []
        maps = []
        actions = []
        log_probs = []
        rewards = []
        values = []
        dones = []
        
        for transition in episode_buffer:
            states.append(transition[0])
            maps.append(transition[1])
            actions.append(transition[2])
            log_probs.append(transition[3])
            rewards.append(transition[4])
            values.append(transition[5])
            dones.append(transition[6])
        
        # 转换为numpy数组以提高计算稳定性
        rewards = np.array(rewards)
        values_arr = np.array(values)
        dones = np.array(dones)
        
        # 为最后状态添加值估计
        # 如果游戏结束，则使用0；否则使用最后状态的值
        if len(values) > 0:
            if dones[-1]:
                last_value = np.zeros_like(values_arr[0])
            else:
                last_value = values_arr[-1]
        else:
            last_value = np.zeros((1,))
        
        # 计算GAE和回报
        advantages, returns = self.compute_gae(rewards, values_arr, dones)
        
        # 准备内存缓冲区
        self.memory.clear()
        for i in range(len(states)):
            self.memory.append((
                states[i],
                maps[i],
                actions[i],
                log_probs[i],
                returns[i],
                advantages[i]
            ))
        
        # 用收集的经验更新策略
        self.update(epochs=4, batch_size=min(64, len(self.memory)))