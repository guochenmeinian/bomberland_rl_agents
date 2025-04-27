# agent/ppo_agent.py
from encodings import normalize_encoding
import torch
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from model.ppo_model import PPOModel
import random

class PPOAgent:
    def __init__(self, config):
        self.device = config.device
        self.model = PPOModel(
            self_state_dim=config.self_state_dim,
            map_channels=config.map_channels,
            map_size=config.map_size,
            action_dim=config.action_dim,
            num_units=config.num_units
        ).to(self.device)

        self.optimizer = optim.Adam(self.model.parameters(), lr=config.lr)
        self.gamma = config.gamma
        self.lam = config.lam
        self.clip_eps = config.clip_eps
        self.memory = []


    def select_detonate_target(self, unit_id, current_bomb_infos, game_state):
        if isinstance(game_state, dict) and "payload" in game_state:
            game_state = game_state["payload"]
            
        if len(current_bomb_infos) == 0:
            return None

        # Get all bombs that belong to this unit
        candidate_targets = []
        for x, y, bomb_owner in current_bomb_infos:
            if bomb_owner == unit_id:
                candidate_targets.append((x, y))
        
        if len(candidate_targets) == 0:
            return None


        return random.choice(candidate_targets)
        

    def select_actions(self, self_states, full_map, alive_mask, current_bomb_infos, current_bomb_count, unit_ids, current_state):
        # 检查输入维度
        if isinstance(self_states, np.ndarray) and len(self_states.shape) == 2:  # (num_units, self_state_dim)
            self_states = np.expand_dims(self_states, 0)  # 添加批次维度 -> (1, num_units, self_state_dim)
        
        if isinstance(full_map, np.ndarray) and len(full_map.shape) == 3:  # (C, H, W)
            full_map = np.expand_dims(full_map, 0)  # 添加批次维度 -> (1, C, H, W)
        
        # The batch size in this project should be 1, so B = 1
        self_states = torch.tensor(self_states, dtype=torch.float32).to(self.device)    # (B, num_units, self_state_dim)
        full_map = torch.tensor(full_map, dtype=torch.float32).to(self.device)      # (B, C, H, W)

        with torch.no_grad():
            logits_list, value = self.model(self_states, full_map)
        
        # Since each agent has 3 units, we need to store them separately
        actions = []
        log_probs = []
        detonate_targets = []

        for i, logits in enumerate(logits_list):
            if alive_mask[i] == 0:
                action = torch.tensor(6, device=self.device).unsqueeze(0) 
                log_prob = torch.tensor(0.0, device=self.device).unsqueeze(0) 
                detonate_targets.append(None)
            else:
                my_unit_id = unit_ids[i]
                mask = torch.ones_like(logits, dtype=torch.bool)
                
                if current_bomb_count >= 3:
                    mask[0, 4] = False 
                
                candidate_target = self.select_detonate_target(my_unit_id, current_bomb_infos, current_state)
                if candidate_target is None:
                    mask[0, 5] = False

                logits_masked = logits.clone()
                logits_masked[~mask] = -1e10

                probs = torch.softmax(logits_masked, dim=-1)
                dist = torch.distributions.Categorical(probs)
                # Sample an action from the distribution
                action = dist.sample()
                log_prob = dist.log_prob(action)

                if action.item() == 4:
                    current_bomb_count += 1
                
                if action.item() == 5:
                    detonate_targets.append(candidate_target)
                else:
                    detonate_targets.append(None)

            actions.append(action)
            log_probs.append(log_prob)

        actions = torch.cat(actions, dim=0).unsqueeze(0)      # (B=1, num_units)
        log_probs = torch.cat(log_probs, dim=0).unsqueeze(0)  # (B=1, num_units)

        return actions.cpu().numpy(), log_probs.cpu().numpy(), value.squeeze().cpu().numpy(), detonate_targets



    def update_from_buffer(self, episode_buffer):
        if not episode_buffer:
            return

        states, maps, actions, log_probs, rewards, values, dones = zip(*episode_buffer)

        rewards = np.array(rewards)
        values_arr = np.array(values)
        dones = np.array(dones)

        advantages, returns = self.compute_gae(rewards, values_arr, dones)

        self.memory.clear()
        for i in range(len(states)):
            self.memory.append((states[i], maps[i], actions[i], log_probs[i], returns[i], advantages[i]))

        self.update()

    def compute_gae(self, rewards, values, dones):
        advantages = []
        gae = 0
        values = np.append(values, values[-1])

        for step in reversed(range(len(rewards))):
            delta = rewards[step] + self.gamma * values[step + 1] * (1 - dones[step]) - values[step]
            gae = delta + self.gamma * self.lam * (1 - dones[step]) * gae
            advantages.insert(0, gae)

        returns = np.array(advantages) + values[:-1]
        return advantages, returns

    def update(self, epochs=4, batch_size=64):
        states, maps, actions, old_log_probs, returns, advantages = zip(*self.memory)

        states = torch.tensor(np.array(states), dtype=torch.float32).to(self.device)
        maps = torch.tensor(np.array(maps), dtype=torch.float32).squeeze(1).to(self.device)
        actions = torch.tensor(np.array(actions)).to(self.device)
        old_log_probs = torch.tensor(np.array(old_log_probs), dtype=torch.float32).to(self.device)
        returns = torch.tensor(np.array(returns), dtype=torch.float32).to(self.device)
        advantages = torch.tensor(np.array(advantages), dtype=torch.float32).to(self.device)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)


        dataset = torch.utils.data.TensorDataset(states, maps, actions, old_log_probs, returns, advantages)
        loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

        total_policy_loss_value = 0 
        total_value_loss_value = 0
        total_loss_value = 0

        for _ in range(epochs):
            for b_s, b_m, b_a, b_old_lp, b_ret, b_adv in loader:
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

                total_policy_loss_value += total_policy_loss.item()
                total_value_loss_value += value_loss.item()
                total_loss_value += loss.item()
            
        num_batches = len(loader) * epochs
        avg_policy_loss = total_policy_loss_value / num_batches
        avg_value_loss = total_value_loss_value / num_batches
        avg_loss = total_loss_value / num_batches

        print(f"Update stats - Policy loss: {avg_policy_loss:.4f}, Value loss: {avg_value_loss:.4f}, Total loss: {avg_loss:.4f}")

        self.memory.clear()
