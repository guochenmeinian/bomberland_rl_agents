# agent/ppo_agent.py
import torch
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from model.ppo_model import PPOModel

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
        
        # Scoring system for each bomb
        bomb_scores = []
        
        for bomb_x, bomb_y in candidate_targets:
            score = 0
            blast_diameter = 0
            
            # Find the bomb's blast diameter in entities
            for entity in game_state["entities"]:
                if entity["type"] == "b" and entity["x"] == bomb_x and entity["y"] == bomb_y:
                    blast_diameter = entity.get("blast_diameter", 3)  # Default to 3 if not found
                    break
            
            blast_radius = blast_diameter // 2
            
            # Check for enemy units in blast range
            enemy_id = "b" if game_state["connection"]["agent_id"] == "a" else "a"
            enemy_unit_ids = game_state["agents"][enemy_id]["unit_ids"]
            
            for enemy_unit_id in enemy_unit_ids:
                if enemy_unit_id in game_state["unit_state"]:
                    enemy_unit = game_state["unit_state"][enemy_unit_id]
                    enemy_x, enemy_y = enemy_unit["coordinates"]
                    
                    # Check if enemy is in blast range (same row or column and within blast radius)
                    if (bomb_x == enemy_x and abs(bomb_y - enemy_y) <= blast_radius) or \
                    (bomb_y == enemy_y and abs(bomb_x - enemy_x) <= blast_radius):
                        # Higher score for enemies with less health
                        score += 10 * (4 - enemy_unit["hp"])  # Assuming max HP is 3
                        
                        # Extra points if enemy is stunned (easier target)
                        if enemy_unit.get("stunned", 0) > game_state["tick"]:
                            score += 5
            
            # Check for friendly units in blast range (negative score)
            friendly_id = game_state["connection"]["agent_id"]
            friendly_unit_ids = game_state["agents"][friendly_id]["unit_ids"]
            
            for friendly_unit_id in friendly_unit_ids:
                if friendly_unit_id in game_state["unit_state"] and friendly_unit_id != unit_id:  # Don't count self
                    friendly_unit = game_state["unit_state"][friendly_unit_id]
                    friendly_x, friendly_y = friendly_unit["coordinates"]
                    
                    # Check if friendly is in blast range
                    if (bomb_x == friendly_x and abs(bomb_y - friendly_y) <= blast_radius) or \
                    (bomb_y == friendly_y and abs(bomb_x - friendly_x) <= blast_radius):
                        # Heavily penalize endangering our own units
                        score -= 20
                        
                        # Even higher penalty if friendly unit is low on health
                        if friendly_unit["hp"] == 1:
                            score -= 10
            
            # Check for destructible blocks in range
            for entity in game_state["entities"]:
                if entity["type"] in ["w", "o"] and \
                ((entity["x"] == bomb_x and abs(entity["y"] - bomb_y) <= blast_radius) or \
                    (entity["y"] == bomb_y and abs(entity["x"] - bomb_x) <= blast_radius)):
                    # Modest bonus for destroying blocks
                    score += 2
                    
                    # Extra points for ore blocks which might drop power-ups
                    if entity["type"] == "o":
                        score += 1
            
            # Bonus if the bomb is about to expire anyway
            for entity in game_state["entities"]:
                if entity["type"] == "b" and entity["x"] == bomb_x and entity["y"] == bomb_y:
                    ticks_until_explosion = entity["expires"] - game_state["tick"]
                    if ticks_until_explosion <= 5:  # If bomb will explode soon anyway
                        score += 3
                        
                        # Even higher bonus if it's about to explode next tick
                        if ticks_until_explosion <= 2:
                            score += 5
            
            bomb_scores.append((bomb_x, bomb_y, score))
        
        # If no bomb has a positive score, prefer bombs that will explode soon
        if all(score <= 0 for _, _, score in bomb_scores):
            for i, (bomb_x, bomb_y, score) in enumerate(bomb_scores):
                for entity in game_state["entities"]:
                    if entity["type"] == "b" and entity["x"] == bomb_x and entity["y"] == bomb_y:
                        ticks_until_explosion = entity["expires"] - game_state["tick"]
                        bomb_scores[i] = (bomb_x, bomb_y, -ticks_until_explosion)  # Negative so lowest ticks = highest score
        
        # Select the bomb with the highest score
        if bomb_scores:
            best_bomb = max(bomb_scores, key=lambda x: x[2])
            if best_bomb[2] > -10:  # Don't detonate if all options are very bad
                return (best_bomb[0], best_bomb[1])
        
        # If all bombs have very negative scores, better not to detonate
        return None


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
                action = torch.tensor(6, device=self.device)
                log_prob = torch.tensor(0.0, device=self.device)  # log(1) = 0
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

        actions = torch.stack(actions, dim=1)      # (B, num_units)
        log_probs = torch.stack(log_probs, dim=1)  # (B, num_units)

        return actions.cpu().numpy(), log_probs.cpu().numpy(), value.cpu().numpy(), detonate_targets


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

        states = torch.tensor(states, dtype=torch.float32).to(self.device)
        maps = torch.tensor(maps, dtype=torch.float32).to(self.device)
        actions = torch.tensor(actions).to(self.device)
        old_log_probs = torch.tensor(old_log_probs, dtype=torch.float32).to(self.device)
        returns = torch.tensor(returns, dtype=torch.float32).to(self.device)
        advantages = torch.tensor(advantages, dtype=torch.float32).to(self.device)

        dataset = torch.utils.data.TensorDataset(states, maps, actions, old_log_probs, returns, advantages)
        loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

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

        self.memory.clear()
