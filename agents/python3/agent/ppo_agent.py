# agent/ppo_agent.py
from encodings import normalize_encoding
import torch
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from model.ppo_model import PPOModel
import random
import wandb

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
        
        # Get all bombs that belong to this unit
        if len(current_bomb_infos) == 0:
            return None

        candidate_targets = []
        for x, y, bomb_owner in current_bomb_infos:
            if bomb_owner == unit_id:
                candidate_targets.append((x, y))
        
        if len(candidate_targets) == 0:
            return None
        
        bomb_scores = []

        for bomb_x, bomb_y in candidate_targets:
            score = 0
            blast_diameter = 0
            ticks_until_explosion = 9999

            # Find the blast diameter for the bomb
            for entity in game_state["entities"]:
                if entity["type"] == "b" and entity["x"] == bomb_x and entity["y"] == bomb_y:
                    blast_diameter = entity.get("blast_diameter", 3)  # Default to 3 if missing
                    break

            blast_radius = blast_diameter // 2

            # Identify self_agent and enemy_agent info
            unit_to_agent = {}
            for agent_id, agent_info in game_state["agents"].items():
                for uid in agent_info["unit_ids"]:
                    unit_to_agent[uid] = agent_id

            if unit_id not in unit_to_agent:
                print(f"[Warning] Unit {unit_id} not found in agents.")
                return None

            self_agent = unit_to_agent[unit_id]
            enemy_agent = "b" if self_agent == "a" else "a"

            teammates = game_state["agents"][self_agent]["unit_ids"]
            enemies = game_state["agents"][enemy_agent]["unit_ids"]

            # Score based on enemies hit
            for enemy_id in enemies:
                if enemy_id in game_state["unit_state"]:
                    enemy_unit = game_state["unit_state"][enemy_id]
                    ex, ey = enemy_unit["coordinates"]

                    if (bomb_x == ex and abs(bomb_y - ey) <= blast_radius) or \
                    (bomb_y == ey and abs(bomb_x - ex) <= blast_radius):
                        # Prefer damaging low-HP enemies
                        score += 10 * (4 - enemy_unit["hp"])

                        # Bonus if enemy is stunned
                        if enemy_unit.get("stunned", 0) > game_state["tick"]:
                            score += 10

            # Penalty based on teammates in blast
            for mate_id in teammates:
                if mate_id in game_state["unit_state"] and mate_id != unit_id:  # Don't count self
                    mate_unit = game_state["unit_state"][mate_id]
                    mx, my = mate_unit["coordinates"]

                    if (bomb_x == mx and abs(bomb_y - my) <= blast_radius) or \
                    (bomb_y == my and abs(bomb_x - mx) <= blast_radius):
                        # Heavily penalize hitting teammates
                        score -= 20

                        # Even worse if teammate has low HP
                        if mate_unit["hp"] == 1:
                            score -= 10
            
            # Check for destructible blocks in range
            for entity in game_state["entities"]:
                ex, ey = entity["x"], entity["y"]
                if entity["type"] in ["w", "o"] and \
                    (bomb_x == ex and abs(bomb_y - ey) <= blast_radius) or \
                    (bomb_y == ey and abs(bomb_x - ex) <= blast_radius):

                    # Modest bonus for destroying blocks
                    if entity["type"] == "w":  # wood block，hp == 1
                        score += 5
                    elif entity["type"] == "o":  # ore block，default hp == 3
                        hp = entity.get("hp", 3)
                        if hp == 1:
                            # only 1 hp left, treat as wood block
                            score += 5
                        elif hp == 2:
                            # 血厚的矿，加少点
                            score += 3
                        else:        
                            score += 1

                # Bonus if the bomb is about to expire anyway
                if entity["type"] == "b" and entity["x"] == bomb_x and entity["y"] == bomb_y:
                    ticks_until_explosion = entity["expires"] - game_state["tick"]
                    if ticks_until_explosion == 5:  # If bomb will explode soon anyway
                        score += 1
                                    
            bomb_scores.append((bomb_x, bomb_y, score, ticks_until_explosion))
        
        # If no bomb has a positive score, prefer bombs that will explode soon
        # if all(score <= 0 for _, _, score, _ in bomb_scores):
        #     bomb_scores = [(bx, by, -ticks, ticks) for bx, by, _, ticks in bomb_scores]
        
        # Select the bomb with the highest score
        if bomb_scores:
            best_bomb = max(bomb_scores, key=lambda x: x[2])
            return (best_bomb[0], best_bomb[1])
            
        # If all bombs have very negative scores, better not to detonate
        return None # shouldn't be triggered
        

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
            if alive_mask[i] == 0: # unit already dead, do nothing
                action = torch.tensor(6, device=self.device).unsqueeze(0) 
                log_prob = torch.tensor(0.0, device=self.device).unsqueeze(0) 
                detonate_targets.append(None)
            else:
                my_unit_id = unit_ids[i]
                mask = torch.ones_like(logits, dtype=torch.bool)
                
                if current_bomb_count >= 3: # no more bomb allo
                    mask[0, 4] = False 
                
                candidate_target = self.select_detonate_target(my_unit_id, current_bomb_infos, current_state)
                
                if candidate_target is None: # no detonate target, mask detonate action (shouldn't be triggered though)
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


    def update_from_buffer(self, episode_buffer, current_episode):
        if not episode_buffer:
            return
        
        self.memory.clear()

        for sequence in episode_buffer:
            # sequence 是一个 list of (state, map, action, log_prob, reward, value, done)
            rewards = np.array([step[4] for step in sequence])
            values_arr = np.array([step[5] for step in sequence])
            dones = np.array([step[6] for step in sequence])

            advantages, returns = self.compute_gae(rewards, values_arr, dones)

            new_sequence = []
            for (state, map_data, action, log_prob, reward, value, done), adv, ret in zip(sequence, advantages, returns):
                new_sequence.append((state, map_data, action, log_prob, ret, adv))

            self.memory.append(new_sequence)
        
        self.update(current_episode)

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

    def update(self, current_episode, epochs=4, batch_size=4):
        # memory 是 list of sequences
        all_sequences = self.memory

        # 预处理每个sequence
        sequences = []
        for seq in all_sequences:
            states, maps, actions, old_log_probs, returns, advantages = zip(*seq)
            states = torch.tensor(np.array(states), dtype=torch.float32).to(self.device)
            maps = torch.tensor(np.array(maps), dtype=torch.float32).to(self.device)
            actions = torch.tensor(actions).to(self.device)
            old_log_probs = torch.tensor(old_log_probs, dtype=torch.float32).to(self.device)
            returns = torch.tensor(returns, dtype=torch.float32).to(self.device)
            advantages = torch.tensor(advantages, dtype=torch.float32).to(self.device)

            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
            sequences.append((states, maps, actions, old_log_probs, returns, advantages))

        total_policy_loss_value = 0
        total_value_loss_value = 0
        total_loss_value = 0

        for _ in range(epochs):
            # print("LEN:", len(sequences))
            random_batches = [sequences[i:i + batch_size] for i in range(0, len(sequences), batch_size)]
            for batch in random_batches:
                policy_loss_sum = 0
                value_loss_sum = 0

                for seq in batch:
                    s_seq, m_seq, a_seq, old_lp_seq, ret_seq, adv_seq = seq
                    T = s_seq.shape[0]

                    policy_losses = []
                    value_losses = []

                    for t in range(T):
                        s_t = s_seq[t].unsqueeze(0)  # (1, num_units, self_state_dim)
                        m_t = m_seq[t] # .unsqueeze(0)  # (1, C, H, W)

                        logits_list, value = self.model(s_t, m_t)

                        total_unit_loss = 0
                        for unit_idx, logits in enumerate(logits_list):
                            dist = torch.distributions.Categorical(logits=logits)
                            log_prob = dist.log_prob(a_seq[t, unit_idx])
                            ratio = torch.exp(log_prob - old_lp_seq[t, unit_idx])

                            surr1 = ratio * adv_seq[t]
                            surr2 = torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps) * adv_seq[t]
                            total_unit_loss += -torch.min(surr1, surr2)

                        policy_losses.append(total_unit_loss)
                        value_losses.append(F.mse_loss(value.squeeze(0), ret_seq[t]))

                    policy_loss = torch.stack(policy_losses).mean()
                    value_loss = torch.stack(value_losses).mean()
                    loss = policy_loss + 0.5 * value_loss

                    policy_loss_sum += policy_loss
                    value_loss_sum += value_loss

                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()

                total_policy_loss_value += policy_loss_sum.item()
                total_value_loss_value += value_loss_sum.item()
                total_loss_value += (policy_loss_sum.item() + value_loss_sum.item())

        num_batches = len(random_batches) * epochs
        avg_policy_loss = total_policy_loss_value / num_batches
        avg_value_loss = total_value_loss_value / num_batches
        avg_loss = total_loss_value / num_batches

        # 🔵 wandb记录
        wandb.log({
            "train/policy_loss": avg_policy_loss,
            "train/value_loss": avg_value_loss,
            "train/total_loss": avg_loss,
            "train/episode": current_episode
        }, step=current_episode)

        print(f"Update stats - Policy loss: {avg_policy_loss:.4f}, Value loss: {avg_value_loss:.4f}, Total loss: {avg_loss:.4f}")

        self.memory.clear()
