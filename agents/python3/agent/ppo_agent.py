# agent/ppo_agent.py
from encodings import normalize_encoding
import torch
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from model.ppo_model import PPOModel
import random
import wandb
from collections import deque
from config import Config 

class TemporalWindow:
    """store most rencet T (steps) observations to generate sequences"""
    def __init__(self, max_len):
        self.max_len = max_len                                      # sequence_length
        self.self_states = deque(maxlen=max_len)                    # (num_units, dim)
        self.full_maps = deque(maxlen=max_len)                      # (num_channels, height, width)

    def push(self, self_state, full_map):
        self.self_states.append(self_state)
        self.full_maps.append(full_map)

    def get_sequence(self):
        """
        Returns:
            states: (max_len, num_units, dim)
            maps: (max_len, num_channels, height, width)
            valid_len: (not used)
        """
        T = len(self.self_states) # number of timesteps that's not empty

        if T == 0:
            # (edge case) start game: store a all zero dummy sequence
            state_dummy = np.zeros((self.max_len, Config.num_units, Config.self_state_dim), dtype=np.float32)
            map_dummy = np.zeros((self.max_len, Config.map_channels, Config.map_size, Config.map_size), dtype=np.float32)
            return state_dummy, map_dummy, 0

        states = np.stack(self.self_states)                         # (max_len, num_units, dim)
        maps = np.stack(self.full_maps)                             # (max_len, num_channels, height, width)
        
        pad_len = self.max_len - T
        if pad_len > 0:
            # use zeros for paddings
            state_pad = np.zeros((pad_len, *states.shape[1:]), dtype=states.dtype)
            map_pad = np.zeros((pad_len, *maps.shape[1:]), dtype=maps.dtype)

            states = np.concatenate([state_pad, states], axis=0)    # (max_len, num_units, dim)
            maps = np.concatenate([map_pad, maps], axis=0)          # (max_len, num_channels, height, width)

        return states, maps, T                                      # timesteps used

    def is_ready(self):
        return len(self.self_states) >= self.max_len

    def reset(self):
        self.self_states.clear()
        self.full_maps.clear()


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

        self.total_num_episodes = config.num_episodes
        self.optimizer = optim.Adam(self.model.parameters(), lr=config.lr)
        self.gamma = config.gamma
        self.lam = config.lam
        self.clip_eps = config.clip_eps
        self.memory = []
        self.num_units = config.num_units
        self.temporal_windows = [TemporalWindow(max_len=config.sequence_length) for _ in range(config.num_envs)]
        self.kl_beta = config.kl_beta
        self.kl_target = config.kl_target
        self.kl_update_rate = config.kl_update_rate
        self.batch_size = config.batch_size
        self.epochs = config.epochs
        self.min_kl_beta = config.min_kl_beta
        self.max_kl_beta = config.max_kl_beta

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
                            # relatively weak block, add some scores
                            score += 3
                        else:        
                            # needs 3 bombs, only add a little score
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
        return None # shouldn't be triggered at all
        

    def select_actions(self, self_states, full_map, alive_mask, current_bomb_infos, current_bomb_count, unit_ids, current_state, env_id=0):
        if isinstance(self_states, np.ndarray) and self_states.ndim == 3 and self_states.shape[0] == 1:
            self_states = np.squeeze(self_states, axis=0)
        if isinstance(full_map, np.ndarray) and full_map.ndim == 4 and full_map.shape[0] == 1:
            full_map = np.squeeze(full_map, axis=0)

        # only push this state
        self.temporal_windows[env_id].push(self_states, full_map)

        # seq_states, seq_maps = self.temporal_windows[env_id].get_sequence()
        # 添加 batch 维度 → (1, T, N, D), (1, T, C, H, W)
        # if seq_states.ndim == 3:  # (T, N, D)
        #     seq_states = seq_states.unsqueeze(0)  # → (1, T, N, D)

        # if seq_maps.ndim == 4:    # (T, C, H, W)
        #     seq_maps = seq_maps.unsqueeze(0)      # → (1, T, C, H, W)

        # 取序列 → (T, N, D), (T, C, H, W)
        seq_states, seq_maps, _ = self.temporal_windows[env_id].get_sequence()

        # 添加 batch 维度 → (1, T, N, D), (1, T, C, H, W)
        seq_states = torch.tensor(seq_states, dtype=torch.float32).unsqueeze(0).to(self.device)
        seq_maps = torch.tensor(seq_maps, dtype=torch.float32).unsqueeze(0).to(self.device)

        # valid_len_tensor = torch.tensor([valid_len], device=self.device)  # shape: (1,)

        # print("seq_states.shape:", seq_states.shape)
        # print("seq_maps.shape:", seq_maps.shape)

        with torch.no_grad():
            self.model.eval()
            logits, values = self.model(seq_states, seq_maps)  # logits: (1, T, N, A), values: (1, T)

        actions = []
        log_probs = []
        detonate_targets = []

        latest_logits = logits[:, -1]  # (1, N, A)
        latest_values = values[:, -1]  # (1,)

        for i in range(self.model.num_units):
            unit_logits = latest_logits[0, i]
            mask = torch.ones_like(unit_logits, dtype=torch.bool)

            if current_bomb_count >= 3:
                mask[4] = False

            candidate_target = self.select_detonate_target(unit_ids[i], current_bomb_infos, current_state)
            if candidate_target is None:
                mask[5] = False

            masked_logits = unit_logits.clone()
            masked_logits[~mask] = -1e10

            probs = torch.softmax(masked_logits, dim=-1)
            dist = torch.distributions.Categorical(probs)

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

        actions = torch.stack(actions).unsqueeze(0)
        log_probs = torch.stack(log_probs).unsqueeze(0)
        
        return actions.cpu().numpy(), log_probs.cpu().numpy(), latest_values.squeeze().cpu().numpy(), detonate_targets, latest_logits[0].cpu().numpy()

    def update_from_buffer(self, episode_buffer, current_episode, epochs, batch_size):
        if not episode_buffer:
            return
        
        self.memory.clear()

        for sequence in episode_buffer:
            if not sequence:
                continue

            # sequence 是一个 list of (state, map, action, log_prob, reward, value, done)
            rewards = np.array([step[4] for step in sequence])
            values_arr = np.array([step[5] for step in sequence])
            dones = np.array([step[6] for step in sequence])

            advantages, returns = self.compute_gae(rewards, values_arr, dones)

            new_sequence = []
            for (state, map_data, action, log_prob, reward, value, done, old_logits), adv, ret in zip(sequence, advantages, returns):
                new_sequence.append((state, map_data, action, log_prob, ret, adv, old_logits))

            self.memory.append(new_sequence)

        self.update(current_episode, epochs, batch_size)

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

    def vectorized_ppo_update(self, episode_idx, model, optimizer, batch_data, device, clip_eps):
        self_states, full_maps, actions, old_log_probs, returns, advantages, old_logits = batch_data

        # Tensor
        self_states = torch.tensor(self_states, dtype=torch.float32).to(device)        # (B, T, N, D)
        full_maps = torch.tensor(full_maps, dtype=torch.float32).to(device)            # (B, T, C, H, W)
        actions = torch.tensor(actions, dtype=torch.long).to(device)                   # (B, T, N)
        old_log_probs = torch.tensor(old_log_probs, dtype=torch.float32).to(device)    # (B, T, N)
        returns = torch.tensor(returns, dtype=torch.float32).to(device)                # (B, T)
        advantages = torch.tensor(advantages, dtype=torch.float32).to(device)          # (B, T)
        old_logits = torch.tensor(old_logits, dtype=torch.float32).to(device)          # (B, T, N, A)

        # standard advantage
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # forward
        logits, values = model(self_states, full_maps)  # logits: (B, T, N, A), values: (B, T)
        B, T, N, A = logits.shape

        # reshape → 扁平结构用于 loss 计算
        flat_logits = logits.view(B * T * N, A)
        flat_old_logits = old_logits.view(B * T * N, A)
        flat_actions = actions.view(-1)
        flat_old_log_probs = old_log_probs.view(-1)
        flat_advantages = advantages.unsqueeze(-1).expand(-1, -1, N).contiguous().view(-1)  # (B*T*N,)

        # 构造新旧策略分布
        with torch.no_grad():
            old_dist = torch.distributions.Categorical(logits=flat_old_logits)
        new_dist = torch.distributions.Categorical(logits=flat_logits)

        flat_log_probs = new_dist.log_prob(flat_actions)
        ratio = torch.exp(flat_log_probs - flat_old_log_probs)

        # PPO Clip Loss
        surr1 = ratio * flat_advantages
        surr2 = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * flat_advantages
        policy_loss = -torch.min(surr1, surr2).mean()

        # Value Loss
        value_loss = F.mse_loss(values, returns)

        # KL 散度: + self.kl_beta * kl 
        kl = torch.distributions.kl_divergence(old_dist, new_dist).mean()

        # 熵奖励: - entropy_coeff * entropy
        entropy_coeff = max(0.001, 0.01 * (1 - episode_idx / self.total_num_episodes))
        entropy = new_dist.entropy().mean()
        
        # 总 Loss（含 KL 正则项 & 熵奖励）
        loss = policy_loss + 0.5 * value_loss + self.kl_beta * kl - entropy_coeff * entropy

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        return policy_loss.item(), value_loss.item(), loss.item(), kl.item(), entropy.item()


    # min_kl_beta防止为0，数值可调; max_kl_beta可选上限限制
    def update(self, current_episode, epochs=5, batch_size=32):
        if not self.memory:
            return

        # memory 是 List[sequence]，每个 sequence 是 List[Tuple]
        batched_states = []
        batched_maps = []
        batched_actions = []
        batched_old_log_probs = []
        batched_returns = []
        batched_advantages = []
        batched_old_logits = []

        for seq in self.memory:
            s_seq, m_seq, a_seq, old_lp_seq, ret_seq, adv_seq, old_logits_seq = zip(*seq)

            # ✅ (T, C, H, W) not (T, 1, C, H, W)
            m_seq = [np.squeeze(m, axis=0) if m.ndim == 4 else m for m in m_seq]
            
            batched_states.append(np.array(s_seq))              # (T, N, D)
            batched_maps.append(np.array(m_seq))                # (T, C, H, W)
            batched_actions.append(np.array(a_seq))             # (T, N)
            batched_old_log_probs.append(np.array(old_lp_seq))  # (T, N)
            batched_returns.append(np.array(ret_seq))           # (T,)
            batched_advantages.append(np.array(adv_seq))        # (T,)
            batched_old_logits.append(np.array(old_logits_seq))

        # 拼成 (B, T, ...)
        batched_states = np.stack(batched_states)               # (B, T, N, D)
        batched_maps = np.stack(batched_maps)                   # (B, T, C, H, W)
        batched_actions = np.stack(batched_actions)             # (B, T, N)
        batched_old_log_probs = np.stack(batched_old_log_probs)
        batched_returns = np.stack(batched_returns)             # (B, T)
        batched_advantages = np.stack(batched_advantages)       # (B, T)
        batched_old_logits = np.stack(batched_old_logits)       # (B, T, N, A)

        total_policy_loss = 0
        total_value_loss = 0
        total_loss = 0
        total_kl = 0
        total_entropy = 0

        B = batched_states.shape[0]
        indices = np.arange(B)

        print("update batch_size:", B)
        print("batched_states shape:", batched_states.shape)
        print("batched_maps shape:", batched_maps.shape)
        print("batched_actions shape:", batched_actions.shape)
        print("batched_old_log_probs shape:", batched_old_log_probs.shape)
        print("batched_returns shape:", batched_returns.shape)
        print("batched_advantages shape:", batched_advantages.shape)
        print("batched_old_logits shape:", batched_old_logits.shape)

        for _ in range(epochs):
            np.random.shuffle(indices)
            for start in range(0, B, batch_size):
                end = start + batch_size
                batch_idx = indices[start:end]

                mini_batch = (
                    batched_states[batch_idx],
                    batched_maps[batch_idx],
                    batched_actions[batch_idx],
                    batched_old_log_probs[batch_idx],
                    batched_returns[batch_idx],
                    batched_advantages[batch_idx],
                    batched_old_logits[batch_idx]
                )

                policy_loss, value_loss, loss, kl, entropy = self.vectorized_ppo_update(
                    episode_idx=current_episode,
                    model=self.model,
                    optimizer=self.optimizer,
                    batch_data=mini_batch,
                    device=self.device,
                    clip_eps=self.clip_eps,
                )

                if kl > self.kl_target * 1.5:
                    self.kl_beta = min(self.kl_beta * self.kl_update_rate, self.max_kl_beta)
                elif kl < self.kl_target * 0.5:
                    self.kl_beta = max(self.kl_beta / self.kl_update_rate, self.min_kl_beta)

                total_policy_loss += policy_loss
                total_value_loss += value_loss
                total_loss += loss
                total_kl += kl
                total_entropy += entropy

        num_batches = (B // batch_size) * epochs
        wandb.log({
            "train/policy_loss": total_policy_loss / num_batches,
            "train/value_loss": total_value_loss / num_batches,
            "train/total_loss": total_loss / num_batches,
            "train/kl": total_kl / num_batches,
            "train/entropy": total_entropy / num_batches,
            "train/kl_beta": self.kl_beta,
            "train/episode": current_episode,
        }, step=current_episode)

        print(f"Update stats - Policy loss: {total_policy_loss / num_batches:.4f}, Value loss: {total_value_loss / num_batches:.4f}, Total loss: {total_loss / num_batches:.4f}")

        self.memory.clear()
