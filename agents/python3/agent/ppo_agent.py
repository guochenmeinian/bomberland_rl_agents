# agent/ppo_agent.py
from encodings import normalize_encoding
import torch
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from model.ppo_model import PPOModel
import random
import wandb
from config import Config 

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
        """
        Args:
            self_states: (N, D)               # 每个 unit 的自状态
            full_map:    (C, H, W)            # 当前帧地图信息
            alive_mask:  (N,)                 # 存活 unit mask（未用）
            current_bomb_infos: List[(x,y,owner_id)]
            current_bomb_count: int
            unit_ids: List[str]
            current_state: dict

        Returns:
            actions:        (1, N)
            log_probs:      (1, N)
            value:          float
            detonate_targets: List[Optional[(x, y)]]
            logits:         (N, A)
        """

        # 🟦 Step 1: preprocess输入 → 添加 batch 维度
        if isinstance(self_states, np.ndarray):
            self_states = torch.tensor(self_states, dtype=torch.float32)
        if isinstance(full_map, np.ndarray):
            full_map = torch.tensor(full_map, dtype=torch.float32)

        # only add batch dim if not already batched
        if self_states.ndim == 2:  # (N, D)
            self_states = self_states.unsqueeze(0)  # → (1, N, D)
        if full_map.ndim == 3:  # (C, H, W)
            full_map = full_map.unsqueeze(0)        # → (1, C, H, W)

        # 🟦 Step 2: forward 模型，获得 logits 和 value
        with torch.no_grad():
            self.model.eval()
            logits, values = self.model(self_states, full_map)  # logits: (1, N, A), values: (1,)

        logits = logits[0]   # (N, A)
        value = values[0]    # scalar → float

        actions = []
        log_probs = []
        detonate_targets = []

        # 🟦 Step 3: 遍历每个 unit，按合法动作采样 action
        for i in range(self.model.num_units):
            unit_logits = logits[i]         # (A,)
            mask = torch.ones_like(unit_logits, dtype=torch.bool)

            # 限制最多放3颗炸弹
            if current_bomb_count >= 3:
                mask[4] = False

            # 判断是否可引爆
            candidate_target = self.select_detonate_target(unit_ids[i], current_bomb_infos, current_state)
            if candidate_target is None:
                mask[5] = False

            # 屏蔽非法动作
            masked_logits = unit_logits.clone()
            masked_logits[~mask] = -1e10

            # debug logging
            if torch.isnan(masked_logits).any():
                print(f"[NaN Warning] masked_logits unit {i}:", masked_logits.tolist())

            # fallback if all masked
            if (~mask).all():
                print(f"[Fallback] unit {i} has no valid action, default to stay")
                action = torch.tensor(6, device=unit_logits.device)
                log_prob = torch.tensor(0.0, device=unit_logits.device)
            else:
                probs = torch.softmax(masked_logits, dim=-1)
                if torch.isnan(probs).any() or probs.sum().item() == 0:
                    print(f"[Warning] invalid probs for unit {i}")
                    print("  unit_logits:", unit_logits.tolist())
                    print("  mask:", mask.tolist())
                    print("  masked_logits:", masked_logits.tolist())
                    probs = torch.ones_like(probs) / probs.shape[0]  # uniform fallback

                dist = torch.distributions.Categorical(probs)
                action = dist.sample()
                log_prob = dist.log_prob(action)


            # 炸弹计数更新
            if action.item() == 4:
                current_bomb_count += 1

            # detonate 目标记录
            if action.item() == 5:
                detonate_targets.append(candidate_target)
            else:
                detonate_targets.append(None)

            actions.append(action)
            log_probs.append(log_prob)

        # 🟦 Step 4: 返回动作结果（添加 batch 维度）
        actions = torch.stack(actions).unsqueeze(0)     # (1, N)
        log_probs = torch.stack(log_probs).unsqueeze(0) # (1, N)

        return (
            actions.cpu().numpy(),         # (1, N)
            log_probs.cpu().numpy(),       # (1, N)
            value.item(),                  # scalar
            detonate_targets,              # list[N]
            logits.cpu().numpy()           # (N, A)
        )

    def update_from_buffer(self, episode_buffer, current_episode, epochs, batch_size):
        if not episode_buffer:
            return

        self.memory.clear()

        rewards = np.array([step[4] for step in episode_buffer])
        values = np.array([step[5] for step in episode_buffer])
        dones = np.array([step[6] for step in episode_buffer])

        rewards = np.clip(rewards, -10, 10)  # 防止 value loss 爆炸

        advantages, returns = self.compute_gae(rewards, values, dones)
        wandb.log({
            "train/advantage_mean": np.mean(advantages),
            "train/advantage_std": np.std(advantages)
        }, step=current_episode)


        for i, (state, map_obs, action, log_prob, _, _, _, old_logits) in enumerate(episode_buffer):
            self.memory.append((
                state, map_obs, action, log_prob,
                returns[i], advantages[i], old_logits
            ))

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

        self_states = torch.tensor(self_states, dtype=torch.float32).to(device)      # (B, N, D)
        full_maps = torch.tensor(full_maps, dtype=torch.float32).to(device)          # (B, C, H, W)
        actions = torch.tensor(actions, dtype=torch.long).to(device)                 # (B, N)
        old_log_probs = torch.tensor(old_log_probs, dtype=torch.float32).to(device)  # (B, N)
        returns = torch.tensor(returns, dtype=torch.float32).to(device)              # (B,)
        advantages = torch.tensor(advantages, dtype=torch.float32).to(device)        # (B,)
        old_logits = torch.tensor(old_logits, dtype=torch.float32).to(device)        # (B, N, A)

        B, N = actions.shape
        advantages = advantages.view(B, 1).expand(B, N)  # (B, N)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        flat_advantages = advantages.reshape(-1)         # (B*N,)

        logits, values = model(self_states, full_maps)  # logits: (B, N, A), values: (B,)
        B, N, A = logits.shape

        flat_logits = logits.view(B * N, A)
        if torch.isnan(flat_logits).any():
            print("NaN detected in logits!")
            print(flat_logits)
            raise ValueError("NaN in logits")

        if flat_logits.abs().max().item() > 1e4:
            print(f"[Warning] logits too large: max={flat_logits.abs().max().item()}")

        flat_old_logits = old_logits.view(B * N, A)
        flat_actions = actions.view(-1)
        flat_old_log_probs = old_log_probs.view(-1)

        with torch.no_grad():
            old_dist = torch.distributions.Categorical(logits=flat_old_logits)
        new_dist = torch.distributions.Categorical(logits=flat_logits)

        flat_log_probs = new_dist.log_prob(flat_actions)
        ratio = torch.exp(flat_log_probs - flat_old_log_probs)

        surr1 = ratio * flat_advantages
        surr2 = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * flat_advantages
        policy_loss = -torch.min(surr1, surr2).mean()

        value_loss = F.mse_loss(values, returns)

        kl = torch.distributions.kl_divergence(old_dist, new_dist).mean()
        # entropy_coeff = max(0.003, 0.01 * (1 - episode_idx / self.total_num_episodes))
        entropy_coeff = 0.1
        entropy = new_dist.entropy().mean()

        loss = policy_loss + 0.5 * value_loss + self.kl_beta * kl - entropy_coeff * entropy

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0) # prevent explode
        optimizer.step()

        return policy_loss.item(), value_loss.item(), loss.item(), kl.item(), entropy.item()



    # min_kl_beta防止为0，数值可调; max_kl_beta可选上限限制
    def update(self, current_episode, epochs=5, batch_size=32):
        if not self.memory:
            return

        # memory 是 list of step tuple
        batched_states, batched_maps, batched_actions = [], [], []
        batched_old_log_probs, batched_returns, batched_advantages, batched_old_logits = [], [], [], []

        for item in self.memory:
            s, m, a, logp, ret, adv, old_logits = item
            if m.ndim == 4:
                m = np.squeeze(m, axis=0)  # 防止 map 是 (1, C, H, W)
            batched_states.append(s)
            batched_maps.append(m)
            batched_actions.append(a)
            batched_old_log_probs.append(logp)
            batched_returns.append(ret)
            batched_advantages.append(adv)
            batched_old_logits.append(old_logits)

        # 拼接为 (B, ...)
        batched_states = np.stack(batched_states)
        batched_maps = np.stack(batched_maps)
        batched_actions = np.stack(batched_actions)
        batched_old_log_probs = np.stack(batched_old_log_probs)
        batched_returns = np.stack(batched_returns)
        batched_advantages = np.stack(batched_advantages)
        batched_old_logits = np.stack(batched_old_logits)

        total_policy_loss = 0
        total_value_loss = 0
        total_loss = 0
        total_kl = 0
        total_entropy = 0

        B = batched_states.shape[0]
        indices = np.arange(B)

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

                # KL 更新
                if kl > self.kl_target * 1.5:
                    self.kl_beta = min(self.kl_beta * self.kl_update_rate, self.max_kl_beta)
                elif kl < self.kl_target * 0.5:
                    self.kl_beta = max(self.kl_beta / self.kl_update_rate, self.min_kl_beta)

                total_policy_loss += policy_loss
                total_value_loss += value_loss
                total_loss += loss
                total_kl += kl
                total_entropy += entropy

        num_batches = max(1, (B // batch_size) * epochs)

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
