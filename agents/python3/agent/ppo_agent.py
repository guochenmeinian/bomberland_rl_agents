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

    def select_actions(self, self_states, local_maps, alive_mask):
        self_states = torch.tensor(self_states[None], dtype=torch.float32).to(self.device)
        local_maps = torch.tensor(local_maps[None], dtype=torch.float32).to(self.device)

        with torch.no_grad():
            logits_list, value = self.model(self_states, local_maps)

        actions = []
        log_probs = []

        for i, logits in enumerate(logits_list):
            if alive_mask[i] == 0:
                action = torch.tensor(6, device=self.device)
                log_prob = torch.tensor(0.0, device=self.device)
            else:
                probs = torch.softmax(logits, dim=-1)
                dist = torch.distributions.Categorical(probs)
                action = dist.sample()
                log_prob = dist.log_prob(action)

            actions.append(action)
            log_probs.append(log_prob)

        actions = torch.stack(actions, dim=1)
        log_probs = torch.stack(log_probs, dim=1)

        return actions.cpu().numpy(), log_probs.cpu().numpy(), value.cpu().numpy()

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
