import random
from dataclasses import dataclass

import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F

from model import QNetwork
from replay_buffer import ReplayBuffer


@dataclass
class DQNConfig:
    buffer_size: int = 100_000
    batch_size: int = 64
    gamma: float = 0.99
    tau: float = 1e-3
    lr: float = 5e-4
    update_every: int = 4
    seed: int = 0


class Agent:
    def __init__(self, state_size: int, action_size: int, cfg: DQNConfig, device: torch.device):
        self.state_size = state_size
        self.action_size = action_size
        self.cfg = cfg
        self.device = device

        random.seed(cfg.seed)
        np.random.seed(cfg.seed)
        torch.manual_seed(cfg.seed)

        # Q-Networks
        self.qnetwork_local = QNetwork(state_size, action_size, cfg.seed).to(device)
        self.qnetwork_target = QNetwork(state_size, action_size, cfg.seed).to(device)

        # Optimizer
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=cfg.lr)

        # Replay buffer
        self.memory = ReplayBuffer(cfg.buffer_size, cfg.batch_size, cfg.seed, device)

        # Time step counter
        self.t_step = 0

    def act(self, state: np.ndarray, eps: float = 0.0) -> int:
        "Return an action for given state using epsilon-greedy policy."
        state_t = torch.from_numpy(state).float().unsqueeze(0).to(self.device)

        self.qnetwork_local.eval()
        with torch.no_grad():
            q_values = self.qnetwork_local(state_t)  # shape (1, action_size)
        self.qnetwork_local.train()

        if random.random() > eps:
            return int(torch.argmax(q_values, dim=1).item())
        else:
            return int(random.choice(np.arange(self.action_size)))



    def step(self, state, action, reward, next_state, done):
        """Save experience and learn every UPDATE_EVERY steps."""
        self.memory.add(state, action, reward, next_state, done)

        self.t_step = (self.t_step + 1) % self.cfg.update_every
        if self.t_step == 0:
            if len(self.memory) >= self.cfg.batch_size:
                experiences = self.memory.sample()
                self.learn(experiences)

    def learn(self, experiences):
        """Update value parameters using a batch of experience tuples."""
        states, actions, rewards, next_states, dones = experiences

        # --- DQN target ---
        # Q_targets_next = max_a' Q_target(next_state, a')
        Q_targets_next = self.qnetwork_target(next_states).detach().max(1)[0].unsqueeze(1)

        # If done, no bootstrap:
        Q_targets = rewards + (self.cfg.gamma * Q_targets_next * (1 - dones))

        # --- expected Q ---
        Q_expected = self.qnetwork_local(states).gather(1, actions)

        loss = F.mse_loss(Q_expected, Q_targets)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.qnetwork_local.parameters(), 1.0)
        self.optimizer.step()

        self.soft_update(self.qnetwork_local, self.qnetwork_target)

    def soft_update(self, local_model, target_model):
        """Soft update model parameters: θ_target = τ*θ_local + (1-τ)*θ_target"""
        tau = self.cfg.tau
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)

