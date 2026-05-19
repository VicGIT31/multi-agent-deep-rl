import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import random
import numpy as np
import json
from collections import deque
from dataclasses import dataclass

@dataclass
class Config:
    gamma: float
    learning_rate: float
    memory_size: int
    batch_size: int
    epsilon_start: float
    epsilon_end: float
    epsilon_decay: float
    train_every: int
    moy_target: float
    moy_size: int
    max_rep: int
    print_every: int
    tau: float
    fuel_penalty: float
    leg_engine_penalty: float
    enable_wind: bool
    wind_power: float
    turbulence_power: float

    @classmethod
    def from_json(cls, path: str) -> "Config":
        try:
            with open(path) as f:
                data = json.load(f)
        except FileNotFoundError:
            raise FileNotFoundError(f"Fichier de configuration introuvable : '{path}'")
        eps = data["epsilon"]
        lander = data["lander"]
        return cls(
            **data["training"],
            **data["logging"],
            epsilon_start=eps["start"],
            epsilon_end=eps["end"],
            epsilon_decay=eps["decay"],
            **lander,
        )

def _get_device():
    if torch.cuda.is_available():
        try:
            torch.zeros(1).cuda()
            return torch.device("cuda")
        except RuntimeError:
            pass
    return torch.device("cpu")

DEVICE = _get_device()


class DuelingDQN(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(DuelingDQN, self).__init__()
        self.feature = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU()
        )
        self.value = nn.Linear(128, 1)
        self.advantage = nn.Linear(128, action_dim)

    def forward(self, x):
        x = self.feature(x)
        v = self.value(x)
        a = self.advantage(x)
        return v + (a - a.mean())


class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (torch.FloatTensor(np.array(states)).to(DEVICE),
                torch.LongTensor(actions).to(DEVICE),
                torch.FloatTensor(rewards).to(DEVICE),
                torch.FloatTensor(np.array(next_states)).to(DEVICE),
                torch.FloatTensor(dones).to(DEVICE))


class Agent:
    def __init__(self, state_dim, action_dim, cfg: Config):
        self.cfg = cfg
        self.model = DuelingDQN(state_dim, action_dim).to(DEVICE)
        self.target_model = DuelingDQN(state_dim, action_dim).to(DEVICE)
        self.target_model.load_state_dict(self.model.state_dict())

        self.optimizer = optim.Adam(self.model.parameters(), lr=cfg.learning_rate)
        self.memory = ReplayBuffer(cfg.memory_size)
        self.epsilon = cfg.epsilon_start
        self.action_dim = action_dim

    def act(self, state, train=True):
        if train and random.random() < self.epsilon:
            return random.randrange(self.action_dim)

        state = torch.FloatTensor(state).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            return self.model(state).argmax().item()

    def train_step(self):
        if len(self.memory.buffer) < self.cfg.batch_size:
            return

        states, actions, rewards, next_states, dones = self.memory.sample(self.cfg.batch_size)

        current_q = self.model(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            next_actions = self.model(next_states).argmax(1, keepdim=True)
            next_q = self.target_model(next_states).gather(1, next_actions).squeeze(1)
            target_q = rewards + (self.cfg.gamma * next_q * (1 - dones))

        loss = F.smooth_l1_loss(current_q, target_q)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()

        self.soft_update()

    def soft_update(self):
        for target_param, local_param in zip(self.target_model.parameters(), self.model.parameters()):
            target_param.data.copy_(self.cfg.tau * local_param.data + (1.0 - self.cfg.tau) * target_param.data)
