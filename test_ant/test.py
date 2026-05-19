# ── 1. Imports ────────────────────────────────────────────────────────────────
import gc
import os
from itertools import count

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import gymnasium as gym
import matplotlib
matplotlib.use("Agg")  # pas de live plot pendant l'entraînement (économie RAM)
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal

torch.set_num_threads(1)

# ── 2. Hyperparameters ────────────────────────────────────────────────────────
HIDDEN           = 128
BATCH_SIZE       = 128
GAMMA            = 0.99
TAU              = 0.005
LR               = 3e-4
ALPHA            = 0.2
START_STEPS      = 1000
UPDATE_EVERY     = 2
MEMORY_CAPACITY  = 25_000
NUM_EPISODES     = 400
DEMO_EPISODES    = 10
MAX_EP_STEPS     = 500
PLOT_EVERY       = 25

# ── 3. Device ─────────────────────────────────────────────────────────────────
device = torch.device(
    "cuda" if torch.cuda.is_available() else
    "mps"  if torch.backends.mps.is_available() else
    "cpu"
)
print(f"device: {device}")

# ── 4. Replay buffer (numpy, préalloué) ───────────────────────────────────────
class ReplayMemory(object):

    def __init__(self, capacity, n_observations, n_actions):
        self.capacity = capacity
        self.idx = 0
        self.size = 0
        self.states      = np.zeros((capacity, n_observations), dtype=np.float32)
        self.actions     = np.zeros((capacity, n_actions),      dtype=np.float32)
        self.next_states = np.zeros((capacity, n_observations), dtype=np.float32)
        self.rewards     = np.zeros((capacity,),                dtype=np.float32)
        self.dones       = np.zeros((capacity,),                dtype=np.float32)

    def push(self, s, a, ns, r, d):
        i = self.idx
        self.states[i]      = s
        self.actions[i]     = a
        self.next_states[i] = ns
        self.rewards[i]     = r
        self.dones[i]       = d
        self.idx  = (self.idx + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size):
        idx = np.random.randint(0, self.size, size=batch_size)
        return (
            torch.from_numpy(self.states[idx]),
            torch.from_numpy(self.actions[idx]),
            torch.from_numpy(self.next_states[idx]),
            torch.from_numpy(self.rewards[idx]),
            torch.from_numpy(self.dones[idx]),
        )

    def __len__(self):
        return self.size


# ── 5. Neural networks ────────────────────────────────────────────────────────
LOG_STD_MIN = -20
LOG_STD_MAX = 2


class Actor(nn.Module):

    def __init__(self, n_observations, n_actions, action_scale, action_bias):
        super(Actor, self).__init__()
        self.layer1  = nn.Linear(n_observations, HIDDEN)
        self.layer2  = nn.Linear(HIDDEN, HIDDEN)
        self.mean    = nn.Linear(HIDDEN, n_actions)
        self.log_std = nn.Linear(HIDDEN, n_actions)
        self.register_buffer("action_scale", torch.as_tensor(action_scale, dtype=torch.float32))
        self.register_buffer("action_bias",  torch.as_tensor(action_bias,  dtype=torch.float32))

    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        mean    = self.mean(x)
        log_std = self.log_std(x).clamp(LOG_STD_MIN, LOG_STD_MAX)
        return mean, log_std

    def sample(self, x):
        mean, log_std = self.forward(x)
        std = log_std.exp()
        normal = Normal(mean, std)
        z = normal.rsample()
        y = torch.tanh(z)
        action = y * self.action_scale + self.action_bias
        log_prob = normal.log_prob(z) - torch.log(self.action_scale * (1 - y.pow(2)) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)
        return action, log_prob


class Critic(nn.Module):

    def __init__(self, n_observations, n_actions):
        super(Critic, self).__init__()
        self.layer1 = nn.Linear(n_observations + n_actions, HIDDEN)
        self.layer2 = nn.Linear(HIDDEN, HIDDEN)
        self.layer3 = nn.Linear(HIDDEN, 1)

    def forward(self, state, action):
        x = torch.cat([state, action], dim=1)
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.layer3(x)


# ── 6. Environment & training setup ──────────────────────────────────────────
env = gym.make("Ant-v5", max_episode_steps=MAX_EP_STEPS)

state, info    = env.reset()
n_actions      = env.action_space.shape[0]
n_observations = env.observation_space.shape[0]
action_scale   = (env.action_space.high - env.action_space.low) / 2.0
action_bias    = (env.action_space.high + env.action_space.low) / 2.0

actor    = Actor(n_observations, n_actions, action_scale, action_bias).to(device)
critic_1 = Critic(n_observations, n_actions).to(device)
critic_2 = Critic(n_observations, n_actions).to(device)
target_1 = Critic(n_observations, n_actions).to(device)
target_2 = Critic(n_observations, n_actions).to(device)
target_1.load_state_dict(critic_1.state_dict())
target_2.load_state_dict(critic_2.state_dict())
for p in target_1.parameters(): p.requires_grad = False
for p in target_2.parameters(): p.requires_grad = False

actor_optimizer    = optim.Adam(actor.parameters(),    lr=LR)
critic_1_optimizer = optim.Adam(critic_1.parameters(), lr=LR)
critic_2_optimizer = optim.Adam(critic_2.parameters(), lr=LR)

memory = ReplayMemory(MEMORY_CAPACITY, n_observations, n_actions)

steps_done      = 0
episode_rewards = []
episode_lengths = []


# ── 7. Training functions ─────────────────────────────────────────────────────
def select_action(state_np):
    global steps_done
    steps_done += 1
    if steps_done < START_STEPS:
        return env.action_space.sample()
    with torch.no_grad():
        s = torch.from_numpy(state_np).float().unsqueeze(0)
        action, _ = actor.sample(s)
        return action.numpy().flatten()


def soft_update(target, source):
    with torch.no_grad():
        for t_param, s_param in zip(target.parameters(), source.parameters()):
            t_param.data.mul_(1 - TAU)
            t_param.data.add_(s_param.data, alpha=TAU)


def optimize_model():
    if len(memory) < BATCH_SIZE:
        return
    state_b, action_b, next_state_b, reward_b, done_b = memory.sample(BATCH_SIZE)

    with torch.no_grad():
        next_action, next_log_prob = actor.sample(next_state_b)
        q1 = target_1(next_state_b, next_action)
        q2 = target_2(next_state_b, next_action)
        next_q = torch.min(q1, q2) - ALPHA * next_log_prob
        target_q = reward_b.unsqueeze(1) + GAMMA * (1.0 - done_b.unsqueeze(1)) * next_q

    current_q1 = critic_1(state_b, action_b)
    current_q2 = critic_2(state_b, action_b)

    loss_q1 = F.smooth_l1_loss(current_q1, target_q)
    loss_q2 = F.smooth_l1_loss(current_q2, target_q)

    critic_1_optimizer.zero_grad(set_to_none=True)
    loss_q1.backward()
    torch.nn.utils.clip_grad_value_(critic_1.parameters(), 100)
    critic_1_optimizer.step()

    critic_2_optimizer.zero_grad(set_to_none=True)
    loss_q2.backward()
    torch.nn.utils.clip_grad_value_(critic_2.parameters(), 100)
    critic_2_optimizer.step()

    new_action, log_prob = actor.sample(state_b)
    q1_new = critic_1(state_b, new_action)
    q2_new = critic_2(state_b, new_action)
    actor_loss = (ALPHA * log_prob - torch.min(q1_new, q2_new)).mean()

    actor_optimizer.zero_grad(set_to_none=True)
    actor_loss.backward()
    torch.nn.utils.clip_grad_value_(actor.parameters(), 100)
    actor_optimizer.step()

    soft_update(target_1, critic_1)
    soft_update(target_2, critic_2)


# ── 8. Visualisation (sauvegarde fichier seulement) ──────────────────────────
def save_plot(final=False, path="test_ant/training_curve.png"):
    fig, ax = plt.subplots(figsize=(11, 6))
    rewards = np.array(episode_rewards, dtype=np.float32)
    title = "SAC sur Ant-v5 — résultat final" if final else \
            f"SAC sur Ant-v5 — entraînement (épisode {len(rewards)}/{NUM_EPISODES})"
    ax.set_title(title, fontsize=13)
    ax.set_xlabel("Épisode")
    ax.set_ylabel("Récompense totale")
    ax.grid(True, alpha=0.3)
    ax.plot(rewards, color="#1f77b4", alpha=0.45, linewidth=0.8, label="reward par épisode")
    if len(rewards) >= 10:
        window = min(50, len(rewards))
        kernel = np.ones(window, dtype=np.float32) / window
        means = np.convolve(rewards, kernel, mode="valid")
        x = np.arange(window - 1, window - 1 + len(means))
        ax.plot(x, means, color="#d62728", linewidth=2.0,
                label=f"moyenne glissante ({window} ep)")
        if window >= 10:
            stds = np.array([rewards[i:i + window].std() for i in range(len(means))],
                            dtype=np.float32)
            ax.fill_between(x, means - stds, means + stds,
                            color="#d62728", alpha=0.15, label="± écart-type")
    if len(rewards) > 0:
        best = rewards.max()
        ax.axhline(best, color="green", linestyle="--", linewidth=0.8,
                   label=f"meilleur: {best:.1f}")
    ax.legend(fontsize=9, loc="best")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


# ── 9. Training loop ─────────────────────────────────────────────────────────
update_counter = 0

for i_episode in range(NUM_EPISODES):
    state, info = env.reset()
    total_reward = 0.0

    for t in count():
        action = select_action(state)
        observation, reward, terminated, truncated, _ = env.step(action)
        total_reward += float(reward)
        done = terminated or truncated

        memory.push(state, action, observation, reward, float(terminated))
        state = observation

        update_counter += 1
        if update_counter % UPDATE_EVERY == 0:
            optimize_model()

        if done:
            episode_rewards.append(total_reward)
            episode_lengths.append(t + 1)
            print(f"ep {i_episode + 1:4d}/{NUM_EPISODES}  steps={t + 1:4d}  "
                  f"reward={total_reward:9.2f}  buffer={len(memory):5d}")
            break

    if (i_episode + 1) % PLOT_EVERY == 0:
        save_plot(final=False)
        gc.collect()

print("Complete")
save_plot(final=True)
env.close()

# ── 10. Pygame demo — trained agent ──────────────────────────────────────────
print(f"\n=== Démo visuelle (render_mode=human) — {DEMO_EPISODES} épisodes ===")

demo_env = gym.make("Ant-v5", render_mode="human", max_episode_steps=MAX_EP_STEPS)
actor.eval()
demo_rewards = []
for ep in range(DEMO_EPISODES):
    state, _ = demo_env.reset()
    total_r, done = 0.0, False
    while not done:
        with torch.no_grad():
            s = torch.from_numpy(state).float().unsqueeze(0)
            mean, _ = actor.forward(s)
            a = (torch.tanh(mean) * actor.action_scale + actor.action_bias).numpy().flatten()
        state, r, term, trunc, _ = demo_env.step(a)
        total_r += float(r)
        done = term or trunc
    demo_rewards.append(total_r)
    print(f"  trained ep {ep + 1:2d}/{DEMO_EPISODES}  reward: {total_r:9.2f}")
demo_env.close()

print(f"\nDémo: mean={np.mean(demo_rewards):.2f}  std={np.std(demo_rewards):.2f}  "
      f"min={np.min(demo_rewards):.1f}  max={np.max(demo_rewards):.1f}")
print("Graphe sauvegardé dans test_ant/training_curve.png")
