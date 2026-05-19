# ── 1. Imports ────────────────────────────────────────────────────────────────
import math
import random
from collections import namedtuple, deque
from itertools import count

import gymnasium as gym
import matplotlib
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# ── 2. Hyperparameters ────────────────────────────────────────────────────────
BATCH_SIZE = 128
GAMMA      = 0.99
EPS_START  = 0.9
EPS_END    = 0.05
EPS_DECAY  = 5000
TAU        = 0.005
LR         = 5e-4

# ── 3. Device ─────────────────────────────────────────────────────────────────
device = torch.device(
    "cuda" if torch.cuda.is_available() else
    "mps"  if torch.backends.mps.is_available() else
    "cpu"
)
print(device)

# ── 4. Visualisation setup ────────────────────────────────────────────────────
is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display

plt.ion()

# ── 5. Data structures ────────────────────────────────────────────────────────
Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))


class ReplayMemory(object):

    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


# ── 6. Neural network ─────────────────────────────────────────────────────────
class DQN(nn.Module):

    def __init__(self, n_observations, n_actions):
        super(DQN, self).__init__()
        self.layer1 = nn.Linear(n_observations, 128)
        self.layer2 = nn.Linear(128, 128)
        self.layer3 = nn.Linear(128, n_actions)

    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.layer3(x)


# ── 7. Environment & training setup ──────────────────────────────────────────
env = gym.make("Acrobot-v1")

state, info    = env.reset()
n_actions      = env.action_space.n
n_observations = len(state)

policy_net = DQN(n_observations, n_actions).to(device)
target_net = DQN(n_observations, n_actions).to(device)
target_net.load_state_dict(policy_net.state_dict())

optimizer = optim.AdamW(policy_net.parameters(), lr=LR, amsgrad=True)
memory    = ReplayMemory(50000)

steps_done       = 0
episode_rewards  = []

# ── 8. Training functions ─────────────────────────────────────────────────────
def select_action(state):
    global steps_done
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * \
        math.exp(-1. * steps_done / EPS_DECAY)
    steps_done += 1
    if sample > eps_threshold:
        with torch.no_grad():
            return policy_net(state).max(1).indices.view(1, 1)
    else:
        return torch.tensor([[env.action_space.sample()]], device=device, dtype=torch.long)


def optimize_model():
    if len(memory) < BATCH_SIZE:
        return
    transitions = memory.sample(BATCH_SIZE)
    batch = Transition(*zip(*transitions))

    non_final_mask = torch.tensor(
        tuple(map(lambda s: s is not None, batch.next_state)),
        device=device, dtype=torch.bool
    )
    non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])
    state_batch  = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    state_action_values = policy_net(state_batch).gather(1, action_batch)

    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    with torch.no_grad():
        next_state_values[non_final_mask] = target_net(non_final_next_states).max(1).values
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    criterion = nn.SmoothL1Loss()
    loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_value_(policy_net.parameters(), 100)
    optimizer.step()


# ── 9. Visualisation ──────────────────────────────────────────────────────────
def plot_rewards(show_result=False):
    plt.figure(1)
    rewards_t = torch.tensor(episode_rewards, dtype=torch.float)
    plt.clf()
    plt.title('Result' if show_result else 'Training...')
    plt.xlabel('Episode')
    plt.ylabel('Total reward')
    plt.plot(rewards_t.numpy(), label="reward par épisode", alpha=0.6)
    if len(rewards_t) >= 100:
        means = rewards_t.unfold(0, 100, 1).mean(1).view(-1)
        x = torch.arange(99, 99 + len(means))
        plt.plot(x.numpy(), means.numpy(), color="red", linewidth=1.5, label="moyenne 100 ep")
    plt.axhline(-100, color="green", linestyle="--", linewidth=0.8, label="seuil -100")
    plt.legend(fontsize=8, loc="lower right")
    plt.pause(0.001)
    if is_ipython:
        if not show_result:
            display.display(plt.gcf())
            display.clear_output(wait=True)
        else:
            display.display(plt.gcf())


# ── 10. Training loop ─────────────────────────────────────────────────────────
num_episodes = 600 if torch.cuda.is_available() or torch.backends.mps.is_available() else 300

for i_episode in range(num_episodes):
    state, info = env.reset()
    state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
    total_reward = 0.0

    for t in count():
        action = select_action(state)
        observation, reward, terminated, truncated, _ = env.step(action.item())
        total_reward += reward
        reward = torch.tensor([reward], device=device)
        done   = terminated or truncated

        next_state = None if terminated else \
            torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)

        memory.push(state, action, next_state, reward)
        state = next_state

        optimize_model()

        # Soft update of the target network
        target_net_state_dict = target_net.state_dict()
        policy_net_state_dict = policy_net.state_dict()
        for key in policy_net_state_dict:
            target_net_state_dict[key] = (
                policy_net_state_dict[key] * TAU
                + target_net_state_dict[key] * (1 - TAU)
            )
        target_net.load_state_dict(target_net_state_dict)

        if done:
            episode_rewards.append(total_reward)
            plot_rewards()
            break

print('Complete')
plot_rewards(show_result=True)
plt.ioff()

# ── 11. Evaluation suite ──────────────────────────────────────────────────────
# Objectif : mesurer proprement la qualité de la politique apprise en la
# comparant à une baseline aléatoire. Même env, même nombre d'épisodes,
# politique greedy (ε = 0) pour l'agent, aucune mise à jour du réseau.
import numpy as np

N_EVAL       = 50   # épisodes par politique
SOLVE_SCORE  = -100 # seuil de réussite Acrobot
N_ACTIONS    = env.action_space.n


def greedy_policy(state_np):
    with torch.no_grad():
        s = torch.tensor(state_np, dtype=torch.float32, device=device).unsqueeze(0)
        return int(policy_net(s).max(1).indices.item())


def random_policy(_state_np):
    return int(env.action_space.sample())


def evaluate(policy_fn, n_episodes, label):
    rewards, lengths, actions = [], [], []
    for ep in range(n_episodes):
        state, _ = env.reset()
        total_r, steps, done = 0.0, 0, False
        while not done:
            a = policy_fn(state)
            actions.append(a)
            state, r, term, trunc, _ = env.step(a)
            total_r += r
            steps += 1
            done = term or trunc
        rewards.append(total_r)
        lengths.append(steps)
    rewards = np.array(rewards)
    lengths = np.array(lengths)
    actions = np.array(actions)
    success = (rewards >= SOLVE_SCORE).mean() * 100
    print(f"[{label}]  mean={rewards.mean():7.2f}  std={rewards.std():5.2f}  "
          f"min={rewards.min():7.1f}  max={rewards.max():7.1f}  "
          f"median={np.median(rewards):7.1f}  "
          f"steps_moy={lengths.mean():5.1f}  success>= {SOLVE_SCORE}: {success:5.1f}%")
    return rewards, lengths, actions


print(f"\n=== Évaluation sur {N_EVAL} épisodes (ε=0, aucun apprentissage) ===")
policy_net.eval()
trained_r, trained_l, trained_a = evaluate(greedy_policy, N_EVAL, "TRAINED")
random_r, random_l, random_a = evaluate(random_policy, N_EVAL, "RANDOM ")

# Gain de l'apprentissage
delta_mean = trained_r.mean() - random_r.mean()
print(f"\nAmélioration moyenne par rapport au random : {delta_mean:+.1f} points "
      f"({(delta_mean / abs(random_r.mean())) * 100:+.1f}% d'écart relatif)")

# ── 12. Analytical plots ──────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(13, 9))
fig.suptitle("Évaluation — agent entraîné vs baseline aléatoire", fontsize=13)

# (a) Histogramme des rewards
ax = axes[0, 0]
bins = np.linspace(min(trained_r.min(), random_r.min()) - 10,
                   max(trained_r.max(), random_r.max()) + 10, 25)
ax.hist(random_r, bins=bins, alpha=0.55, label=f"random (μ={random_r.mean():.1f})", color="#d62728")
ax.hist(trained_r, bins=bins, alpha=0.7, label=f"trained (μ={trained_r.mean():.1f})", color="#2ca02c")
ax.axvline(SOLVE_SCORE, color="black", linestyle="--", linewidth=0.8)
ax.set_title("Distribution des récompenses (50 épisodes)")
ax.set_xlabel("Reward total"); ax.set_ylabel("Fréquence")
ax.legend(fontsize=8)

# (b) Boxplot pour comparer dispersion
ax = axes[0, 1]
ax.boxplot([random_r, trained_r], tick_labels=["random", "trained"], showmeans=True)
ax.axhline(SOLVE_SCORE, color="green", linestyle="--", linewidth=0.8, label=f"seuil {SOLVE_SCORE}")
ax.set_title("Dispersion des récompenses")
ax.set_ylabel("Reward total")
ax.legend(fontsize=8)

# (c) Distribution des actions (trained)
ax = axes[1, 0]
action_labels = ["-1 (gauche)", "0 (neutre)", "+1 (droite)"]
trained_counts = [np.sum(trained_a == i) / len(trained_a) * 100 for i in range(N_ACTIONS)]
random_counts  = [np.sum(random_a  == i) / len(random_a)  * 100 for i in range(N_ACTIONS)]
x = np.arange(N_ACTIONS)
ax.bar(x - 0.18, random_counts, 0.36, label="random", color="#d62728", alpha=0.7)
ax.bar(x + 0.18, trained_counts, 0.36, label="trained", color="#2ca02c", alpha=0.85)
ax.set_xticks(x); ax.set_xticklabels(action_labels)
ax.set_title("Distribution des actions choisies")
ax.set_ylabel("% des steps")
ax.legend(fontsize=8)

# (d) Longueur des épisodes (steps pour finir)
ax = axes[1, 1]
ax.hist(random_l, bins=15, alpha=0.55, label=f"random (μ={random_l.mean():.0f})", color="#d62728")
ax.hist(trained_l, bins=15, alpha=0.7, label=f"trained (μ={trained_l.mean():.0f})", color="#2ca02c")
ax.set_title("Longueur des épisodes (steps jusqu'à fin)")
ax.set_xlabel("Steps"); ax.set_ylabel("Fréquence")
ax.legend(fontsize=8)

plt.tight_layout(rect=[0, 0, 1, 0.96])

# ── 13. Pygame demo — trained vs random ───────────────────────────────────────
env.close()
print("\n=== Démo pygame ===")

print("→ 3 épisodes avec l'agent entraîné")
demo_env = gym.make("Acrobot-v1", render_mode="human")
for ep in range(3):
    state, _ = demo_env.reset()
    total_r, done = 0.0, False
    while not done:
        a = greedy_policy(state)
        state, r, term, trunc, _ = demo_env.step(a)
        total_r += r
        done = term or trunc
    print(f"  trained ep {ep + 1}  reward: {total_r:6.1f}")
demo_env.close()

print("→ 1 épisode avec une policy aléatoire (pour comparaison)")
demo_env = gym.make("Acrobot-v1", render_mode="human")
state, _ = demo_env.reset()
total_r, done = 0.0, False
while not done:
    a = int(demo_env.action_space.sample())
    state, r, term, trunc, _ = demo_env.step(a)
    total_r += r
    done = term or trunc
print(f"  random  ep 1   reward: {total_r:6.1f}")
demo_env.close()

plt.show()
