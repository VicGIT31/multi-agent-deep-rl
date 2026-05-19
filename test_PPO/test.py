import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
import gymnasium as gym
import numpy as np
import pickle
import matplotlib.pyplot as plt
import matplotlib

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(ActorCritic, self).__init__()
        self.feature_layer = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU()
        )
        self.actor = nn.Linear(128, action_dim)
        self.critic = nn.Linear(128, 1)

    def forward(self, state):
        x = self.feature_layer(state)
        probs = F.softmax(self.actor(x), dim=-1)
        value = self.critic(x)
        return probs, value


class PPOAgent:
    def __init__(self, state_dim, action_dim):
        self.gamma = 0.99
        self.lmbda = 0.95
        self.eps_clip = 0.2
        self.k_epochs = 10
        self.entropy_coef = 0.01

        self.policy = ActorCritic(state_dim, action_dim).to(device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=3e-4)
        self.mse_loss = nn.MSELoss()

    def select_action(self, state):
        with torch.no_grad():
            state = torch.FloatTensor(state).to(device)
            probs, value = self.policy(state)
            dist = Categorical(probs)
            action = dist.sample()
            return action.item(), dist.log_prob(action).item(), value.item()

    def update(self, memory):
        states = torch.FloatTensor(np.array(memory['states'])).to(device)
        actions = torch.LongTensor(memory['actions']).to(device)
        old_logprobs = torch.FloatTensor(memory['logprobs']).to(device)
        rewards = memory['rewards']
        values = memory['values']
        masks = memory['masks']

        returns = []
        gae = 0
        for i in reversed(range(len(rewards))):
            next_value = values[i+1] if i+1 < len(values) else 0
            delta = rewards[i] + self.gamma * next_value * masks[i] - values[i]
            gae = delta + self.gamma * self.lmbda * masks[i] * gae
            returns.insert(0, gae + values[i])

        returns = torch.FloatTensor(returns).to(device)
        advantages = returns - torch.FloatTensor(values).to(device)
        if len(advantages) > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        for _ in range(self.k_epochs):
            new_probs, state_values = self.policy(states)
            dist = Categorical(new_probs)
            new_logprobs = dist.log_prob(actions)
            entropy = dist.entropy().mean()

            ratios = torch.exp(new_logprobs - old_logprobs)
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages

            actor_loss = -torch.min(surr1, surr2).mean()
            critic_loss = self.mse_loss(state_values.squeeze(), returns)
            loss = actor_loss + 0.5 * critic_loss - self.entropy_coef * entropy

            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.policy.parameters(), 0.5)
            self.optimizer.step()


class TrainingPlot:
    def __init__(self):
        matplotlib.use("TkAgg")
        plt.ion()
        self.fig, self.ax = plt.subplots()
        self.ax.set_xlabel("Épisode")
        self.ax.set_ylabel("Score")
        self.ax.set_title("Entraînement PPO - LunarLander")
        self.line_score, = self.ax.plot([], [], alpha=0.4, color="steelblue", label="Score")
        self.line_avg, = self.ax.plot([], [], color="orange", linewidth=2, label="Moyenne (100 ep)")
        self.ax.axhline(y=200, color="green", linestyle="--", linewidth=1, label="Objectif (200)")
        self.ax.legend()

    def update(self, ep, all_scores, all_avgs, score, avg_score):
        episodes = list(range(len(all_scores)))
        self.line_score.set_data(episodes, all_scores)
        self.line_avg.set_data(episodes, all_avgs)
        self.ax.relim()
        self.ax.autoscale_view()
        self.ax.set_title(f"Épisode {ep} | Score: {score:.1f} | Moy: {avg_score:.1f}")
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

    def finish(self, avg_score):
        self.ax.set_title(f"Objectif atteint ! Moyenne: {avg_score:.1f} sur 100 épisodes")
        plt.ioff()
        plt.show()


def train():
    env = gym.make("LunarLander-v3")
    env = gym.wrappers.NormalizeObservation(env)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    agent = PPOAgent(state_dim, action_dim)

    update_timestep = 2048
    timestep = 0
    max_episodes = 5000
    scores_window = []
    all_scores = []
    all_avgs = []

    plot = TrainingPlot()

    memory = {'states': [], 'actions': [], 'logprobs': [], 'rewards': [], 'values': [], 'masks': []}

    for ep in range(max_episodes):
        state, _ = env.reset()
        score = 0

        for _ in range(1000):
            timestep += 1
            action, logprob, val = agent.select_action(state)
            next_state, reward, done, trunc, _ = env.step(action)

            memory['states'].append(state)
            memory['actions'].append(action)
            memory['logprobs'].append(logprob)
            memory['rewards'].append(reward)
            memory['values'].append(val)
            memory['masks'].append(1 - (done or trunc))

            state = next_state
            score += reward

            if timestep >= update_timestep:
                agent.update(memory)
                for key in memory: memory[key] = []
                timestep = 0

            if done or trunc:
                break

        scores_window.append(score)
        if len(scores_window) > 100:
            scores_window.pop(0)
        avg_score = np.mean(scores_window)
        all_scores.append(score)
        all_avgs.append(avg_score)

        if ep % 10 == 0:
            torch.save(agent.policy.state_dict(), "ppo_lunar_current.pth")
            plot.update(ep, all_scores, all_avgs, score, avg_score)

        if avg_score >= 200 and len(scores_window) >= 100:
            torch.save(agent.policy.state_dict(), "ppo_lunar_model_250.pth")

            stats = {"mean": env.obs_rms.mean, "var": env.obs_rms.var}
            with open("env_stats.pkl", "wb") as f:
                pickle.dump(stats, f)

            plot.finish(avg_score)
            break


if __name__ == "__main__":
    train()
