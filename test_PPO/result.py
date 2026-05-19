import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import gymnasium as gym
import numpy as np
import pickle
import json
import os


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
        return probs, self.critic(x)


def run_test():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    env = gym.make("LunarLander-v3", render_mode="human")

    if not os.path.exists("env_stats.pkl"):
        print("Erreur: env_stats.pkl manquant !")
        return

    with open("env_stats.pkl", "rb") as f:
        stats = pickle.load(f)

    mean = stats["mean"]
    std = np.sqrt(stats["var"] + 1e-8)

    model = ActorCritic(8, 4).to(device)
    model.load_state_dict(torch.load("ppo_lunar_model_250.pth", map_location=device, weights_only=True))
    model.eval()

    with open("seeds.json", "r") as f:
        seeds = json.load(f)["eval_seeds"]

    for ep, seed in enumerate(seeds):
        state, _ = env.reset(seed=seed)
        score = 0
        done = False

        while not done:
            state = (state - mean) / std

            state_t = torch.FloatTensor(state).to(device)
            with torch.no_grad():
                probs, _ = model(state_t)
                action = torch.argmax(probs).item()

            state, reward, term, trunc, _ = env.step(action)
            score += reward
            done = term or trunc

        print(f"Épisode {ep + 1} | Seed : {seed} | Score : {score:.2f}")

    env.close()


if __name__ == "__main__":
    run_test()
