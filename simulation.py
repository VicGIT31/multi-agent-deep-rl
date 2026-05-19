import csv
import os
import subprocess
import sys
import time
import gymnasium as gym
import numpy as np
import torch
from collections import deque

from agent import Agent, Config, DEVICE

if __name__ == "__main__":
    case_dir = sys.argv[1] if len(sys.argv) > 1 else "cases/config"
    config_path = os.path.join(case_dir, "config.json")
    try:
        cfg = Config.from_json(config_path)
    except FileNotFoundError as e:
        print(f"Erreur : {e}")
        sys.exit(1)

    env = gym.make(
        "LunarLander-v3",
        enable_wind=cfg.enable_wind,
        wind_power=cfg.wind_power,
        turbulence_power=cfg.turbulence_power,
    )
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    agent = Agent(state_dim, action_dim, cfg)
    scores_window = deque(maxlen=cfg.moy_size)

    print(f"Entraînement sur {DEVICE}")

    start_time = time.time()
    os.makedirs(case_dir, exist_ok=True)
    csv_path = os.path.join(case_dir, "scores.csv")
    pth_path = os.path.join(case_dir, "model.pth")
    csv_file = open(csv_path, "w", newline="")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(["episode", "score", "moyenne", "epsilon", "memoire"])

    for episode in range(1, cfg.max_rep + 1):
        state, _ = env.reset()
        total_reward = 0
        done = False
        step_count = 0

        while not done:
            action = agent.act(state)
            next_state, reward, term, trunc, _ = env.step(action)
            if cfg.leg_engine_penalty > 0 and (state[6] == 1 or state[7] == 1) and action != 0:
                reward -= cfg.leg_engine_penalty
            if cfg.fuel_penalty > 0 and action != 0:
                reward -= cfg.fuel_penalty
            done = term or trunc

            agent.memory.push(state, action, reward, next_state, done)

            step_count += 1
            if step_count % cfg.train_every == 0:
                agent.train_step()

            state = next_state
            total_reward += reward

        agent.epsilon = max(cfg.epsilon_end, agent.epsilon * cfg.epsilon_decay)
        scores_window.append(total_reward)
        avg_score = np.mean(scores_window)
        mem_used = len(agent.memory.buffer)

        csv_writer.writerow([episode, f"{total_reward:.2f}", f"{avg_score:.2f}", f"{agent.epsilon:.4f}", mem_used])
        csv_file.flush()

        if episode % cfg.print_every == 0:
            print(f"Épisode {episode}\tMoyenne sur {cfg.moy_size} épisodes: {avg_score:7.2f}\tEps: {agent.epsilon:.2f}\tMém: {mem_used}/{cfg.memory_size}")

        if avg_score >= cfg.moy_target and len(scores_window) >= cfg.moy_size:
            print(f"\nRÉSOLU en {episode} épisodes, Moyenne: {avg_score:.2f}")
            torch.save(agent.model.state_dict(), pth_path)
            break

    csv_file.close()
    elapsed = time.time() - start_time
    subprocess.run([sys.executable, "viewer.py", csv_path, "--runtime", str(int(elapsed))])
