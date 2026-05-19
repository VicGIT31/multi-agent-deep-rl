import os
import sys
import gymnasium as gym
import torch

from agent import Config, DuelingDQN, DEVICE

case_dir = sys.argv[1] if len(sys.argv) > 1 else "cases/config"
config_path = os.path.join(case_dir, "config.json")
if not os.path.exists(config_path):
    print(f"Erreur : fichier de configuration introuvable : {config_path}", file=sys.stderr)
    sys.exit(1)
cfg = Config.from_json(config_path)

env = gym.make(
    "LunarLander-v3",
    render_mode="human",
    enable_wind=cfg.enable_wind,
    wind_power=cfg.wind_power,
    turbulence_power=cfg.turbulence_power,
)
model = DuelingDQN(env.observation_space.shape[0], env.action_space.n).to(DEVICE)
model_path = os.path.join(case_dir, "model.pth")
if not os.path.exists(model_path):
    print(f"Erreur : fichier modèle introuvable : {model_path}", file=sys.stderr)
    sys.exit(1)
model.load_state_dict(torch.load(model_path, map_location=DEVICE, weights_only=True))
model.eval()

for ep in range(5):
    state, _ = env.reset()
    total_reward = 0
    done = False
    while not done:
        with torch.no_grad():
            action = model(torch.FloatTensor(state).unsqueeze(0).to(DEVICE)).argmax().item()
        state, reward, term, trunc, _ = env.step(action)
        total_reward += reward
        done = term or trunc
    print(f"Épisode {ep + 1}  score: {total_reward:.2f}")

env.close()
