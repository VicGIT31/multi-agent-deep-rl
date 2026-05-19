import argparse
import json
import os
import sys

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import pandas as pd


def load_config(csv_path):
    path = os.path.join(os.path.dirname(csv_path), "config.json")
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f), path
    return None, None


def build_stats(df, config, config_path, runtime=None):
    solved = df[df["moyenne"] >= 250]
    lines = [
        f"Score max     : {df['score'].max():.2f}",
        f"Meilleure moy : {df['moyenne'].max():.2f}",
        f"Score final   : {df['score'].iloc[-1]:.2f}",
        f"Moyenne finale: {df['moyenne'].iloc[-1]:.2f}",
        f"Résolu        : épisode {solved['episode'].iloc[0]}" if not solved.empty else "Résolu        : non (moyenne < 250)",
    ]
    if runtime is not None:
        h, rem = divmod(runtime, 3600)
        m, s = divmod(rem, 60)
        runtime_str = f"{h}h {m:02d}m {s:02d}s" if h else f"{m}m {s:02d}s"
        lines.append(f"Durée run     : {runtime_str}")
    if config:
        lander = config.get("lander", {})
        wind_on = lander.get("enable_wind", False)
        wind_str = f"oui  (power={lander.get('wind_power', '?')}  turb={lander.get('turbulence_power', '?')})" if wind_on else "non"
        lines += [
            f"Vent          : {wind_str}",
            f"Pénalité fuel : {lander.get('fuel_penalty', '?')}",
            f"Pénalité jambe: {lander.get('leg_engine_penalty', '?')}",
        ]
    if config_path:
        lines.append(f"Config        : {config_path}")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Visualise les résultats d'entraînement DQN")
    parser.add_argument("csv", help="Fichier CSV de résultats")
    parser.add_argument("--smooth", type=int, default=10, help="Fenêtre de lissage (défaut: 10)")
    parser.add_argument("--runtime", type=int, default=None, help="Durée d'entraînement en secondes")
    args = parser.parse_args()

    if not os.path.exists(args.csv):
        sys.exit(f"Fichier introuvable : {args.csv}")

    df = pd.read_csv(args.csv)
    config, config_path = load_config(args.csv)
    stats_text = build_stats(df, config, config_path, args.runtime)
    print(stats_text)

    fig = plt.figure(figsize=(14, 10))
    fig.suptitle(f"DQN Training — {os.path.basename(args.csv)}", fontsize=13)
    gs = gridspec.GridSpec(3, 2, figure=fig, height_ratios=[0.55, 2.5, 2.5], hspace=0.45, wspace=0.35)

    # Stats panel
    ax_stats = fig.add_subplot(gs[0, :])
    ax_stats.axis("off")
    ax_stats.text(
        0.5, 0.5, stats_text,
        transform=ax_stats.transAxes,
        fontsize=10, fontfamily="monospace",
        ha="center", va="center",
        bbox=dict(boxstyle="round,pad=0.6", facecolor="#fffbe6", edgecolor="#ccaa00", linewidth=1),
    )

    # Score par épisode
    ax1 = fig.add_subplot(gs[1, 0])
    ax1.plot(df["episode"], df["score"].rolling(args.smooth).mean(),
             label=f"score lissé (f={args.smooth})", alpha=0.8)
    ax1.plot(df["episode"], df["moyenne"], label="moyenne glissante", linewidth=1.5)
    ax1.axhline(250, color="green", linestyle="--", linewidth=0.8, label="seuil 250")
    ax1.set_title("Score par épisode")
    ax1.set_xlabel("Épisode")
    ax1.legend(fontsize=8)

    # Epsilon
    ax2 = fig.add_subplot(gs[1, 1])
    ax2.plot(df["episode"], df["epsilon"], color="orange")
    ax2.set_title("Epsilon (exploration)")
    ax2.set_xlabel("Épisode")

    # Mémoire
    ax3 = fig.add_subplot(gs[2, 0])
    ax3.plot(df["episode"], df["memoire"], color="purple")
    ax3.set_title("Mémoire utilisée")
    ax3.set_xlabel("Épisode")

    # Distribution des scores
    ax4 = fig.add_subplot(gs[2, 1])
    ax4.hist(df["score"], bins=30, color="steelblue", edgecolor="white", linewidth=0.4)
    ax4.set_title("Distribution des scores")
    ax4.set_xlabel("Score")
    ax4.set_ylabel("Fréquence")

    jpg_path = os.path.splitext(args.csv)[0] + ".jpg"
    fig.savefig(jpg_path, dpi=150, bbox_inches="tight")
    print(f"Image sauvegardée : {jpg_path}")

    plt.show()


if __name__ == "__main__":
    main()
