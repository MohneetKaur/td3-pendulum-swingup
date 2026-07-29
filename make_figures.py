"""Post-processing only: reads results/log.json (written by train.py) and
renders the presentation figures. Has no dependency on networks.py, td3.py,
or replay_buffer.py since everything it needs was already logged as plain
numbers during training."""
import json
import os

import numpy as np
import matplotlib.pyplot as plt


def moving_average(x, window=10):
    x = np.asarray(x, dtype=float)
    if len(x) < window:
        return x
    kernel = np.ones(window) / window
    return np.convolve(x, kernel, mode="valid")


def main(results_dir="results/main", out_dir="figures"):
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(results_dir, "log.json")) as f:
        log = json.load(f)

    # 1. Episode reward: raw + smoothed
    rewards = log["episode_reward"]
    smoothed = moving_average(rewards, window=10)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(range(1, len(rewards) + 1), rewards, alpha=0.3, label="Episode reward (raw)")
    ax.plot(
        range(10, len(rewards) + 1),
        smoothed,
        linewidth=2,
        label="10-episode moving average",
    )
    ax.set_xlabel("Episode")
    ax.set_ylabel("Cumulative reward")
    ax.set_title("TD3 Training Reward on Pendulum-v1")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "1_episode_reward.png"), dpi=150)
    plt.close(fig)

    # 1b. Deterministic eval reward (noise-free policy, more reliable signal)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    eval_ep = log["eval_episode"]
    eval_mean = np.array(log["eval_reward_mean"])
    eval_std = np.array(log["eval_reward_std"])
    ax.plot(eval_ep, eval_mean, marker="o", color="tab:green", label="Eval mean reward")
    ax.fill_between(eval_ep, eval_mean - eval_std, eval_mean + eval_std, alpha=0.2, color="tab:green")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Mean evaluation reward (5 episodes, no exploration noise)")
    ax.set_title("TD3 Deterministic Policy Evaluation over Training")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "2_eval_reward.png"), dpi=150)
    plt.close(fig)

    # 2. Critic and actor loss
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    axes[0].plot(log["critic_loss"], color="tab:red", linewidth=0.8)
    axes[0].set_title("Critic Loss (MSE, both Q-networks)")
    axes[0].set_xlabel("Gradient step")
    axes[0].set_ylabel("Loss")
    axes[0].grid(alpha=0.3)

    axes[1].plot(log["actor_loss"], color="tab:blue", linewidth=0.8)
    axes[1].set_title("Actor Loss (-Q1(s, pi(s)))")
    axes[1].set_xlabel("Update (every policy_delay steps)")
    axes[1].set_ylabel("Loss")
    axes[1].grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "3_losses.png"), dpi=150)
    plt.close(fig)

    # 3. Before/after pendulum angle trace
    theta_untrained = np.array(log["theta_untrained"])
    theta_trained = np.array(log["theta_trained"])
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)
    axes[0].plot(theta_untrained, color="gray")
    axes[0].axhline(0, color="black", linestyle="--", linewidth=1, label="upright (theta=0)")
    axes[0].set_title("Pendulum Angle - Untrained Policy")
    axes[0].set_xlabel("Timestep")
    axes[0].set_ylabel("theta (radians)")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].plot(theta_trained, color="tab:purple")
    axes[1].axhline(0, color="black", linestyle="--", linewidth=1, label="upright (theta=0)")
    axes[1].set_title("Pendulum Angle - Trained TD3 Policy")
    axes[1].set_xlabel("Timestep")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "4_angle_trace.png"), dpi=150)
    plt.close(fig)

    print(f"Figures written to {out_dir}/")


if __name__ == "__main__":
    main()
