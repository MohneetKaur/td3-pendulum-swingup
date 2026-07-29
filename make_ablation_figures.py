"""Builds the ablation comparison, multi-seed variance, and robustness
histogram figures from the extra training runs (results_ablation_*,
results_seed1, results_seed2) alongside the main results/ run."""
import json
import os

import numpy as np
import torch
import gymnasium as gym
import matplotlib.pyplot as plt

from td3 import TD3Agent


def load(path):
    with open(os.path.join(path, "log.json")) as f:
        return json.load(f)


def ablation_comparison(out_dir="figures"):
    runs = {
        "Full TD3": "results",
        "No double-Q (single critic)": "results_ablation_no_doubleq",
        "No delayed updates": "results_ablation_no_delay",
        "No target smoothing": "results_ablation_no_smoothing",
        "Vanilla DDPG (all off)": "results_ablation_vanilla_ddpg",
    }
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:gray"]
    for (label, path), color in zip(runs.items(), colors):
        log = load(path)
        style = dict(linewidth=2.5, color=color) if label == "Full TD3" else dict(
            linewidth=1.5, color=color, linestyle="--"
        )
        ax.plot(log["eval_episode"], log["eval_reward_mean"], marker="o", markersize=3, label=label, **style)

    ax.set_xlabel("Episode")
    ax.set_ylabel("Mean evaluation reward (5 episodes, no exploration noise)")
    ax.set_title("Ablation: Effect of Removing Each TD3 Mechanism (seed=0)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "5_ablation_comparison.png"), dpi=150)
    plt.close(fig)


def multiseed_variance(out_dir="figures"):
    paths = ["results", "results_seed1", "results_seed2"]
    logs = [load(p) for p in paths]
    eval_eps = logs[0]["eval_episode"]
    # all three runs share the same eval schedule (eval_every=10, 200 episodes)
    means = np.array([log["eval_reward_mean"] for log in logs])  # (3, n_evals)
    mean_of_means = means.mean(axis=0)
    std_of_means = means.std(axis=0)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(eval_eps, mean_of_means, color="tab:blue", linewidth=2, label="Mean across 3 seeds")
    ax.fill_between(
        eval_eps,
        mean_of_means - std_of_means,
        mean_of_means + std_of_means,
        alpha=0.25,
        color="tab:blue",
        label="+/- 1 std across seeds",
    )
    for i, log in enumerate(logs):
        ax.plot(eval_eps, log["eval_reward_mean"], color="gray", alpha=0.4, linewidth=1)
    ax.set_xlabel("Episode")
    ax.set_ylabel("Mean evaluation reward")
    ax.set_title("Full TD3: Variance Across 3 Training Seeds")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "6_multiseed_variance.png"), dpi=150)
    plt.close(fig)


def robustness_histogram(out_dir="figures", n_seeds=30):
    env = gym.make("Pendulum-v1")
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])

    agent = TD3Agent(state_dim, action_dim, max_action)
    agent.actor.load_state_dict(torch.load("results/actor.pt"))
    agent.actor.eval()

    returns = []
    for seed in range(n_seeds):
        state, _ = env.reset(seed=2000 + seed)
        done, ep_r = False, 0.0
        while not done:
            a = agent.select_action(state, noise_std=0.0)
            state, r, term, trunc, _ = env.step(a)
            ep_r += r
            done = term or trunc
        returns.append(ep_r)
    env.close()
    returns = np.array(returns)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.hist(returns, bins=12, color="tab:purple", edgecolor="black", alpha=0.75)
    ax.axvline(returns.mean(), color="black", linestyle="--", linewidth=1.5, label=f"mean = {returns.mean():.1f}")
    ax.set_xlabel("Episode return (deterministic policy)")
    ax.set_ylabel("Count")
    ax.set_title(f"Final Policy Robustness Across {n_seeds} Random Start Angles")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "7_robustness_histogram.png"), dpi=150)
    plt.close(fig)

    print(f"robustness: mean={returns.mean():.1f} std={returns.std():.1f} "
          f"min={returns.min():.1f} max={returns.max():.1f} "
          f"worse_than_-300={(returns < -300).sum()}/{n_seeds}")


if __name__ == "__main__":
    os.makedirs("figures", exist_ok=True)
    ablation_comparison()
    multiseed_variance()
    robustness_histogram()
    print("Ablation/variance/robustness figures written to figures/")
