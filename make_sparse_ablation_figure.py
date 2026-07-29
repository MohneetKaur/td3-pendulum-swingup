"""Builds the sparse-reward ablation comparison figure from the
results_sparse_* runs -- a harder variant of Pendulum-v1 (see
sparse_reward_wrapper.py) used to stress-test whether each TD3 mechanism
is actually load-bearing, since the dense-reward ablation (see
make_ablation_figures.py) showed no meaningful difference between variants.
"""
import json
import os

import matplotlib.pyplot as plt


def load(path):
    with open(os.path.join(path, "log.json")) as f:
        return json.load(f)


def main(out_dir="figures"):
    runs = {
        "Full TD3": "results_sparse_full_td3",
        "No double-Q (single critic)": "results_sparse_no_doubleq",
        "No delayed updates": "results_sparse_no_delay",
        "No target smoothing": "results_sparse_no_smoothing",
        "Vanilla DDPG (all off)": "results_sparse_vanilla_ddpg",
    }
    colors = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:gray"]

    fig, ax = plt.subplots(figsize=(8, 5))
    for (label, path), color in zip(runs.items(), colors):
        log = load(path)
        style = dict(linewidth=2.5, color=color) if label == "Full TD3" else dict(
            linewidth=1.8, color=color, linestyle="--"
        )
        ax.plot(log["eval_episode"], log["eval_reward_mean"], marker="o", markersize=3, label=label, **style)

    ax.axhline(-200, color="black", linestyle=":", linewidth=1, alpha=0.5, label="random-policy floor")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Mean evaluation reward (sparse: 0 near-upright, -1 elsewhere)")
    ax.set_title("Sparse-Reward Pendulum: Effect of Removing Each TD3 Mechanism")
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "8_sparse_reward_ablation.png"), dpi=150)
    plt.close(fig)
    print("Sparse-reward ablation figure written to figures/8_sparse_reward_ablation.png")


if __name__ == "__main__":
    main()
