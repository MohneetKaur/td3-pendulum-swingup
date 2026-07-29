import argparse
import json
import os

import gymnasium as gym
import numpy as np

from replay_buffer import ReplayBuffer
from td3 import TD3Agent


def evaluate(agent, env_name, episodes=5, seed=100):
    env = gym.make(env_name)
    returns = []
    for i in range(episodes):
        state, _ = env.reset(seed=seed + i)
        done, ep_return = False, 0.0
        while not done:
            action = agent.select_action(state, noise_std=0.0)
            state, reward, terminated, truncated, _ = env.step(action)
            ep_return += reward
            done = terminated or truncated
        returns.append(ep_return)
    env.close()
    return float(np.mean(returns)), float(np.std(returns))


def record_angle_trace(agent, env_name, seed=42):
    """Run one greedy episode and log pendulum angle (theta) over time."""
    env = gym.make(env_name)
    state, _ = env.reset(seed=seed)
    thetas, done = [], False
    while not done:
        cos_th, sin_th = state[0], state[1]
        thetas.append(float(np.arctan2(sin_th, cos_th)))
        action = agent.select_action(state, noise_std=0.0)
        state, _, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
    env.close()
    return thetas


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", default="Pendulum-v1")
    parser.add_argument("--episodes", type=int, default=200)
    parser.add_argument("--max-steps", type=int, default=200)  # Pendulum-v1 default horizon
    parser.add_argument("--start-steps", type=int, default=2000)  # pure exploration warmup
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--explore-noise", type=float, default=0.1)  # fraction of max_action
    parser.add_argument("--eval-every", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default="results")
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)

    np.random.seed(args.seed)
    import torch

    torch.manual_seed(args.seed)

    env = gym.make(args.env)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])

    agent = TD3Agent(state_dim, action_dim, max_action)
    buffer = ReplayBuffer(state_dim, action_dim)

    # capture behavior before any training for the before/after figure
    theta_untrained = record_angle_trace(agent, args.env)

    log = {
        "episode_reward": [],
        "eval_episode": [],
        "eval_reward_mean": [],
        "eval_reward_std": [],
        "critic_loss": [],
        "actor_loss": [],
    }

    total_steps = 0
    for episode in range(1, args.episodes + 1):
        state, _ = env.reset(seed=args.seed + episode)
        episode_reward = 0.0

        for _ in range(args.max_steps):
            total_steps += 1

            if total_steps < args.start_steps:
                action = env.action_space.sample()
            else:
                action = agent.select_action(state, noise_std=args.explore_noise)

            next_state, reward, terminated, truncated, _ = env.step(action)
            done_flag = float(terminated or truncated)

            buffer.add(state, action, reward, next_state, done_flag)
            state = next_state
            episode_reward += reward

            if len(buffer) >= args.batch_size and total_steps >= args.start_steps:
                info = agent.train(buffer, batch_size=args.batch_size)
                log["critic_loss"].append(info["critic_loss"])
                if info["actor_loss"] is not None:
                    log["actor_loss"].append(info["actor_loss"])

            if terminated or truncated:
                break

        log["episode_reward"].append(episode_reward)
        print(f"Episode {episode:4d} | reward: {episode_reward:8.2f} | buffer: {len(buffer)}")

        if episode % args.eval_every == 0:
            mean_r, std_r = evaluate(agent, args.env)
            log["eval_episode"].append(episode)
            log["eval_reward_mean"].append(mean_r)
            log["eval_reward_std"].append(std_r)
            print(f"  eval @ ep {episode}: {mean_r:.2f} +/- {std_r:.2f}")

    log["theta_untrained"] = theta_untrained
    log["theta_trained"] = record_angle_trace(agent, args.env)

    with open(os.path.join(args.out, "log.json"), "w") as f:
        json.dump(log, f)

    torch.save(agent.actor.state_dict(), os.path.join(args.out, "actor.pt"))
    torch.save(agent.critic.state_dict(), os.path.join(args.out, "critic.pt"))

    env.close()
    print(f"Done. Logs and checkpoints written to {args.out}/")


if __name__ == "__main__":
    main()
