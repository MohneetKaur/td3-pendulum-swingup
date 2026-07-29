# TD3 for Pendulum-v1 Swing-Up

Implementation of Twin Delayed Deep Deterministic Policy Gradient (TD3,
[Fujimoto et al. 2018](https://arxiv.org/abs/1802.09477)) solving the
inverted pendulum swing-up problem on Gymnasium's `Pendulum-v1` environment.

Prepared for the Jefferson Lab CST Data Scientist I interview problem set.

## What's here

- `networks.py` — Actor (deterministic policy) and twin-critic Q-networks.
- `replay_buffer.py` — fixed-size circular experience replay buffer.
- `td3.py` — `TD3Agent` class implementing the three TD3 mechanisms:
  1. **Clipped double Q-learning** — two critics, Bellman target uses `min(Q1, Q2)` to
     counter overestimation bias.
  2. **Delayed policy updates** — actor and target networks updated every
     `policy_delay` critic steps (default 2).
  3. **Target policy smoothing** — clipped Gaussian noise added to the target
     action to regularize the critic against sharp, spurious Q-value peaks.
- `train.py` — coordinates environment rollout, replay buffer storage, and
  agent updates; periodically evaluates the deterministic (noise-free) policy
  and logs metrics.
- `make_figures.py` — generates the presentation figures from a training run.

## Setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Run

```bash
python3 train.py --episodes 200 --start-steps 2000 --out results
python3 make_figures.py
```

Outputs:
- `results/log.json` — episode rewards, eval rewards, critic/actor losses, angle traces
- `results/actor.pt`, `results/critic.pt` — trained model checkpoints
- `figures/` — presentation-ready PNGs

## Results

Trained for 200 episodes (~40k environment steps). Deterministic policy
evaluation reward improved from **-1602 (episode 10)** to **-167 (episode 200)**,
converging by roughly episode 100-150.

See `figures/`:
- `1_episode_reward.png` / `2_eval_reward.png` — learning curves
- `3_losses.png` — critic and actor loss over training
- `4_angle_trace.png` — pendulum angle over one episode, untrained vs. trained
  policy. The untrained policy spins continuously; the trained policy swings
  up once and holds upright (theta = 0).

## Key hyperparameters (defaults in `train.py` / `td3.py`)

| Param | Value |
|---|---|
| discount (gamma) | 0.99 |
| target smoothing coefficient (tau) | 0.005 |
| target policy noise | 0.2 x max_action |
| noise clip | 0.5 x max_action |
| policy delay | 2 |
| exploration noise (rollout) | 0.1 x max_action |
| batch size | 256 |
| warmup steps (random actions) | 2000 |
