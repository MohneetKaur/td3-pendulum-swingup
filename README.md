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
  and logs metrics. Exposes ablation flags (`--no-double-q`, `--policy-delay`,
  `--policy-noise`, `--noise-clip`) so the same script can train degraded
  variants for comparison.
- `make_figures.py` — generates the core presentation figures from a training run.
- `make_ablation_figures.py` — generates the ablation comparison, multi-seed
  variance, and robustness figures from the additional runs below.

## Setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Run

```bash
# main run
python3 train.py --episodes 200 --start-steps 2000 --out results
python3 make_figures.py

# ablation + multi-seed runs (used by make_ablation_figures.py)
python3 train.py --episodes 200 --start-steps 2000 --no-double-q --out results_ablation_no_doubleq
python3 train.py --episodes 200 --start-steps 2000 --policy-delay 1 --out results_ablation_no_delay
python3 train.py --episodes 200 --start-steps 2000 --policy-noise 0 --noise-clip 0 --out results_ablation_no_smoothing
python3 train.py --episodes 200 --start-steps 2000 --no-double-q --policy-delay 1 --policy-noise 0 --noise-clip 0 --out results_ablation_vanilla_ddpg
python3 train.py --episodes 200 --start-steps 2000 --seed 1 --out results_seed1
python3 train.py --episodes 200 --start-steps 2000 --seed 2 --out results_seed2
python3 make_ablation_figures.py
```

Outputs:
- `results*/log.json` — episode rewards, eval rewards, critic/actor losses, angle traces
- `results*/actor.pt`, `results*/critic.pt` — trained model checkpoints
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
- `5_ablation_comparison.png` — full TD3 vs. each mechanism individually
  disabled vs. vanilla DDPG (all three off).
- `6_multiseed_variance.png` — mean +/- std learning curve across 3 training
  seeds of the full TD3 config.
- `7_robustness_histogram.png` — final trained policy evaluated across 30
  random starting angles.

### Ablation study: what we found

We trained four additional variants, each disabling exactly one TD3
mechanism (plus a "vanilla DDPG" variant with all three off), same episode
budget and seed as the main run, to test whether each mechanism is actually
load-bearing on this problem:

| Variant | Final eval reward (ep 200) |
|---|---|
| Full TD3 | -167.4 |
| No double-Q (single critic target) | -143.4 |
| No delayed updates (`policy_delay=1`) | -145.6 |
| No target policy smoothing | -142.3 |
| Vanilla DDPG (all three off) | -145.2 |

**Honest finding:** on `Pendulum-v1`, none of the ablations caused a
dramatic failure — all variants, including vanilla DDPG, converged to
similar final performance, and in this run vanilla DDPG had the smoothest
convergence with the smallest mid-training dip. This is not a bug; it
reflects that Pendulum-v1 is a simple, low-dimensional, densely-shaped-reward
benchmark that doesn't stress DDPG's known failure modes (Q-value
overestimation, actor-critic instability) as hard as higher-dimensional or
sparser-reward control problems would. The multi-seed variance run below
supports this: across 3 seeds of the *full* TD3 config alone, the
episode-70 dip ranges from -423 to -714 depending purely on seed — larger
than the spread between different ablated variants at seed 0. In other
words, on this task, seed variance dominates over the specific mechanism
being ablated. TD3's mechanisms are best motivated as protecting against
failure modes that compound over harder problems (e.g. higher-dimensional
continuous control, longer horizons, or sparser rewards) — Pendulum-v1's
simplicity is exactly why it's a common introductory benchmark, and this
result is consistent with that.

### Multi-seed variance (full TD3, 3 seeds)

Final eval reward across seeds 0/1/2: -167.4 / -143.1 / -154.3
(mean -155.0, std 10.1). All three seeds show a transient dip around
episode 60-70 before converging tightly by episode ~100-120 — see
`6_multiseed_variance.png`.

### Robustness across starting angles

Evaluating the final trained policy (seed 0) across 30 different random
starting angles: mean return -157.9, std 93.1, range [-368.3, -0.1].
3 of 30 episodes (10%) scored worse than -300 — these correspond to
starts near the bottom of the pendulum, which inherently cost more reward
to swing up from than starts near the top, even under an optimal policy.
See `7_robustness_histogram.png`.

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
