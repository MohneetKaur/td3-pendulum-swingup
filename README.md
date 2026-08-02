# TD3 for Pendulum-v1 Swing-Up

Implementation of Twin Delayed Deep Deterministic Policy Gradient (TD3,
[Fujimoto et al. 2018](https://arxiv.org/abs/1802.09477)) solving the
inverted pendulum swing-up problem on Gymnasium's `Pendulum-v1` environment.

My solution to Problem #1 from the Jefferson Lab CST Data Scientist I panel interview.

## What's here

```
jlab-td3-interview/
├── td3/                          core algorithm (importable package)
│   ├── networks.py                 Actor + twin-critic Q-networks
│   ├── replay_buffer.py            fixed-size circular experience buffer
│   ├── agent.py                    TD3Agent - the algorithm itself
│   └── sparse_reward_wrapper.py    SparseRewardPendulum env wrapper
├── train.py                      entry point: coordinates rollout + training + logging
├── make_figures.py                figures 1-4 (main run)
├── make_ablation_figures.py       figures 5-7 (dense-reward ablation, seeds, robustness)
├── make_sparse_ablation_figure.py figure 8 (sparse-reward ablation)
├── results/                       one subfolder per training run (log.json, actor.pt, critic.pt)
├── figures/                       generated PNGs, 1-8
├── logs/                          captured stdout per run
├── README.md, requirements.txt, .gitignore
```

- `td3/networks.py` - Actor (deterministic policy) and twin-critic Q-networks.
- `td3/replay_buffer.py` - fixed-size circular experience replay buffer.
- `td3/agent.py` - `TD3Agent` class implementing the three TD3 mechanisms:
  1. **Clipped double Q-learning** - two critics, Bellman target uses `min(Q1, Q2)` to
     counter overestimation bias.
  2. **Delayed policy updates** - actor and target networks updated every
     `policy_delay` critic steps (default 2).
  3. **Target policy smoothing** - clipped Gaussian noise added to the target
     action to regularize the critic against sharp, spurious Q-value peaks.
- `td3/sparse_reward_wrapper.py` - `SparseRewardPendulum`, a Gymnasium `Wrapper`
  around `Pendulum-v1` (same dynamics/state/action space) that replaces the
  dense shaped reward with a sparse `0` near-upright / `-1` elsewhere signal -
  used to stress-test the ablations below on a harder version of the same task.
- `train.py` - coordinates environment rollout, replay buffer storage, and
  agent updates; periodically evaluates the deterministic (noise-free) policy
  and logs metrics. Exposes ablation flags (`--no-double-q`, `--policy-delay`,
  `--policy-noise`, `--noise-clip`) so the same script can train degraded
  variants for comparison.
- `make_figures.py` - generates the core presentation figures from a training run.
- `make_ablation_figures.py` - generates the dense-reward ablation comparison,
  multi-seed variance, and robustness figures.
- `make_sparse_ablation_figure.py` - generates the sparse-reward ablation
  comparison figure.

## Setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Run

```bash
# main run (--out defaults to results/main)
python3 train.py --episodes 200 --start-steps 2000
python3 make_figures.py

# ablation + multi-seed runs (used by make_ablation_figures.py)
python3 train.py --episodes 200 --start-steps 2000 --no-double-q --out results/ablation_no_doubleq
python3 train.py --episodes 200 --start-steps 2000 --policy-delay 1 --out results/ablation_no_delay
python3 train.py --episodes 200 --start-steps 2000 --policy-noise 0 --noise-clip 0 --out results/ablation_no_smoothing
python3 train.py --episodes 200 --start-steps 2000 --no-double-q --policy-delay 1 --policy-noise 0 --noise-clip 0 --out results/ablation_vanilla_ddpg
python3 train.py --episodes 200 --start-steps 2000 --seed 1 --out results/seed1
python3 train.py --episodes 200 --start-steps 2000 --seed 2 --out results/seed2
python3 make_ablation_figures.py

# sparse-reward ablation (used by make_sparse_ablation_figure.py)
python3 train.py --episodes 200 --start-steps 2000 --sparse-reward --out results/sparse_full_td3
python3 train.py --episodes 200 --start-steps 2000 --sparse-reward --no-double-q --out results/sparse_no_doubleq
python3 train.py --episodes 200 --start-steps 2000 --sparse-reward --policy-delay 1 --out results/sparse_no_delay
python3 train.py --episodes 200 --start-steps 2000 --sparse-reward --policy-noise 0 --noise-clip 0 --out results/sparse_no_smoothing
python3 train.py --episodes 200 --start-steps 2000 --sparse-reward --no-double-q --policy-delay 1 --policy-noise 0 --noise-clip 0 --out results/sparse_vanilla_ddpg
python3 make_sparse_ablation_figure.py
```

Outputs:
- `results/<name>/log.json` - episode rewards, eval rewards, critic/actor losses, angle traces
- `results/<name>/actor.pt`, `results/<name>/critic.pt` - trained model checkpoints
- `figures/` - presentation-ready PNGs

## Results

Trained for 200 episodes (~40k environment steps). Deterministic policy
evaluation reward improved from **-1602 (episode 10)** to **-167 (episode 200)**,
converging by roughly episode 100-150.

See `figures/`:
- `1_episode_reward.png` / `2_eval_reward.png` - learning curves
- `3_losses.png` - critic and actor loss over training
- `4_angle_trace.png` - pendulum angle over one episode, untrained vs. trained
  policy. The untrained policy spins continuously; the trained policy swings
  up once and holds upright (theta = 0).
- `5_ablation_comparison.png` - full TD3 vs. each mechanism individually
  disabled vs. vanilla DDPG (all three off).
- `6_multiseed_variance.png` - mean +/- std learning curve across 3 training
  seeds of the full TD3 config.
- `7_robustness_histogram.png` - final trained policy evaluated across 30
  random starting angles.

### Ablation comparison (dense reward)

Trained four more variants, each with exactly one TD3 mechanism turned off
(plus a "vanilla DDPG" run with all three off), same episode budget and
seed as the main run - the point was to check whether each mechanism is
actually doing anything on this particular problem.

| Variant | Final eval reward (ep 200) |
|---|---|
| Full TD3 | -167.4 |
| No double-Q (single critic target) | -143.4 |
| No delayed updates (`policy_delay=1`) | -145.6 |
| No target policy smoothing | -142.3 |
| Vanilla DDPG (all three off) | -145.2 |

None of the four ablations caused a real breakdown here - every variant,
including vanilla DDPG, landed in a similar final range, and vanilla DDPG
actually had the smoothest run of the bunch, with the smallest mid-training
dip. That's not a bug in the setup, it's a property of the task itself:
Pendulum-v1's default reward is dense and low-dimensional, so it never
really stresses the failure modes (Q-value overestimation, actor-critic
instability) that these mechanisms exist to guard against. The multi-seed
run further down backs this up - across 3 seeds of the *full* config alone,
the episode-70 dip swings anywhere from -423 to -714 purely from random
seed, which is a wider spread than what separates the different ablated
variants at a single fixed seed. On this task, seed noise outweighs
mechanism choice. TD3's fixes are better motivated by harder settings -
higher-dimensional control, longer horizons, sparser rewards - which is
also probably why Pendulum-v1 ended up such a common intro benchmark: it's
forgiving enough that plain DDPG mostly gets away with it.

### Sparse-reward ablation

Since the dense-reward comparison didn't settle anything, the natural next
step was to make the task actually hard enough to test the claim properly.
`SparseRewardPendulum` (`sparse_reward_wrapper.py`) keeps the exact same
physics but swaps the reward for `0` within about 11 degrees of upright and
`-1` everywhere else - no gradient pointing toward the goal anymore, much
harder credit assignment, and a lot closer to the conditions where DDPG's
instabilities are known to actually show up. Reran the same 5 configs on it.

| Variant | Final eval reward (ep 200, sparse) |
|---|---|
| Full TD3 | -32.2 |
| No double-Q (single critic) | -35.8 |
| No delayed updates | -40.4 |
| No target policy smoothing | -196.4 (failed to learn) |
| Vanilla DDPG (all three off) | -188.2 (failed to learn) |

This time it's not close (see `8_sparse_reward_ablation.png`). Target policy
smoothing turns out to be the piece that's actually load-bearing here -
drop it and the agent never gets off the random-policy floor (-200) across
the whole 200-episode budget. Vanilla DDPG dies the same way, for the same
reason. Double-Q and delayed updates matter too, just differently - both
eventually reach close to full TD3's final level, they just take much
longer to get there: full TD3 is already around -40 by episode 60, while
no-double-Q and no-delay don't catch up until roughly episode 130-150.
So on this version of the task, smoothing decides whether learning happens
at all, and the other two mostly decide how fast it happens.

Makes sense given what smoothing is actually for - stopping the critic from
latching onto one narrow, spurious action instead of judging a whole
neighborhood of similar ones. Sparse reward gives the critic almost no
grounded signal to work with, since most transitions score an identical
`-1`, so it's far more exposed to that exact failure than under a dense
reward. That's the whole story of why this mechanism barely registered
above but turned out to be decisive here.

### Multi-seed variance (full TD3, 3 seeds)

Same config, three seeds (0/1/2), same 200 episodes each: final eval reward
comes out to -167.4 / -143.1 / -154.3 (mean -155.0, std 10.1). All three
share the same transient dip around episode 60-70 before settling in by
roughly episode 100-120. See `6_multiseed_variance.png`.

### Robustness across starting angles

Took the seed-0 final policy and ran it on 30 fresh random starting angles
instead of just the handful seen during training: mean return -157.9,
std 93.1, range [-368.3, -0.1]. Three of the thirty (10%) come in worse
than -300 - all of them starts near the bottom of the pendulum, which
costs more to swing up from than a near-upright start, even for a genuinely
optimal policy. See `7_robustness_histogram.png`.

## Key hyperparameters (defaults in `train.py` / `td3/agent.py`)

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
