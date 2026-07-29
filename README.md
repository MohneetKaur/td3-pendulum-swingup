# TD3 for Pendulum-v1 Swing-Up

Implementation of Twin Delayed Deep Deterministic Policy Gradient (TD3,
[Fujimoto et al. 2018](https://arxiv.org/abs/1802.09477)) solving the
inverted pendulum swing-up problem on Gymnasium's `Pendulum-v1` environment.

Prepared for the Jefferson Lab CST Data Scientist I interview problem set.

## What's here

```
jlab-td3-interview/
├── td3/                          core algorithm (importable package)
│   ├── networks.py                 Actor + twin-critic Q-networks
│   ├── replay_buffer.py            fixed-size circular experience buffer
│   ├── agent.py                    TD3Agent — the algorithm itself
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

- `td3/networks.py` — Actor (deterministic policy) and twin-critic Q-networks.
- `td3/replay_buffer.py` — fixed-size circular experience replay buffer.
- `td3/agent.py` — `TD3Agent` class implementing the three TD3 mechanisms:
  1. **Clipped double Q-learning** — two critics, Bellman target uses `min(Q1, Q2)` to
     counter overestimation bias.
  2. **Delayed policy updates** — actor and target networks updated every
     `policy_delay` critic steps (default 2).
  3. **Target policy smoothing** — clipped Gaussian noise added to the target
     action to regularize the critic against sharp, spurious Q-value peaks.
- `td3/sparse_reward_wrapper.py` — `SparseRewardPendulum`, a Gymnasium `Wrapper`
  around `Pendulum-v1` (same dynamics/state/action space) that replaces the
  dense shaped reward with a sparse `0` near-upright / `-1` elsewhere signal —
  used to stress-test the ablations below on a harder version of the same task.
- `train.py` — coordinates environment rollout, replay buffer storage, and
  agent updates; periodically evaluates the deterministic (noise-free) policy
  and logs metrics. Exposes ablation flags (`--no-double-q`, `--policy-delay`,
  `--policy-noise`, `--noise-clip`) so the same script can train degraded
  variants for comparison.
- `make_figures.py` — generates the core presentation figures from a training run.
- `make_ablation_figures.py` — generates the dense-reward ablation comparison,
  multi-seed variance, and robustness figures.
- `make_sparse_ablation_figure.py` — generates the sparse-reward ablation
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
- `results/<name>/log.json` — episode rewards, eval rewards, critic/actor losses, angle traces
- `results/<name>/actor.pt`, `results/<name>/critic.pt` — trained model checkpoints
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

### Follow-up: sparse-reward ablation

The dense-reward ablation above found no meaningful difference between
variants because Pendulum-v1's default reward is densely shaped — it gives
gradient toward the goal at every step, which doesn't stress the failure
modes TD3's mechanisms exist to prevent. To actually test that hypothesis,
we re-ran the same 5 configs on `SparseRewardPendulum` (`sparse_reward_wrapper.py`):
same environment dynamics, but reward is `0` within ~11 degrees of upright
and `-1` everywhere else, removing the dense gradient and making credit
assignment much harder — closer to the conditions where DDPG's known
instabilities actually surface.

| Variant | Final eval reward (ep 200, sparse) |
|---|---|
| Full TD3 | -32.2 |
| No double-Q (single critic) | -35.8 |
| No delayed updates | -40.4 |
| **No target policy smoothing** | **-196.4 (failed to learn)** |
| **Vanilla DDPG (all three off)** | **-188.2 (failed to learn)** |

See `8_sparse_reward_ablation.png`. This time the ablation is decisive:
**target policy smoothing is the load-bearing mechanism on this harder
task** — removing it alone collapses performance to essentially the
random-policy floor (-200), and it never recovers within the 200-episode
budget. Vanilla DDPG fails for the same reason (it also lacks smoothing).
Double-Q and delayed updates matter too, but differently: both eventually
converge to close to full TD3's final performance, just far slower —
looking at the full learning curves, full TD3 reaches ~-40 by episode 60,
while no-double-Q and no-delay don't reach that level until episode
~130-150. So on this task: smoothing is necessary for learning to happen
at all, while double-Q and delay mainly control *how fast* it happens.

This is consistent with the theory: target policy smoothing exists to stop
the critic from overfitting to narrow, spurious action peaks. Under a
sparse reward, the Q-function has very little grounded signal to learn
from (most transitions score `-1` identically), so it's far more prone to
exactly this kind of narrow overfitting than under a dense reward — which
is why smoothing's absence is catastrophic here but nearly invisible under
the dense reward above.

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
