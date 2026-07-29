import gymnasium as gym
import numpy as np


class SparseRewardPendulum(gym.Wrapper):
    """Wraps Pendulum-v1 with a sparse reward: 0 within `angle_threshold`
    radians of upright, -1 everywhere else, replacing the environment's
    default dense shaped reward -(theta^2 + 0.1*theta_dot^2 + 0.001*torque^2).

    Same underlying dynamics, state space, and action space as Pendulum-v1 --
    only the reward signal changes. This removes the dense gradient the
    default reward provides toward the goal, making credit assignment much
    harder and stressing the exact failure modes (Q-value overestimation,
    actor-critic instability) TD3's mechanisms are designed to counter.
    """

    def __init__(self, env, angle_threshold=0.2):
        super().__init__(env)
        self.angle_threshold = angle_threshold

    def step(self, action):
        obs, _dense_reward, terminated, truncated, info = self.env.step(action)
        cos_th, sin_th = obs[0], obs[1]
        theta = np.arctan2(sin_th, cos_th)
        reward = 0.0 if abs(theta) < self.angle_threshold else -1.0
        return obs, reward, terminated, truncated, info
