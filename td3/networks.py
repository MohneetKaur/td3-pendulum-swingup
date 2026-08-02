import torch
import torch.nn as nn


class Actor(nn.Module):
    """Deterministic policy: state -> action. tanh output is scaled by
    max_action so the network's action bound matches the environment's,
    without needing to clip anywhere downstream."""

    def __init__(self, state_dim, action_dim, max_action, hidden_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Tanh(),
        )
        self.max_action = max_action

    def forward(self, state):
        return self.max_action * self.net(state)


class Critic(nn.Module):
    """Twin Q-networks. Both share the same forward() call so training
    can update them together, but each has independent parameters."""

    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super().__init__()
        self.q1 = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        self.q2 = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, state, action):
        sa = torch.cat([state, action], dim=1)
        return self.q1(sa), self.q2(sa)

    def q1_forward(self, state, action):
        """Q1 only - used for the actor's policy gradient. TD3 convention
        is to backprop through a single critic here; both critics are
        only combined (via min) when computing the Bellman target."""
        sa = torch.cat([state, action], dim=1)
        return self.q1(sa)
