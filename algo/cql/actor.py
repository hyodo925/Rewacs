import numpy as np
import torch as torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from .utils import extend_and_repeat

class SocialActorCQL(nn.Module):
    def __init__(
        self,
        D,
        d,
        h_dims=[256],
        action_space=[-1, 1],
        activation="leaky_relu",
        integrator=None,
        log_std_max=0,
        log_std_min=-6,
    ):
        super().__init__()
        self.integrator = integrator

        self.net = self.make_mlp([D] + h_dims, activation=activation, last_act=True)

        self.mean_linear = nn.Linear(h_dims[-1], d)
        self.log_std_logits = nn.Linear(h_dims[-1], d)

        self.act_min = action_space[0]
        self.act_max = action_space[1]

        self.log_std_max = log_std_max
        self.log_std_min = log_std_min

    def make_mlp(self, mlp_dims, activation="leaky_relu", last_act=False):
        layers = []
        for i in range(len(mlp_dims) - 1):
            layers.append(nn.Linear(mlp_dims[i], mlp_dims[i + 1]))
            if i != len(mlp_dims) - 2 or last_act:
                if activation == "relu":
                    layers.append(nn.ReLU())
                elif activation == "leaky_relu":
                    layers.append(nn.LeakyReLU())
        return nn.Sequential(*layers)

    def forward(self, data, repeat: int = None):
        if repeat is not None:
            data = extend_and_repeat(data, 1, repeat)

        if self.integrator is not None:
            data = self.integrator(*data)

        x = self.net(data)
        mean = self.mean_linear(x)

        log_std = torch.sigmoid(self.log_std_logits(x))
        log_std = self.log_std_min + log_std * (self.log_std_max - self.log_std_min)

        return mean, log_std

    def sample(self, data, repeat: int = None):
        mean, log_std = self.forward(data, repeat=repeat)
        std = log_std.exp()

        mean = torch.tanh(mean) * self.act_max
        dist = Normal(mean, std)

        action = dist.rsample()
        log_prob = dist.log_prob(action).sum(axis=-1, keepdim=True)

        return action, log_prob, mean

    def get_log_prob(self, data, action):
        if action.ndim == 3:
            data = extend_and_repeat(data, 1, action.shape[1])

        if self.integrator is not None:
            data = self.integrator(*data)

        x = self.net(data)
        mean = self.mean_linear(x)
        mean = torch.tanh(mean) * self.act_max

        log_std = torch.sigmoid(self.log_std_logits(x))
        log_std = self.log_std_min + log_std * (self.log_std_max - self.log_std_min)
        std = torch.exp(log_std)

        dist = Normal(mean, std)
        log_prob = dist.log_prob(action).sum(axis=-1, keepdim=True)

        return log_prob
