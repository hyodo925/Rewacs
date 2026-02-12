import numpy as np
import torch as torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal


class SocialActorMetaAWAC(nn.Module):
    def __init__(
        self,
        D,
        d,
        h_dims=[256],
        action_space=[-1, 1],
        activation="leaky_relu",
        aggregator=None,
        penultimate_norm=False,
        log_std_max=0,
        log_std_min=-6,
    ):
        super().__init__()
        self.aggregator = aggregator

        self.net = self.make_mlp([D] + h_dims, activation=activation, last_act=True)

        self.mean_linear = nn.Linear(h_dims[-1], d)
        # self.log_std_logits = nn.Parameter(torch.zeros(d, requires_grad=True))
        self.log_std_logits = nn.Linear(h_dims[-1], d)

        self.act_min = action_space[0]
        self.act_max = action_space[1]

        self.log_std_max = log_std_max
        self.log_std_min = log_std_min

        self.penultimate_norm = penultimate_norm

        # self.apply(weights_init_)

    def make_mlp(self, mlp_dims, activation="leaky_relu", last_act=False):
        layers = []
        mlp_dims = mlp_dims
        for i in range(len(mlp_dims) - 1):
            layers.append(nn.Linear(mlp_dims[i], mlp_dims[i + 1]))
            if i != len(mlp_dims) - 2 or last_act:
                if activation == "relu":
                    layers.append(nn.ReLU())
                elif activation == "leaky_relu":
                    layers.append(nn.LeakyReLU())
        net = nn.Sequential(*layers)
        return net

    def get_dist(self, data):
        mean, log_std = self.forward(data)
        std = log_std.exp()

        return Normal(mean, std)

    def forward(self, data):
        if self.aggregator is not None:
            data = self.aggregator(*data)
        x = self.net(data)
        if self.penultimate_norm:
            x = x / torch.norm(x, dim=1).view((-1, 1))
        mean = self.mean_linear(x)

        log_std = torch.sigmoid(self.log_std_logits(x))

        log_std = self.log_std_min + log_std * (self.log_std_max - self.log_std_min)

        return mean, log_std, x

    def get_log_prob(self, data, action):
        if self.aggregator is not None:
            data = self.aggregator(*data)
        x = self.net(data)
        if self.penultimate_norm:
            x = x / torch.norm(x, dim=1).view((-1, 1))
        mean = self.mean_linear(x)
        mean = torch.tanh(mean) * self.act_max
        log_std = torch.sigmoid(self.log_std_logits(x))
        log_std = self.log_std_min + log_std * (self.log_std_max - self.log_std_min)
        std = torch.exp(log_std)
        dist = Normal(mean, std)
        logp_prob = dist.log_prob(action).sum(axis=-1, keepdim=True)

        return logp_prob

    def sample(self, data):
        mean, log_std, x = self.forward(data)
        std = log_std.exp()
        mean = torch.tanh(mean) * self.act_max
        dist = Normal(mean, std)
        action = dist.rsample()
        log_prob = dist.log_prob(action).sum(axis=-1, keepdim=True)
        # action = torch.clamp(action, self.act_min, self.act_max)
        # action = torch.tanh(action) * self.act_max
        # mean = torch.clamp(mean, self.act_min, self.act_max)
        return action, log_prob, mean, x


class SocialMetaCritic(nn.Module):
    def __init__(
        self,
        D,
        d,
        h_dims=[256],
        aggregator=None,
        penultimate_norm=False,
        activation="leaky_relu",
    ):
        super().__init__()
        self.aggregator = aggregator
        self.net = self.make_mlp([D] + h_dims, activation=activation, last_act=True)

        self.last_linear = nn.Linear(h_dims[-1], d)

        self.penultimate_norm = penultimate_norm

        # self.apply(weights_init_)

    def make_mlp(self, mlp_dims, activation="leaky_relu", last_act=False):
        layers = []
        mlp_dims = mlp_dims
        for i in range(len(mlp_dims) - 1):
            layers.append(nn.Linear(mlp_dims[i], mlp_dims[i + 1]))
            if i != len(mlp_dims) - 2 or last_act:
                if activation == "relu":
                    layers.append(nn.ReLU())
                elif activation == "leaky_relu":
                    layers.append(nn.LeakyReLU())
        net = nn.Sequential(*layers)
        return net

    def forward(self, pi_bar, obs, act=None):
        if self.aggregator is not None:
            data = self.aggregator(*obs)

        data = torch.cat([pi_bar, data, act], -1)

        phi = self.net(data)
        if self.penultimate_norm:
            phi = phi / torch.norm(phi, dim=1).view((-1, 1))

        x = self.last_linear(phi)

        x = nn.functional.softplus(x)
        # x = nn.functional.tanh(self.net(data))
        return torch.mean(x)
