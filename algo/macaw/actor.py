import numpy as np
import torch as torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

class SocialActorMACAW(nn.Module):
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
        use_adv_head=False,
        adv_head_dims=[32, 1],
    ):
        super().__init__()
        self.integrator = integrator
        self.use_adv_head = use_adv_head
        self.net = self.make_mlp([D] + h_dims, activation=activation, last_act=True)

        self.mean_linear = nn.Linear(h_dims[-1], d)
        # self.log_std_logits = nn.Parameter(torch.zeros(d, requires_grad=True))
        self.log_std_logits = nn.Linear(h_dims[-1], d)

        if self.use_adv_head:
            adv_input_dim = h_dims[-1] + d   # feature + action
            self.adv_head = self.make_mlp(
                [adv_input_dim] + adv_head_dims,
                activation=activation,
                last_act=False
            )
        
        self.act_min = action_space[0]
        self.act_max = action_space[1]

        self.log_std_max = log_std_max
        self.log_std_min = log_std_min

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

    def forward(self, data, action=None, task_idx=None):
        """
        data: (obs, r_obs) or already integrated tensor
        action: used only when advantage head is enabled
        """
        if isinstance(data, tuple):
            assert self.integrator is not None
            x = self.integrator(*data)
        else:
            x = data
        if task_idx is not None:
            x = torch.cat([x, task_idx], -1)
        x = self.net(x)

        mean = self.mean_linear(x)
        log_std = torch.sigmoid(self.log_std_logits(x))
        log_std = self.log_std_min + log_std * (self.log_std_max - self.log_std_min)
        if action is None:
            return mean, log_std

        adv_in = torch.cat([x, action], dim=-1)
        advantage = self.adv_head(adv_in)

        return mean, advantage

    def get_log_prob(self, data, action):
        if isinstance(data, tuple):
            x = self.integrator(*data)
        else:
            x = data

        x = self.net(x)
        mean = self.mean_linear(x)
        # mean = torch.tanh(self.mean_linear(x)) * self.act_max
        log_std = torch.sigmoid(self.log_std_logits(x))
        log_std = self.log_std_min + log_std * (self.log_std_max - self.log_std_min)
        std = log_std.exp()

        dist = Normal(mean, std)
        logp = dist.log_prob(action).sum(-1, keepdim=True)
        logp -= (2 * (np.log(2) - action - F.softplus(-2 * action))).sum(
            axis=1, keepdim=True
        )
        logp = torch.clamp(logp, min=-100.0)

        if self.use_adv_head:
            adv = self.adv_head(torch.cat([x, action], dim=-1))
            return logp, adv

        return logp

    def sample(self, data):
        mean, log_std = self.forward(data)
        std = log_std.exp()
        # mean = torch.tanh(mean) * self.act_max
        dist = Normal(mean, std)
        action = dist.rsample()
        log_prob = dist.log_prob(action).sum(axis=-1, keepdim=True)
        log_prob -= (2 * (np.log(2) - action - F.softplus(-2 * action))).sum(
            axis=1, keepdim=True
        )
        log_prob = torch.clamp(log_prob, min=-100.0)
        # action = torch.clamp(action, self.act_min, self.act_max)
        action = torch.tanh(action) * self.act_max
        # mean = torch.clamp(mean, self.act_min, self.act_max)
        return action, log_prob, mean
    