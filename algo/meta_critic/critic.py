import numpy as np
import torch as torch
import torch.nn as nn
from .utils import extend_and_repeat
import torch.nn.functional as F

class SocialCritic(nn.Module):
    def __init__(
        self, D, d, h_dims=[256], integrator=None, activation="leaky_relu", single=False
    ):
        super().__init__()
        self.integrator = integrator
        self.single = single
        self.net1 = self.make_mlp(
            [D] + h_dims + [d], activation=activation, last_act=True
        )

        if not single:
            self.net2 = self.make_mlp(
                [D] + h_dims + [d], activation=activation, last_act=True
            )

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

    def forward(self, obs, act=None):
        #For calql method
        if act is not None and act.ndim == 3:
            obs = extend_and_repeat(obs, dim=1, repeat=act.shape[1])
        if self.integrator != None:
            data = self.integrator(*obs)

        if not self.single:
            data = torch.cat([data, act], -1)
    
        out1 = self.net1(data)

        if not self.single:
            out2 = self.net2(data)
            return out1, out2
        else:
            return out1