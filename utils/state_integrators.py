import numpy as np
import torch as torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.uniform import Uniform
from torch.nn.functional import softmax

from utils.graph import pos_to_graph
from utils.layers import GCLayer, GraphAttentionLayer


def fanin_init(size, fanin=None):
    fanin = fanin or size[0]
    v = 1.0 / np.sqrt(fanin)
    return torch.Tensor(size).uniform_(-v, v)


class ObsEnc(nn.Module):
    def __init__(self, D, d, h_dims=[64], last_act=True):
        super().__init__()
        self.net = self.make_mlp([D] + h_dims + [d], last_act=last_act)
        # self.init_weights(init_w)

    def make_mlp(self, mlp_dims, activation="mish", last_act=False):
        layers = []
        mlp_dims = mlp_dims
        for i in range(len(mlp_dims) - 1):
            layers.append(nn.Linear(mlp_dims[i], mlp_dims[i + 1]))
            if i != len(mlp_dims) - 2 or last_act:
                if activation == "relu":
                    layers.append(nn.ReLU())
                elif activation == "leaky_relu":
                    layers.append(nn.LeakyReLU())
                elif activation == "mish":
                    layers.append(nn.Mish())
        net = nn.Sequential(*layers)
        return net

    def init_weights(self, init_w):
        self.model.fc1.weight.data = fanin_init(self.model.fc1.weight.data.size())
        self.model.fc2.weight.data = fanin_init(self.model.fc2.weight.data.size())
        self.model.fc3.weight.data.uniform_(-init_w, init_w)

    def forward(self, data):
        out = self.net(data)

        return out


class GCObsIntegrator(nn.Module):
    def __init__(
        self, obs_dim, r_obs_dim, projection_dim, enc_hdims=[64], prediction=False
    ):
        super().__init__()
        self.enc_r_obs = ObsEnc(
            r_obs_dim, projection_dim, h_dims=enc_hdims, last_act=True
        )
        self.enc_obs = ObsEnc(obs_dim, projection_dim, h_dims=enc_hdims, last_act=True)
        self.gcl1 = GCLayer(projection_dim, projection_dim)
        self.gcl2 = GCLayer(projection_dim, projection_dim)
        self.output_dim = projection_dim
        self.prediction = prediction

        self.projection_dim = projection_dim
        self.obs_dim = obs_dim
        self.r_obs_dim = r_obs_dim
        # self.register_parameter('w_a', nn.Parameter(torch.randn(projection_dim, projection_dim).detach()))

    def forward(self, obs, r_obs):
        n = obs.shape[0]
        p_num = obs.shape[1]
        obs_cat = torch.cat(
            (
                r_obs[:, :, :2],
                obs.reshape((n, -1, self.obs_dim))[:, :, :2],
            ),
            1,
        )
        _, adj_mat = pos_to_graph(obs_cat.reshape(-1, p_num + 1, 2).detach())

        enc_r_obs = self.enc_r_obs(r_obs)
        enc_obs = self.enc_obs(obs.reshape((n, -1, self.obs_dim)))
        obs_stack = torch.cat(
            (enc_r_obs.reshape(n, -1, self.projection_dim), enc_obs), 1
        )
        # A = torch.matmul(torch.matmul(obs_stack, self.w_a), obs_stack.permute(0, 2, 1))
        # adj_mat = softmax(A, dim=2)
        obs_gc = self.gcl1(obs_stack, adj_mat) + obs_stack
        obs_gc2 = self.gcl2(obs_gc, adj_mat) + obs_gc

        if self.prediction:
            integrated = obs_gc2[:, 1:, :].reshape(-1, p_num, self.output_dim)
        else:
            integrated = obs_gc2[:, 0, :].reshape(-1, self.output_dim)

        return integrated
        # return obs_stack.reshape(n, -1)


class EmbeddedGaussianIntegrator(nn.Module):
    def __init__(
        self, obs_dim, r_obs_dim, projection_dim, enc_hdims=[64], prediction=False
    ):
        super().__init__()
        self.enc_r_obs = ObsEnc(
            r_obs_dim, projection_dim, h_dims=enc_hdims, last_act=True
        )
        self.enc_obs = ObsEnc(obs_dim, projection_dim, h_dims=enc_hdims, last_act=True)
        self.gcl1 = GCLayer(projection_dim, projection_dim)
        self.gcl2 = GCLayer(projection_dim, projection_dim)
        self.output_dim = projection_dim
        self.prediction = prediction

        self.projection_dim = projection_dim
        self.obs_dim = obs_dim
        self.r_obs_dim = r_obs_dim
        self.register_parameter(
            "w_a", nn.Parameter(torch.randn(projection_dim, projection_dim).detach())
        )

    def forward(self, obs, r_obs):
        n = obs.shape[0]
        p_num = obs.shape[1]
        if len(r_obs.shape) < 3:
            r_obs = r_obs.reshape((n, 1, self.r_obs_dim))

        enc_r_obs = self.enc_r_obs(r_obs)
        enc_obs = self.enc_obs(obs.reshape((n, -1, self.obs_dim)))
        obs_stack = torch.cat(
            (enc_r_obs.reshape(n, -1, self.projection_dim), enc_obs), 1
        )
        A = torch.matmul(torch.matmul(obs_stack, self.w_a), obs_stack.permute(0, 2, 1))
        adj_mat = softmax(A, dim=2)
        obs_gc = self.gcl1(obs_stack, adj_mat) + obs_stack
        obs_gc2 = self.gcl2(obs_gc, adj_mat) + obs_gc
        if self.prediction:
            integrated = obs_gc2[:, 1:, :].reshape(-1, p_num, self.output_dim)
        else:
            integrated = obs_gc2[:, 0, :].reshape(-1, self.output_dim)

        return integrated
        # return obs_stack.reshape(n, -1)


class GATIntegrator(nn.Module):
    def __init__(
        self,
        obs_dim,
        r_obs_dim,
        projection_dim,
        enc_hdims=[64],
        concat=True,
        n_heads=8,
        prediction=False,
        dropout_rate=0.0,
        alpha=0.2,
    ):
        super().__init__()
        self.concat = concat
        self.n_heads = n_heads
        self.dropout_rate = dropout_rate
        self.alpha = alpha

        if concat:
            assert projection_dim % n_heads == 0
            self.h_dim = projection_dim // n_heads
        else:
            self.h_dim = projection_dim

        self.projection_dim = projection_dim

        self.enc_r_obs = ObsEnc(
            r_obs_dim, projection_dim, h_dims=enc_hdims, last_act=True
        )
        self.enc_obs = ObsEnc(obs_dim, projection_dim, h_dims=enc_hdims, last_act=True)
        self.output_dim = projection_dim
        self.prediction = prediction

        self.projection_dim = projection_dim
        self.obs_dim = obs_dim
        self.r_obs_dim = r_obs_dim

        self.gat1 = GraphAttentionLayer(
            in_features=projection_dim,
            out_features=projection_dim,
            n_heads=n_heads,
            dropout_rate=dropout_rate,
            alpha=alpha,
            concat=concat,
        )

        self.gat2 = GraphAttentionLayer(
            in_features=projection_dim,
            out_features=projection_dim,
            n_heads=n_heads,
            dropout_rate=dropout_rate,
            alpha=alpha,
            concat=concat,
        )

    def forward(self, obs, r_obs):
        n = obs.shape[0]
        p_num = obs.shape[1]
        if len(r_obs.shape) < 3:
            r_obs = r_obs.reshape((n, 1, self.r_obs_dim))

        enc_r_obs = self.enc_r_obs(r_obs)
        enc_obs = self.enc_obs(obs.reshape((n, -1, self.obs_dim)))
        obs_stack = torch.cat(
            (enc_r_obs.reshape(n, -1, self.projection_dim), enc_obs), 1
        )

        obs_gat1 = self.gat1(obs_stack) + obs_stack
        obs_gat2 = self.gat2(obs_gat1) + obs_gat1
        if self.prediction:
            integrated = obs_gat2[:, 1:, :].reshape(-1, p_num, self.output_dim)
        else:
            integrated = obs_gat2[:, 0, :].reshape(-1, self.output_dim)

        return integrated
        # return obs_stack.reshape(n, -1)
