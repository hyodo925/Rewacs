import numpy as np
import torch as torch
import torch.nn as nn
from torch.nn.functional import softmax
from torch.nn.functional import  relu



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


class GCLayer(nn.Module):
    def __init__(self, d, D, device="cpu"):
        super(GCLayer, self).__init__()
        self.d = d
        self.D = D
        self.device = device
        self.W = nn.Parameter(torch.randn(d, D))

    def to(self, device):
        super().to(device)
        self.device = device

    def forward(self, X, A):
        X_gc = torch.matmul(A, X)

        gc = torch.matmul(X_gc, self.W)
        out = relu(gc)
        return out

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


class EmbeddedGaussianIntegratorRepeat(nn.Module):
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
        if r_obs.ndim == 4 and r_obs.shape[2] == 1:
            r_obs = r_obs.squeeze(2)
        if obs.ndim == 4:
            B, R, P, D = obs.shape
            obs = obs.reshape(B * R, P, D)

            if r_obs.ndim == 3:
                r_obs = r_obs.reshape(B * R, -1)
            else:
                r_obs = r_obs.unsqueeze(1).repeat_interleave(R, dim=1)
                r_obs = r_obs.reshape(B * R, -1)
        else:
            B, P, D = obs.shape
            R = 1

        n = obs.shape[0]       
        p_num = obs.shape[1]    

        if r_obs.ndim < 3:
            r_obs = r_obs.reshape((n, 1, self.r_obs_dim))

        enc_r_obs = self.enc_r_obs(r_obs)
        enc_obs = self.enc_obs(obs.reshape((n, -1, self.obs_dim)))

        obs_stack = torch.cat(
            (enc_r_obs.reshape(n, -1, self.projection_dim), enc_obs), 1
        )

        A = torch.matmul(
            torch.matmul(obs_stack, self.w_a),
            obs_stack.permute(0, 2, 1),
        )
        adj_mat = softmax(A, dim=2)

        obs_gc = self.gcl1(obs_stack, adj_mat) + obs_stack
        obs_gc2 = self.gcl2(obs_gc, adj_mat) + obs_gc

        if self.prediction:
            integrated = obs_gc2[:, 1:, :]      # [B*R, P, D]
        else:
            integrated = obs_gc2[:, 0, :]       # [B*R, D]

        if R > 1:
            integrated = integrated.reshape(B, R, -1)

        return integrated
