import numpy as np
import torch as torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

LOG_SIG_MAX = 2
LOG_SIG_MIN = -20
epsilon = 1e-6


def fanin_init(size, fanin=None):
    fanin = fanin or size[0]
    v = 1.0 / np.sqrt(fanin)
    return torch.Tensor(size).uniform_(-v, v)


def weights_init_(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight, gain=1)
        torch.nn.init.constant_(m.bias, 0)


class SimpleGC(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, X, adj_mat):
        X_gc = torch.matmul(adj_mat, X)

        return X_gc[:, 1:, :]


class SocialCritic(nn.Module):
    def __init__(
        self,
        D,
        d,
        h_dims=[256],
        aggregator=None,
        penultimate_norm=False,
        activation="leaky_relu",
        single=False,
    ):
        super().__init__()
        self.aggregator = aggregator
        self.single = single
        self.net1 = self.make_mlp([D] + h_dims, activation=activation, last_act=True)

        self.last_linear1 = nn.Linear(h_dims[-1], d)

        if not single:
            self.net2 = self.make_mlp(
                [D] + h_dims, activation=activation, last_act=True
            )
            self.last_linear2 = nn.Linear(h_dims[-1], d)

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

    def forward(self, obs, act=None):
        if self.aggregator is not None:
            data = self.aggregator(*obs)

        if not self.single:
            data = torch.cat([data, act], -1)

        phi1 = self.net1(data)
        if self.penultimate_norm:
            phi1 = phi1 / torch.norm(phi1, dim=1).view((-1, 1))

        out1 = self.last_linear1(phi1)

        if not self.single:
            phi2 = self.net2(data)
            if self.penultimate_norm:
                phi2 = phi2 / torch.norm(phi2, dim=1).view((-1, 1))
                out2 = self.last_linear1(phi2)
            return out1, out2
        else:
            return out1
        
class Actor(nn.Module):
    def __init__(self, D, d, init_w=3e-3):
        super().__init__()

        self.model = nn.Sequential()
        self.model.add_module("fc1", nn.Linear(D, 256))
        # self.model.add_module('ln1', nn.LayerNorm(500))
        self.model.add_module("act1", nn.LeakyReLU())
        self.model.add_module("fc2", nn.Linear(256, 256))
        # self.model.add_module('ln2', nn.LayerNorm(300))
        self.model.add_module("act2", nn.LeakyReLU())
        self.model.add_module("fc3", nn.Linear(256, d))

        # self.model.add_module('act3', nn.LeakyReLU())
        # self.model.add_module('fc4', nn.Linear(256, 256))

        # self.model.add_module('act4', nn.LeakyReLU())
        # self.model.add_module('fc5', nn.Linear(256, d))

        self.model.add_module("act3", nn.Tanh())

        self.init_weights(init_w)

    def init_weights(self, init_w):
        self.model.fc1.weight.data = fanin_init(self.model.fc1.weight.data.size())
        self.model.fc2.weight.data = fanin_init(self.model.fc2.weight.data.size())
        self.model.fc3.weight.data.uniform_(-init_w, init_w)

    def forward(self, data):
        out = self.model(data)

        return out


class SocialActor(nn.Module):
    def __init__(
        self,
        D,
        d,
        h_dims=[256],
        max_action=1.0,
        activation="leaky_relu",
        integrator=None,
    ):
        super().__init__()
        self.integrator = integrator

        self.net = self.make_mlp(
            [D] + h_dims + [d], activation=activation, last_act=True
        )

        self.max_action = max_action

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

    def forward(self, data):
        if self.integrator != None:
            data = self.integrator(*data)
        out = torch.tanh(self.net(data))

        return self.max_action * out


class SocialActorGaussian(nn.Module):
    def __init__(
        self,
        D,
        d,
        h_dims=[256],
        action_space=[-1, 1],
        activation="leaky_relu",
        integrator=None,
        log_sig_max=2,
        log_sig_min=-20,
    ):
        super().__init__()
        self.integrator = integrator

        self.net = self.make_mlp([D] + h_dims, activation=activation)

        self.mean_linear = nn.Linear(h_dims[-1], d)
        self.log_std_linear = nn.Linear(h_dims[-1], d)

        self.action_scale = torch.FloatTensor(
            [(action_space[1] - action_space[0]) / 2.0]
        )

        self.action_bias = torch.FloatTensor(
            [(action_space[1] + action_space[0]) / 2.0]
        )

        self.log_sig_max = log_sig_max
        self.log_sig_min = log_sig_min

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
        if self.integrator != None:
            data = self.integrator(*data)
        x = self.net(data)
        mean = self.mean_linear(x)
        log_std = self.log_std_linear(x)
        log_std = torch.clamp(log_std, min=self.log_sig_min, max=self.log_sig_max)

        return mean, log_std

    def sample(self, data):
        mean, log_std = self.forward(data)
        std = log_std.exp()
        normal = Normal(mean, std)
        x_t = normal.rsample()
        y_t = torch.tanh(x_t)
        action = y_t * self.action_scale + self.action_bias
        log_prob = normal.log_prob(x_t)

        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + epsilon)
        log_prob = log_prob.sum(1, keepdim=True)
        mean = torch.tanh(mean) * self.action_scale + self.action_bias
        return action, log_prob, mean

    def to(self, device):
        self.action_scale = self.action_scale.to(device)
        self.action_bias = self.action_bias.to(device)
        return super().to(device)


class SocialActorPerturbation(nn.Module):
    def __init__(
        self,
        D,
        d,
        h_dim=256,
        phi=0.05,
        max_action=1.0,
        max_latent_action=2.0,
        integrator=None,
    ):
        super().__init__()
        self.phi = phi
        self.integrator = integrator

        self.linear1 = nn.Linear(D, h_dim)
        self.linear2 = nn.Linear(h_dim, h_dim)
        self.linear3 = nn.Linear(h_dim, d)

        self.l4 = nn.Linear(D + d, h_dim)
        self.l5 = nn.Linear(h_dim, h_dim)
        self.l6 = nn.Linear(h_dim, d)
        # self.ln1 = nn.LayerNorm(h_dim)
        # self.ln2 = nn.LayerNorm(h_dim)
        self.max_action = max_action
        self.max_latent_action = max_latent_action

        # self.apply(weights_init_)

    def forward(self, data, flow, return_mid_action=False):
        if self.integrator != None:
            data = self.integrator(*data)
        # out = F.leaky_relu(self.ln1(self.linear1(data)))
        # out = F.leaky_relu(self.ln2(self.linear2(out)))
        latent_action = F.leaky_relu(self.linear1(data))
        latent_action = F.leaky_relu(self.linear2(latent_action))
        latent_action = self.max_latent_action * torch.tanh(self.linear3(latent_action))

        mid_action, _ = flow.inverse(
            latent_action.reshape((data.shape[0], latent_action.shape[-1])), data
        )

        a = F.leaky_relu(self.l4(torch.cat([data, mid_action], 1)))
        a = F.leaky_relu(self.l5(a))
        a = self.phi * torch.tanh(self.l6(a))
        final_action = (a + mid_action).clamp(-self.max_action, self.max_action)

        if return_mid_action:
            return final_action, mid_action
        else:
            return final_action




class SocialGaussianActor(nn.Module):
    def __init__(self, D, d, h_dim=256, integrator=None, action_space=None):
        super().__init__()
        self.integrator = integrator

        self.linear1 = nn.Linear(D, h_dim)
        self.linear2 = nn.Linear(h_dim, h_dim)

        self.mean_linear = nn.Linear(h_dim, d)
        self.log_std_linear = nn.Linear(h_dim, d)

        self.apply(weights_init_)

        # action rescaling
        if action_space is None:
            self.action_scale = torch.tensor(1.0)
            self.action_bias = torch.tensor(0.0)
        else:
            self.action_scale = torch.FloatTensor(
                (action_space.high - action_space.low) / 2.0
            )
            self.action_bias = torch.FloatTensor(
                (action_space.high + action_space.low) / 2.0
            )

    def forward(self, data):
        if self.integrator != None:
            data = self.integrator(*data)
        out = F.leaky_relu(self.linear1(data))
        out = F.leaky_relu(self.linear2(out))
        mean = self.mean_linear(out)
        log_std = self.log_std_linear(out)
        log_std = torch.clamp(log_std, min=LOG_SIG_MIN, max=LOG_SIG_MAX)
        return mean, log_std

    def sample(self, data):
        mean, log_std = self.forward(data)
        std = log_std.exp()
        normal = Normal(mean, std)
        x_t = normal.rsample()  # for reparameterization trick (mean + std * N(0,1))
        y_t = torch.tanh(x_t)
        action = y_t * self.action_scale + self.action_bias
        log_prob = normal.log_prob(x_t)
        # Enforcing Action Bound
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + epsilon)
        log_prob = log_prob.sum(1, keepdim=True)
        mean = torch.tanh(mean) * self.action_scale + self.action_bias
        return action, log_prob, mean

    def to(self, device):
        self.action_scale = self.action_scale.to(device)
        self.action_bias = self.action_bias.to(device)
        return super().to(device)


class Predictor(nn.Module):
    def __init__(self, D, d, h_dims=[64], integrator=None):
        super().__init__()
        self.integrator = integrator

        self.net = self.make_mlp([D] + h_dims + [d])

        # self.apply(weights_init_)

    def make_mlp(self, mlp_dims, activation="relu", last_act=False):
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

    def forward(self, data):
        if self.integrator != None:
            data = self.integrator(*data)
        out = self.net(data)

        return out