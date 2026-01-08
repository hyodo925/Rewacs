import torch as torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from torch.distributions.uniform import Uniform
from torch.nn.functional import softmax

from utils.graph import pos_to_graph
from utils.layers import *

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


class MLP(nn.Module):
    def __init__(self, D, d, h_size=256, last_relu=False, init_w=3e-3):
        super().__init__()

        self.model = nn.Sequential()
        self.model.add_module("fc1", nn.Linear(D, h_size))
        # self.model.add_module('ln1', nn.LayerNorm(500))
        self.model.add_module("act1", nn.ReLU())
        self.model.add_module("fc2", nn.Linear(h_size, h_size))
        # self.model.add_module('ln2', nn.LayerNorm(300))
        self.model.add_module("act2", nn.ReLU())
        self.model.add_module("fc3", nn.Linear(h_size, d))

        if last_relu:
            self.model.add_module("act3", nn.ReLU())

        # self.init_weights(init_w)

    def init_weights(self, init_w):
        self.model.fc1.weight.data = fanin_init(self.model.fc1.weight.data.size())
        self.model.fc2.weight.data = fanin_init(self.model.fc2.weight.data.size())
        self.model.fc3.weight.data.uniform_(-init_w, init_w)

    def forward(self, data):
        out = self.model(data)

        return out


class ObsEnc(nn.Module):
    def __init__(self, D, d, h_dims=[64], last_act=True):
        super().__init__()
        self.net = self.make_mlp([D] + h_dims + [d], last_act=last_act)
        # self.init_weights(init_w)

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

    def init_weights(self, init_w):
        self.model.fc1.weight.data = fanin_init(self.model.fc1.weight.data.size())
        self.model.fc2.weight.data = fanin_init(self.model.fc2.weight.data.size())
        self.model.fc3.weight.data.uniform_(-init_w, init_w)

    def forward(self, data):
        out = self.net(data)

        return out


class DoubleMLP(nn.Module):
    def __init__(self, D, d):
        super().__init__()
        self.mlp1 = MLP(D, d)
        self.mlp2 = MLP(D, d)

    def forward(self, data):
        out1 = self.mlp1(data)
        out2 = self.mlp2(data)

        return out1, out2


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
        if self.integrator != None:
            data = self.integrator(*obs)

        if not self.single:
            data = torch.cat([data, act], 1)

        out1 = self.net1(data)

        if not self.single:
            out2 = self.net2(data)
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


class SocialActorAWAC(nn.Module):
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
        # self.log_std_logits = nn.Parameter(torch.zeros(d, requires_grad=True))
        self.log_std_logits = nn.Linear(h_dims[-1], d)

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

    def forward(self, data):
        if self.integrator != None:
            data = self.integrator(*data)
        x = self.net(data)
        mean = self.mean_linear(x)

        log_std = torch.sigmoid(self.log_std_logits(x))

        log_std = self.log_std_min + log_std * (self.log_std_max - self.log_std_min)

        return mean, log_std

    def get_log_prob(self, data, action):
        if self.integrator != None:
            data = self.integrator(*data)
        x = self.net(data)
        mean = self.mean_linear(x)
        mean = torch.tanh(mean) * self.act_max
        log_std = torch.sigmoid(self.log_std_logits(x))
        log_std = self.log_std_min + log_std * (self.log_std_max - self.log_std_min)
        std = torch.exp(log_std)
        dist = Normal(mean, std)
        logp_prob = dist.log_prob(action).sum(axis=-1, keepdim=True)

        return logp_prob

    def sample(self, data):
        mean, log_std = self.forward(data)
        std = log_std.exp()
        mean = torch.tanh(mean) * self.act_max
        dist = Normal(mean, std)
        action = dist.rsample()
        log_prob = dist.log_prob(action).sum(axis=-1, keepdim=True)
        # action = torch.clamp(action, self.act_min, self.act_max)
        # action = torch.tanh(action) * self.act_max
        # mean = torch.clamp(mean, self.act_min, self.act_max)
        return action, log_prob, mean


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


class MLPEncGCObsIntegrator(nn.Module):
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


class MLPGraphConvEmbeddedGaussianIntegrator(nn.Module):
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


class MLPGATIntegrator(nn.Module):
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


class ConcatenateObsIntegrator(nn.Module):
    def __init__(self, obs_dim, r_obs_dim):
        super().__init__()
        self.obs_dim = obs_dim
        self.r_obs_dim = r_obs_dim

    def forward(self, obs, r_obs):
        n = obs.shape[0]
        if len(r_obs.shape) < 3:
            r_obs = r_obs.reshape((n, 1, self.obs_dim))

        integrated = torch.cat((r_obs.reshape((n, -1)), obs.reshape((n, -1))), 1)

        return integrated

    # def forward(self, state, obs, r_obs):

    #     enc_r_obs = self.enc_r_obs(r_obs)
    #     return enc_r_obs


class MLPEncConcatenateObsIntegrator(nn.Module):
    def __init__(self, obs_dim, r_obs_dim, projection_dim):
        super().__init__()
        self.enc_r_obs = MLP(r_obs_dim, projection_dim, last_relu=True)
        self.enc_obs = MLP(obs_dim, projection_dim, last_relu=True)
        self.output_dim = projection_dim

        self.projection_dim = projection_dim
        self.obs_dim = obs_dim
        self.r_obs_dim = r_obs_dim
        # self.register_parameter('w_a', nn.Parameter(torch.randn(projection_dim, projection_dim).detach()))

    def forward(self, obs, r_obs):
        n = obs.shape[0]

        p_num = obs.shape[1]

        enc_r_obs = self.enc_r_obs(r_obs)
        enc_obs = self.enc_obs(obs.reshape((n, -1, self.obs_dim)))
        integrated = torch.cat(
            (enc_r_obs.reshape(n, -1, self.projection_dim), enc_obs), 1
        )

        return integrated
