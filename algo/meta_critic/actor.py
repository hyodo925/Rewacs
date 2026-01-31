import numpy as np
import torch as torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
try:
    from torch.func import functional_call  # PyTorch ≥ 2.0
except ImportError:
    from torch.nn.utils.stateless import functional_call  # PyTorch 1.9 ～ 1.13
from .utils import extend_and_repeat

class SocialActorMetaCriticAWAC(nn.Module):
    def __init__(
        self,
        D,
        d,
        h_dims=[256],
        action_space=[-1, 1],
        activation="leaky_relu",
        integrator=None,
        log_std_max=0,
        log_std_min=-2,
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
        # norm = torch.norm(x, dim=1, keepdim=True) 
        # x = x / norm            
        mean = self.mean_linear(x)

        log_std = torch.sigmoid(self.log_std_logits(x))

        log_std = self.log_std_min + log_std * (self.log_std_max - self.log_std_min)

        return mean, log_std, x

    def get_log_prob(self, data, action):
        if self.integrator != None:
            data = self.integrator(*data)
        x = self.net(data)
        mean = self.mean_linear(x)
        # mean = torch.tanh(mean) * self.act_max
        log_std = torch.sigmoid(self.log_std_logits(x))
        log_std = self.log_std_min + log_std * (self.log_std_max - self.log_std_min)
        std = torch.exp(log_std)
        dist = Normal(mean, std)
        log_prob = dist.log_prob(action).sum(axis=-1, keepdim=True)
        # log_prob -= (2 * (np.log(2) - action - F.softplus(-2 * action))).sum(
        #     axis=1, keepdim=True
        # )
        # log_prob = torch.clamp(log_prob, min=-100.0)
        return log_prob

    def sample(self, data):
        mean, log_std, x= self.forward(data)
        std = log_std.exp()
        # mean = torch.tanh(mean) * self.act_max
        dist = Normal(mean, std)
        action = dist.rsample()
        log_prob = dist.log_prob(action).sum(axis=-1, keepdim=True)
        log_prob -= (2 * (np.log(2) - action - F.softplus(-2 * action))).sum(
            axis=1, keepdim=True
        )
        # log_prob = torch.clamp(log_prob, min=-100.0)
        # action = torch.clamp(action, self.act_min, self.act_max)
        action = torch.tanh(action) * self.act_max
        # mean = torch.clamp(mean, self.act_min, self.act_max)
        return action, log_prob, mean, x
    

    def sample_with_params(self, data, params ):

        out = functional_call(self, params, (data,))
        mean, log_std, _ = out

        std = log_std.exp()
        normal = Normal(mean, std)
        pre_tanh_value = normal.rsample()
        
        log_prob = normal.log_prob(pre_tanh_value).sum(dim=-1, keepdim=True)
        log_prob -= (2 * (np.log(2) - pre_tanh_value - F.softplus(-2 * pre_tanh_value))).sum(
            axis=1, keepdim=True
        )
        # log_prob = torch.clamp(log_prob, min=-100.0)

        action = torch.tanh(pre_tanh_value) * self.act_max
        
        return action, log_prob, mean, std

    def get_log_prob_with_params(self, data, action, params):

        out = functional_call(self, params, (data,))
        mean, log_std, _ = out

        # log_std = self.log_std_min + log_std * (self.log_std_max - self.log_std_min)
        std = torch.exp(log_std) 
        dist = Normal(mean, std)
        log_prob = dist.log_prob(action).sum(axis=-1, keepdim=True)
        # log_prob -= (2 * (np.log(2) - action - F.softplus(-2 * action))).sum(
        #     axis=1, keepdim=True
        # )
        # log_prob = torch.clamp(log_prob, min=-100.0)

        return log_prob
class SocialActorMetaCriticCalQL(nn.Module):
    def __init__(
        self,
        D,
        d,
        h_dims=[256],
        action_space=[-1, 1],
        activation="leaky_relu",
        integrator=None,
        log_std_max=0,
        log_std_min=-2,
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

        return mean, log_std, x

    def sample(self, data, repeat: int = None):
        mean, log_std, x= self.forward(data, repeat=repeat)
        std = log_std.exp()

        mean = torch.tanh(mean) * self.act_max
        dist = Normal(mean, std)

        action = dist.rsample()
        log_prob = dist.log_prob(action).sum(axis=-1, keepdim=True)

        return action, log_prob, mean, x

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
    
    def sample_with_params(self, data, params ):

        out = functional_call(self, params, (data,))
        mean, log_std, _ = out

        std = log_std.exp()
        normal = Normal(mean, std)
        pre_tanh_value = normal.rsample()
        action = torch.tanh(pre_tanh_value) * self.act_max

        log_prob = normal.log_prob(pre_tanh_value).sum(dim=-1, keepdim=True)
        
        return action, log_prob, mean, std
    
    def get_log_prob_with_params(self, data, action, params):

        out = functional_call(self, params, (data,))
        mean, log_std, _ = out

        std = torch.exp(log_std)
        dist = Normal(mean, std)
        logp_prob = dist.log_prob(action).sum(axis=-1, keepdim=True)

        return logp_prob
    

class SocialActorMetaCriticFQL(nn.Module):
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

        self.net = self.make_mlp([D] + h_dims + [d], activation=activation, last_act=True)

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

        return mean, log_std, x

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

    def sample_one_step_action(self, data, noise, params=None):
        def extract_submodule_params(params, prefix):
            return {
                k[len(prefix) + 1:]: v  
                for k, v in params.items()
                if k.startswith(prefix + '.')
            }
        if params is not None:
            integrator_out = functional_call(self.integrator, extract_submodule_params(params, 'integrator'), data)
        else:
            if self.integrator is not None:
                integrator_out = self.integrator(*data)
            else:
                integrator_out = data
        input_tensor = torch.cat([integrator_out, noise], dim=1)

        if params is not None:
            net_out = functional_call(self.net, extract_submodule_params(params, 'net'), (input_tensor,))
        else:
            net_out = self.net(input_tensor)

        return torch.clamp(net_out, -1, 1), integrator_out


    # def sample_flow_step_action(self, data, bc_flow, noise):
    #     flow_steps = 10
    #     actions = noise
    #     for i in range(flow_steps):
    #         t = torch.full((noise.shape[0],1), i / flow_steps)
    #         vels = bc_flow(data, t, actions)
    #         actions = actions + vels / flow_steps
    #     actions = torch.clamp(actions, -1, 1)
    #     return actions

    
    def sample_flow_step_action(self, data, bc_flow, noise, bc_flow_params=None, flow_steps=10):
        actions = noise
        # list1 = []
        for i in range(flow_steps):
            t = torch.full((noise.shape[0], 1), i / flow_steps, device=noise.device)

            if bc_flow_params is not None:
                vels = functional_call(bc_flow, bc_flow_params, (data, t, actions))
            else:
                vels = bc_flow(data, t, actions)

            actions = actions + vels / flow_steps
            # list1.append(actions)

        return torch.clamp(actions, -1, 1)
        # return torch.cat(list1)