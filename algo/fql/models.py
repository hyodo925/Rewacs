import torch as torch
import torch.nn as nn
from torch.distributions import Normal
from utils.layers import *
try:
    from torch.func import functional_call  # PyTorch ≥ 2.0
except ImportError:
    from torch.nn.utils.stateless import functional_call  # PyTorch 1.9 ～ 1.13

class BC_flow(nn.Module):
    def __init__(
        self, D, d, h_dims=[256], integrator=None, activation="leaky_relu",
    ):
        super().__init__()
        self.integrator = integrator
        self.net1 = self.make_mlp([D] + h_dims + [d], activation=activation)

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

    def forward(self, obs, t, x_t):
        # for name, param in self.net1.named_parameters():
        #     print(f"\n{name}: shape = {param.shape}")
        #     print(param.data)
        if self.integrator != None:
            data = self.integrator(*obs)
        data = torch.cat([data, t, x_t], 1)
        out1 = self.net1(data)
        return out1
    

class SocialActorFromNoise(nn.Module):
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

        self.net = self.make_mlp([D] + h_dims + [d], activation=activation)
        # self.net = self.make_mlp([D] + h_dims, activation=activation)
        # self.mean_linear = nn.Linear(h_dims[-1], d)
        #self.log_std_logits = nn.Parameter(torch.zeros(d, requires_grad=True))
        # self.log_std_logits = nn.Linear(h_dims[-1], d)

        self.act_min = action_space[0]
        self.act_max = action_space[1]

        self.log_std_max = log_std_max
        self.log_std_min = log_std_min
        self.normal_dis_info = []
        self.x_mean_info = []
        self.delta = 1e-7
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

    # def sample_one_step_action(self, data, noise, params=None):
    #     if self.integrator != None:
        
    #         data = self.integrator(*data)
    #     data = torch.cat([data, noise], 1)
    #     action = self.net(data)
    #     action = torch.clamp(action, -1, 1)
    #     return action
    
    def sample_one_step_action(self, data, noise, params=None):
        def extract_submodule_params(params, prefix):
            """'net.', 'integrator.' などで始まるパラメータを取り出してプレフィックスを外す"""
            return {
                k[len(prefix) + 1:]: v  # 'net.fc1.weight' → 'fc1.weight'
                for k, v in params.items()
                if k.startswith(prefix + '.')
            }
        if params is not None:
            # 仮パラメータで self.integrator を通す
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

        return torch.clamp(net_out, -1, 1)


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