import torch.nn as nn
import torch

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