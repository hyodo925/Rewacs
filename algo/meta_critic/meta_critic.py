import torch as torch
import torch.nn as nn

class MetaCriticNet(nn.Module):
    def __init__(self, hidden_dim):
        super(MetaCriticNet, self).__init__()
        self.fc1 = nn.Linear(hidden_dim,100)
        self.fc2 = nn.Linear(100,100)
        self.fc3 = nn.Linear(100,1)
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = nn.functional.softplus(self.fc3(x))
        #x = nn.functional.tanh(self.fc3(x))
        return torch.mean(x)
    
class MetaCriticGraphNet(nn.Module):
    def __init__(
        self, D, d, h_dims=[256], integrator=None, activation="leaky_relu", single=False
    ):
        super().__init__()
        self.integrator = integrator
        self.single = single
        self.net1 = self.make_mlp([D] + h_dims + [d], activation=activation)

        if not single:
            self.net2 = self.make_mlp([D] + h_dims + [d], activation=activation)

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

    def forward(self, obs, act=None, other_output=None, z=None, integrator=None):
        with torch.no_grad():
        # if self.integrator != None:
            data = integrator(*obs)
        # if z is not None and isinstance(z, torch.Tensor) and z.numel() > 0:
        #     # batch_size = z.size(0)
        #     # z_flattened = z.view(batch_size, -1)
        #     z_score = torch.sum(0.5 * torch.sum(z**2, dim=(2,)), dim=(1,))
        #     data = torch.cat([data, act, z_score.unsqueeze(1)], 1)
        #     # data = torch.cat([data, z_score.unsqueeze(1)], 1)
        # else:
        data = torch.cat([data, act, other_output], 1)

        out1 = self.net1(data)
        return torch.mean(out1)
    