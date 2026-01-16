import torch as torch
import torch.nn as nn

def identity(x):
    return x

class ContextEncoder(nn.Module):
    def __init__(self, hidden_dim):
        super(ContextEncoder, self).__init__()
        self.fc1 = nn.Linear(hidden_dim,100)
        self.fc2 = nn.Linear(100,100)
        self.fc3 = nn.Linear(100,1)
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = nn.functional.softplus(self.fc3(x))
        #x = nn.functional.tanh(self.fc3(x))
        return torch.mean(x)
    
class ContextGraphEncoder(nn.Module):
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

    def forward(self, obs, act=None, reward=None):
        # with torch.no_grad():
        if self.integrator != None:
            data = self.integrator(*obs)
        data = torch.cat([data, act, reward], -1)
        x = self.net1(data)
        return x
    