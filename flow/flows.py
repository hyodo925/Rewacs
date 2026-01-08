import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.distributions.normal import Normal
from torch.distributions.multivariate_normal import MultivariateNormal
from flow.utils import MultiInputSequential, create_coupling_blocks

# from flow.distributions import Normal

LOG_SIG_MAX = 2
LOG_SIG_MIN = -20
epsilon = 1e-6


def safe_log(z):
    return torch.log(z + 1e-7)


class RealNVP(nn.Module):
    def __init__(self, n_blocks, input_dim, h_dim, n_hidden, device="cpu"):
        super(RealNVP, self).__init__()
        self.n_blocks = n_blocks
        self.input_dim = input_dim
        self.device = device

        assert input_dim % 2 == 0 or n_blocks % 2 == 0
        self.s, self.t = create_coupling_blocks(
            input_dim, h_dim, n_blocks, n_hidden, layer_type="Linear"
        )
        mask = torch.arange(input_dim).float() % 2
        i_mask = 1 - mask
        mask = torch.stack([mask, i_mask]).repeat(int(n_blocks / 2), 1)
        self.mask = nn.Parameter(mask, requires_grad=False)

        self.base_dist = Normal(torch.zeros(input_dim), torch.ones(input_dim))

    def to(self, device):
        self.device = device
        return super().to(device)

    def inverse(self, z):
        log_det_J, x = z.new_zeros(z.shape[0]), z
        for i in range(0, self.n_blocks):
            x_ = x * self.mask[i]
            s = self.s[i](x_)
            t = self.t[i](x_)
            x = x_ + (1 - self.mask[i]) * (x * torch.exp(s) + t)
            log_det_J += ((1 - self.mask[i]) * s).sum(dim=1)  # log det dx/du
        return x, log_det_J

    def forward(self, x):
        log_det_J, z = x.new_zeros(x.shape[0]), x
        for i in reversed(range(0, self.n_blocks)):
            # z_ = self.mask[i].expand(z.shape[0], -1) * z
            z_ = self.mask[i] * z
            # z_ = torch.matmul(z, self.mask[i].T)
            s = self.s[i](z_)
            t = self.t[i](z_)
            z = (1 - self.mask[i]) * (z - t) * torch.exp(-s) + z_
            # z = torch.matmul(z - t, 1 - self.mask[i]) * torch.exp(-s) + z_
            log_det_J -= ((1 - self.mask[i]) * s).sum(dim=1)
        return z, log_det_J

    def log_prob(self, x):
        z, log_det_J = self.forward(x)
        # p_z = self.p_z(torch.zeros_like(x, device=self.device), torch.ones_like(x))
        log_p = -0.5 * torch.sum(z**2, dim=(1,))
        return log_p + log_det_J

    def sample(self, batchSize):
        z = self.base_dist.sample((batchSize, 1))
        logp = self.base_dist.log_prob(z)
        x = self.inverse(z)
        return x


class GrevNet(nn.Module):
    def __init__(self, n_blocks, input_dim, h_dim, n_hidden):
        super(GrevNet, self).__init__()
        self.n_blocks = n_blocks
        self.input_dim = input_dim
        # self.device = device

        assert input_dim % 2 == 0 or n_blocks % 2 == 0
        self.s, self.t = create_coupling_blocks(
            input_dim, h_dim, n_blocks, n_hidden, layer_type="GAT"
        )
        mask = torch.arange(input_dim).float() % 2
        i_mask = 1 - mask
        mask = torch.stack([mask, i_mask]).repeat(int(n_blocks / 2), 1)
        self.mask = nn.Parameter(mask, requires_grad=False)

        self.base_dist = Normal(torch.zeros(input_dim), torch.ones(input_dim))

    # def to(self, device):
    #     self.device = device
    #     return super().to(device)

    def inverse(self, z):
        log_det_J, x = z.new_zeros(z.shape[0], z.shape[1]), z
        for i in range(0, self.n_blocks):
            x_ = x * self.mask[i]
            s = self.s[i](x_)
            t = self.t[i](x_)
            x = x_ + (1 - self.mask[i]) * (x * torch.exp(s) + t)    
            log_det_J += ((1 - self.mask[i]) * s).sum(dim=(2,))  # log det dx/du
        return x, log_det_J

    def forward(self, x):
        log_det_J, z = x.new_zeros(x.shape[0], x.shape[1]), x
        for i in reversed(range(0, self.n_blocks)):
            # z_ = self.mask[i].expand(z.shape[0], -1) * z
            z_ = self.mask[i] * z
            # z_ = torch.matmul(z, self.mask[i].T)
            s = self.s[i](z_)
            t = self.t[i](z_)
            z = (1 - self.mask[i]) * (z - t) * torch.exp(-s) + z_
            # z = torch.matmul(z - t, 1 - self.mask[i]) * torch.exp(-s) + z_
            log_det_J -= ((1 - self.mask[i]) * s).sum(dim=(2,))
        return z, log_det_J

    def log_prob(self, x):
        z, log_det_J = self.inverse(x)
        # p_z = self.p_z(torch.zeros_like(x, device=self.device), torch.ones_like(x))
        log_p = torch.sum(-0.5 * torch.sum(z**2, dim=(2,)), dim=(1,))
        return log_p + log_det_J

    def sample(self, batchSize):
        z = self.base_dist.sample((batchSize, 1))
        logp = self.base_dist.log_prob(z)
        x = self.inverse(z)
        return x
