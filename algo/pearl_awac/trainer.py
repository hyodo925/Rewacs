import numpy as np

import torch
from torch import nn as nn
import torch.nn.functional as F
from collections import OrderedDict
import numpy as np
import copy
import torch
import torch.optim as optim
from torch import nn as nn

import torch
import numpy as np
import os


def soft_update_from_to(source, target, tau):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(
            target_param.data * (1.0 - tau) + param.data * tau
        )


def copy_model_params_from_to(source, target):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(param.data)


def fanin_init(tensor):
    size = tensor.size()
    if len(size) == 2:
        fan_in = size[0]
    elif len(size) > 2:
        fan_in = np.prod(size[1:])
    else:
        raise Exception("Shape must be have dimension at least 2.")
    bound = 1. / np.sqrt(fan_in)
    return tensor.data.uniform_(-bound, bound)


def fanin_init_weights_like(tensor):
    size = tensor.size()
    if len(size) == 2:
        fan_in = size[0]
    elif len(size) > 2:
        fan_in = np.prod(size[1:])
    else:
        raise Exception("Shape must be have dimension at least 2.")
    bound = 1. / np.sqrt(fan_in)
    new_tensor = FloatTensor(tensor.size())
    new_tensor.uniform_(-bound, bound)
    return new_tensor


def elem_or_tuple_to_variable(elem_or_tuple):
    if isinstance(elem_or_tuple, tuple):
        return tuple(
            elem_or_tuple_to_variable(e) for e in elem_or_tuple
        )
    return from_numpy(elem_or_tuple).float()


def filter_batch(np_batch):
    for k, v in np_batch.items():
        if v.dtype == np.bool:
            yield k, v.astype(int)
        else:
            yield k, v


def np_to_pytorch_batch(np_batch):
    return {
        k: elem_or_tuple_to_variable(x)
        for k, x in filter_batch(np_batch)
        if x.dtype != np.dtype('O')  # ignore object (e.g. dictionaries)
    }

"""
GPU wrappers
"""

_use_gpu = False
device = None


def set_gpu_mode(mode, gpu_id=0):
    global _use_gpu
    global device
    global _gpu_id
    _gpu_id = gpu_id
    _use_gpu = mode
    device = torch.device("cuda:0" if _use_gpu else "cpu")
    if _use_gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(_gpu_id)


def gpu_enabled():
    return _use_gpu


# noinspection PyPep8Naming
def FloatTensor(*args, **kwargs):
    return torch.FloatTensor(*args, **kwargs).to(device)


def from_numpy(*args, **kwargs):
    return torch.from_numpy(*args, **kwargs).float().to(device)


def get_numpy(tensor):
    # not sure if I should do detach or not here
    return tensor.to('cpu').detach().numpy()


def zeros(*sizes, **kwargs):
    return torch.zeros(*sizes, **kwargs).to(device)


def ones(*sizes, **kwargs):
    return torch.ones(*sizes, **kwargs).to(device)


def randn(*args, **kwargs):
    return torch.randn(*args, **kwargs).to(device)


def zeros_like(*args, **kwargs):
    return torch.zeros_like(*args, **kwargs).to(device)


def normal(*args, **kwargs):
    return torch.normal(*args, **kwargs).to(device)

def _product_of_gaussians(mus, sigmas_squared):
    '''
    compute mu, sigma of product of gaussians
    '''
    sigmas_squared = torch.clamp(sigmas_squared, min=1e-7)
    sigma_squared = 1. / torch.sum(torch.reciprocal(sigmas_squared), dim=0, keepdim=True)
    mu = sigma_squared * torch.sum(mus / sigmas_squared, dim=0, keepdim=True)
    return mu, sigma_squared


def _mean_of_gaussians(mus, sigmas_squared):
    '''
    compute mu, sigma of mean of gaussians
    '''
    mu = torch.mean(mus, dim=0)
    sigma_squared = torch.mean(sigmas_squared, dim=0)
    return mu, sigma_squared


def _natural_to_canonical(n1, n2):
    ''' convert from natural to canonical gaussian parameters '''
    mu = -0.5 * n1 / n2
    sigma_squared = -0.5 * 1 / n2
    return mu, sigma_squared


def _canonical_to_natural(mu, sigma_squared):
    ''' convert from canonical to natural gaussian parameters '''
    n1 = mu / sigma_squared
    n2 = -0.5 * 1 / sigma_squared
    return n1, n2
    
class PEARLAWAC(nn.Module):
    def __init__(
        self,
        model,
        tasks,
        actor_optimizer,
        critic_optimizer,
        context_optimizer,
        batch_size,
        polyak=0.995,
        gamma=0.9,
        beta=0.3,
        latent_dim=5,
    ):
        super().__init__() 
        self.alg_name = "PEARLAWAC"
        self.model = model
        self.target = copy.deepcopy(model)
        self.tasks = tasks
        self.actor_optimizer = actor_optimizer
        self.critic_optimizer = critic_optimizer
        self.context_optimizer = context_optimizer
        self.batch_size = batch_size
        self.polyak = polyak
        self.device = model.device
        self.gamma = torch.as_tensor([gamma]).to(self.device)
        self.beta = torch.as_tensor([beta]).to(self.device)

        self.use_ib = True
        self.latent_dim = latent_dim
        self.kl_lambda = 1.0

        self.register_buffer('z', torch.zeros(1, self.latent_dim))
        self.register_buffer('z_means', torch.zeros(1, self.latent_dim))
        self.register_buffer('z_vars', torch.zeros(1, self.latent_dim))

        self.clear_z()

    def clear_z(self, num_tasks=1):
        mu = zeros(num_tasks, self.latent_dim)
        if self.use_ib:
            var = ones(num_tasks, self.latent_dim)
        else:
            var = zeros(num_tasks, self.latent_dim)
        self.z_means = mu
        self.z_vars = var
        self.sample_z()
        self.context = None

    def compute_kl_div(self):
        prior = torch.distributions.Normal(zeros(self.latent_dim), ones(self.latent_dim))
        posteriors = [torch.distributions.Normal(mu, torch.sqrt(var)) for mu, var in zip(torch.unbind(self.z_means), torch.unbind(self.z_vars))]
        kl_divs = [torch.distributions.kl.kl_divergence(post, prior) for post in posteriors]
        kl_div_sum = torch.sum(torch.stack(kl_divs))
        return kl_div_sum

    def infer_posterior_single_task(self, obs, act, rwd):
        params =  self.model.context_encoder(
            obs,
            act.squeeze(1).to(self.device),
            rwd.to(self.device),
        )
        
        if self.use_ib:
            mu = params[..., :self.latent_dim]
            sigma_squared = F.softplus(params[..., self.latent_dim:])

            mu_task, var_task = _product_of_gaussians(mu, sigma_squared)
            
            self.z_means = mu_task.unsqueeze(0) # (1, latent_dim)
            self.z_vars = var_task.unsqueeze(0)  # (1, latent_dim)
        else:
            self.z_means = torch.mean(params, dim=1) # (1, latent_dim)

        self.sample_z()

    def sample_z(self):
        if self.use_ib:
            posteriors = [torch.distributions.Normal(m, torch.sqrt(s)) for m, s in zip(torch.unbind(self.z_means), torch.unbind(self.z_vars))]
            z = [d.rsample() for d in posteriors]
            self.z = torch.stack(z)
        else:
            self.z = self.z_means

    def safe_exp(self, x):
        return torch.where(x < 50, torch.exp(x), x)

    def update(self, data_for_logging=None):
        total_q_loss = 0
        total_actor_loss = 0
        total_kl_loss = 0
        lc = 0
        la = 0
        lkl = 0
        self.clear_z(num_tasks=len(self.tasks))

        for task in self.tasks:
            sample = task.sample(self.batch_size)
            obs, next_obs, r_obs, next_r_obs, act, rwd, done = list(sample.values())

            self.infer_posterior_single_task(
                (
                    obs.reshape(self.batch_size, -1).to(self.device),
                    r_obs.reshape(self.batch_size, -1).to(self.device),
                ),
                act.squeeze().to(self.device),
                rwd.to(self.device),
            )
            task_z = self.z
            task_z = [z.repeat(self.batch_size, 1) for z in task_z]
            task_z = torch.cat(task_z, dim=0)
        
            with torch.no_grad():
                next_act_target, next_log_prob, _ = self.model.actor.sample(
                    (
                        next_obs.to(self.device),
                        next_r_obs.reshape(self.batch_size, 1, -1).to(self.device),
                    ),
                    z=task_z
                )
                Q_target_1, Q_target_2 = self.target.critic(
                    (next_obs.to(self.device), next_r_obs.to(self.device)), next_act_target,z=task_z
                )
                Q_target_min = torch.min(torch.cat((Q_target_1, Q_target_2), 1), dim=1)[
                    0
                ].unsqueeze(-1)

                Q_target = rwd.to(self.device) + (self.gamma * Q_target_min) * done.to(
                    self.device
                )

            Q1, Q2 = self.model.critic(
                (obs.to(self.device), r_obs.to(self.device)),
                act.squeeze().to(self.device),
                z=task_z
            )

            loss_critic = F.mse_loss(Q_target, Q1) + F.mse_loss(Q_target, Q2)
            total_q_loss += loss_critic
            
            if self.use_ib:
                kl_div = self.compute_kl_div()
                kl_loss = self.kl_lambda * kl_div
                total_kl_loss += kl_loss
                
            

            if data_for_logging is not None:
                data_for_logging[0].log(
                    {
                        "loss/critic": lc,
                    },
                    step=data_for_logging[1],
                )

            with torch.no_grad():
                action_gen, log_prob, _ = self.model.actor.sample(
                    (
                        obs.to(self.device),
                        r_obs.reshape(self.batch_size, 1, -1).to(self.device),
                    ),
                    z=task_z.detach()
                )

                qw_ref = torch.min(torch.cat((Q1, Q2), 1), dim=1)[0].reshape((-1, 1))

                v_act1, v_act2 = self.model.critic(
                    (obs.to(self.device), r_obs.to(self.device)),
                    action_gen.detach(),
                    z=task_z.detach()
                )

                qw_gen = torch.min(torch.cat((v_act1, v_act2), 1), dim=1)[0].reshape(
                    (-1, 1)
                )

                adv = torch.max(torch.zeros_like(qw_ref), qw_ref - qw_gen)
                # adv = qw_ref - qw_gen
                # weights = F.softmax(adv / beta, dim=0)
                weights = self.safe_exp(adv / self.beta)

            loss_act = -(
                self.model.actor.get_log_prob(
                    (
                        obs.to(self.device),
                        r_obs.reshape(self.batch_size, 1, -1).to(self.device),
                    ),
                    act.squeeze().to(self.device),
                    z=task_z.detach(),
                )
                * weights
            ).mean()

            total_actor_loss += loss_act
                
        self.actor_optimizer.zero_grad()
        total_actor_loss.backward()
        self.actor_optimizer.step()
        la = total_kl_loss.data.item()

        self.critic_optimizer.zero_grad()
        self.context_optimizer.zero_grad()
        total_kl_loss.backward(retain_graph=True)
        total_q_loss.backward()
        self.critic_optimizer.step()
        self.context_optimizer.step()
        lc = total_q_loss.data.item()
        lkl = total_kl_loss.data.item()
        self.z.detach()

        if data_for_logging is not None:
            data_for_logging[0].log(
                {
                    "loss/actor": la,
                    "loss/critic": lc,
                    "loss/kl": lkl,
                },
                step=data_for_logging[1],
            )

    def update_target(self):
        for param, target_param in zip(
            self.model.parameters(), self.target.parameters()
        ):
            target_param.data.mul_(self.polyak)
            target_param.data.add_((1 - self.polyak) * param.data)
