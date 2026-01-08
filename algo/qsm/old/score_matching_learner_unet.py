# Copyright 2022 Twitter, Inc and Zhendong Wang.
# SPDX-License-Identifier: Apache-2.0

import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from diffusers.training_utils import EMAModel
from diffusers.optimization import get_scheduler
#from utils.logger import logger
from dipo.Unet import ConditionalUnet1D
import random
from qsm.diffusion import Diffusion
from qsm.helpers import EMA
import wandb
from tqdm import tqdm

class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super(Critic, self).__init__()
        self.q1_model = nn.Sequential(nn.Linear(state_dim + action_dim, hidden_dim),
                                      nn.Mish(),
                                      nn.Linear(hidden_dim, hidden_dim),
                                      nn.Mish(),
                                      nn.Linear(hidden_dim, hidden_dim),
                                      nn.Mish(),
                                      nn.Linear(hidden_dim, 1))

        self.q2_model = nn.Sequential(nn.Linear(state_dim + action_dim, hidden_dim),
                                      nn.Mish(),
                                      nn.Linear(hidden_dim, hidden_dim),
                                      nn.Mish(),
                                      nn.Linear(hidden_dim, hidden_dim),
                                      nn.Mish(),
                                      nn.Linear(hidden_dim, 1))

    def forward(self, state, action):
        x = torch.cat([state, action], dim=-1)
        return self.q1_model(x), self.q2_model(x)

    def q1(self, state, action):
        x = torch.cat([state, action], dim=-1)
        return self.q1_model(x)

    def q_min(self, state, action):
        q1, q2 = self.forward(state, action)
        return torch.min(q1, q2)

class ScoreMatchingLearner(object):
    def __init__(self,
                 state_dim,
                 action_dim,
                 max_action,
                 obs_horizon,
                 action_horizon,
                 pred_horizon,
                 device,
                 discount,
                 tau,
                 memory,
                 max_q_backup=False,
                 n_timesteps=100,
                 ema_decay=0.995,
                 step_start_ema=1000,
                 update_ema_every=5,
                 lr=3e-4,
                 lr_decay=False,
                 num_steps=1000,
                 grad_norm=1.0,
                 eval=False
                 ):

        self.actor = ConditionalUnet1D(input_dim=action_dim,global_cond_dim=state_dim*obs_horizon).to(device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=1e-4, weight_decay=1e-6)
        self.lr_scheduler = get_scheduler(
            name='cosine',
            optimizer=self.actor_optimizer,
            num_warmup_steps=500,
            num_training_steps=num_steps
        )
        self.lr_decay = lr_decay
        self.grad_norm = grad_norm

        self.step = 0
        self.step_start_ema = step_start_ema
        self.ema = EMA(ema_decay)
        self.ema_model = copy.deepcopy(self.actor)
        self.actor_target = copy.deepcopy(self.actor).to(device)

        self.update_ema_every = update_ema_every

        self.obs_horizon = obs_horizon
        self.action_horizon = action_horizon
        self.pred_horizon = pred_horizon

        self.ema = EMAModel(
            parameters=self.actor.parameters(),
            power=0.75)

        self.critic = Critic(state_dim, action_dim).to(device)
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=3e-4)

        if lr_decay:
            self.critic_lr_scheduler = CosineAnnealingLR(self.critic_optimizer, T_max=lr_maxt, eta_min=0.)
        self.memory = memory
        self.state_dim = state_dim
        self.max_action = max_action
        self.action_dim = action_dim
        self.discount = discount
        self.tau = tau
        self.device = device
        self.max_q_backup = max_q_backup
        self.n_timesteps = n_timesteps
        if not eval: 
            wandb.init(
            # set the wandb project where this run will be logged
                project="Q-Scorematching_unet",
                config={
                "learning_rate": 3e-4,
                "architecture": "Unet",
                "dataset": "Crowd_sim",
                "enviroment":"circle_crossing",
                "num_steps": n_timesteps,
                }
            )

    def compute_jacobian(self, state, actions):
        actions.requires_grad = True
        value1,value2 = self.critic(state, actions)
        value1 = value1.sum()
        value2 = value2.sum()
        grad_value1 = torch.ones_like(value1)  # 同じ形のテンソルを生成
        value1_jacobian = torch.autograd.grad(value1, actions, grad_outputs=grad_value1, create_graph=True)[0]
        grad_value2 = torch.ones_like(value2)  # 同じ形のテンソルを生成
        value2_jacobian = torch.autograd.grad(value2, actions, grad_outputs=grad_value2, create_graph=True)[0]
        # NumPy配列を使って平均を計算
        critic_jacobian = torch.mean(torch.stack([value1_jacobian, value2_jacobian]), dim=0)
        return critic_jacobian
    def append_memory(self, state, action, reward, next_state, mask,done):
        self.memory.append(state, action, reward, next_state, mask,done)

    def train(self, nbatch,noise_scheduler):
        # Sample replay buffer / batch
        states = nbatch['state'].to(self.device)
        actions = nbatch['action'].to(self.device)
        batch_size = states.shape[0]
        next_state = nbatch['next_state'].to(self.device)
        rewards = nbatch['reward'].to(self.device)
        masks = nbatch['mask'].to(self.device)
        obs_cond = states[:,:self.obs_horizon,:]
        obs_cond = obs_cond.flatten(start_dim=1)
        """ Q Training """
        next_actions = torch.randn((batch_size, self.pred_horizon, 2), device=self.device)
        for k in noise_scheduler.timesteps:
        # predict noise
            next_actions = self.actor_target(sample=next_actions,timestep=k,global_cond=obs_cond)
            # inverse diffusion step (remove noise)
            next_actions = noise_scheduler.step(model_output=noise_pred,timestep=k,sample=next_actions).prev_sample
        for ii in range(self.obs_horizon):
            current_q1, current_q2 = self.critic(states[:,ii,:], actions[:,ii,:])
            #noise_scheduler.set_timesteps(num_diffusion_iters)
            target_q1, target_q2 = self.critic_target(next_state[:,ii,:],next_actions[:,self.obs_horizon - 1 + ii,:])
            target_q = torch.min(target_q1, target_q2)
            target_q = (rewards[:,ii,:] + masks[:,ii,:] * target_q).detach()
            critic_loss = nn.functional.mse_loss(current_q1, target_q) + nn.functional.mse_loss(current_q2, target_q)
            wandb.log({"critic_loss": critic_loss})
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()
        """ Policy Training """
        batch_size = len(actions)
        timesteps = torch.randint(
            0, noise_scheduler.config.num_train_timesteps,
            (actions.shape[0],), device=self.device
        ).long()        
        noise = torch.randn(actions.shape, device=self.device)
        noisy_actions = noise_scheduler.add_noise(actions, noise, timesteps)

        for ii in range(self.obs_horizon):
            action_norms = torch.norm(actions[:,ii,:], dim=1)
            critic_jacobian = self.compute_jacobian(states[:,ii,:], noisy_actions[:,self.obs_horizon - 1 + ii,:].clone())
            jacobian_norms = torch.norm(critic_jacobian,dim=1)
            coefficient = (action_norms/jacobian_norms).unsqueeze(1) 
            # 勾配の形状を確認
            pred_critic_jacobian = self.actor(noisy_actions,timesteps, global_cond=obs_cond)
            actor_loss = nn.functional.mse_loss(pred_critic_jacobian[:,self.obs_horizon - 1 + ii,:], -coefficient*critic_jacobian)
            wandb.log({"QSM_loss": actor_loss})
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            if self.grad_norm > 0: 
                actor_grad_norms = nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=self.grad_norm, norm_type=2)
            self.actor_optimizer.step()
            self.lr_scheduler.step()
            self.ema.step(self.actor.parameters())

        """ Step Target network """
        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        if self.lr_decay: 
            self.critic_lr_scheduler.step()
        self.step += 1

    def sample_action(self, obs_cond,noise_scheduler,random_sample=True):
        noisy_action = torch.randn((1, self.pred_horizon, 2), device=self.device)
        if not random_sample:
            noisy_action = torch.zeros((1, self.pred_horizon, 2), device=self.device)
        naction = noisy_action
        # sample a diffusion iteration for each data point
        for k in noise_scheduler.timesteps:
            naction = self.actor(sample=naction,timestep=k,global_cond=obs_cond)
            # inverse diffusion step (remove noise)
            naction = noise_scheduler.step(model_output=noise_pred,timestep=k,sample=naction).prev_sample
        if random_sample:
            naction = naction + torch.randn_like(naction) * 0.1
        naction = torch.clamp(naction, min=-1.0, max=1.0)
        return naction
    def save_model(self, dir, id=None):
        if id is not None:
            torch.save(self.actor.state_dict(), f'{dir}/actor_{id}.pth')
            torch.save(self.critic.state_dict(), f'{dir}/critic_{id}.pth')
        else:
            torch.save(self.actor.state_dict(), f'{dir}/actor.pth')
            torch.save(self.critic.state_dict(), f'{dir}/critic.pth')

    def load_model(self, dir, id=None):
        if id is not None:
            self.actor.load_state_dict(torch.load(f'{dir}/actor_{id}.pth'))
            self.critic.load_state_dict(torch.load(f'{dir}/critic_{id}.pth'))
        else:
            self.actor.load_state_dict(torch.load(f'{dir}/actor.pth'))
            self.critic.load_state_dict(torch.load(f'{dir}/critic.pth'))


