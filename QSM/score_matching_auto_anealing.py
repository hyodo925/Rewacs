# Copyright 2022 Twitter, Inc and Zhendong Wang.
# SPDX-License-Identifier: Apache-2.0

import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR
#from utils.logger import logger
import random
from QSM.diffusion import Diffusion
from QSM.model import MLP
from QSM.helpers import EMA
import wandb
from tqdm import tqdm
from utils.models import MLPGraphConvEmbeddedGaussianIntegrator
import time
import math

def exponential_increase(t, p_init, p_target, T):
    if p_init == 0:
        raise ValueError("p_init cannot be zero for exponential increase.")
    return p_init * (p_target / p_init) ** (t / T)
def sigmoid_increase(t, p_init, p_target, T, k=10):
    x = (t - T / 2) / T
    return p_init + (p_target - p_init) / (1 + np.exp(-k * x))
class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256,integrator=None):
        super(Critic, self).__init__()
        self.integrator = integrator
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
        if self.integrator != None:
            obs = state[:,5:]
            r_obs = state[:,:5]
            state = self.integrator(obs,r_obs)
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
                 obs_dim,
                 r_obs_dim,
                 projection_dim,
                 max_action,
                 device,
                 tau,
                 memory,
                 beta_schedule='linear',
                 n_timesteps=100,
                 ema_decay=0.995,
                 step_start_ema=1000,
                 update_ema_every=5,
                 lr=1e-3,
                 lr_decay=False,
                 lr_maxt=1000,
                 grad_norm=1.0,
                 eval=False,
                 random_sample=True,
                 M=50,
                 gc=True,
                 k_sample=1000,
                 ):
        actor_integrator = MLPGraphConvEmbeddedGaussianIntegrator(
            obs_dim=obs_dim,
            r_obs_dim=r_obs_dim,
            projection_dim=projection_dim,
            enc_hdims=[64],
        )
        critic_integrator = MLPGraphConvEmbeddedGaussianIntegrator(
            obs_dim=obs_dim,
            r_obs_dim=r_obs_dim,
            projection_dim=projection_dim,
            enc_hdims=[64],
        )

        self.gc = gc
        self.lr_decay = lr_decay
        self.grad_norm = grad_norm
        self.step = 0
        self.step_start_ema = step_start_ema
        self.update_ema_every = update_ema_every
        if self.gc:
            self.critic = Critic(projection_dim, action_dim,integrator=critic_integrator).to(device)
            self.model = MLP(state_dim=projection_dim, action_dim=action_dim, device=device,integrator=actor_integrator)

        else:
            self.critic = Critic(state_dim, action_dim).to(device)
            self.model = MLP(state_dim=state_dim, action_dim=action_dim, device=device)
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=lr)
        self.actor = Diffusion(state_dim=state_dim, action_dim=action_dim, model=self.model, max_action=max_action,
                               beta_schedule=beta_schedule, n_timesteps=n_timesteps,random_sample=random_sample).to(device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr)
        self.ema = EMA(ema_decay)
        self.ema_model = copy.deepcopy(self.actor)
        print(lr_maxt)
        self.numsteps = lr_maxt
        if lr_decay:
            self.actor_lr_scheduler = CosineAnnealingLR(self.actor_optimizer, T_max=lr_maxt, eta_min=0.)
            self.critic_lr_scheduler = CosineAnnealingLR(self.critic_optimizer, T_max=lr_maxt, eta_min=0.)
        self.memory = memory
        self.state_dim = state_dim
        self.max_action = max_action
        self.action_dim = action_dim
        self.tau = tau
        self.device = device
        self.n_timesteps = n_timesteps
        self.M = nn.Parameter(torch.tensor(float(M), dtype=torch.float32, requires_grad=True, device=device))
        self.entropy_optimizer=torch.optim.Adam([self.M],lr=3e-3)
        self.k_sample = k_sample
        self.target_ent = -action_dim
        if not eval: 
            wandb.init(
            # set the wandb project where this run will be logged
                project="QSM-Auto-Annealing-epoch",
                config={
                "learning_rate": lr,
                "architecture": "MLP",
                "dataset": "Crowd_sim",
                "enviroment":"circle_crossing",
                "num_steps": n_timesteps,
                }
            )

    def update_entropy(self, state,global_steps=None, k_sample=None):
        """
        noisy_actionsに対して、log(sum(exp(Q))) の勾配を計算する関数

        Args:
            state (torch.Tensor): 状態 (batch_size, state_dim)
            noisy_actions (torch.Tensor): アクション (batch_size, action_dim)
            k_sample (int, optional): sampling number

        Returns:
            torch.Tensor: noisy_actionsに対する勾配 (batch_size, action_dim)
        """
        if k_sample is None:
            k_sample = self.k_sample
        action = self.actor(state)
        action = torch.clamp(action, min=-1.0, max=1.0)
        self.M = self.M.requires_grad_(True)

        B, action_dim = action.shape

        noise = torch.randn(k_sample, B, action_dim, device=action.device)
        noise = torch.clamp(noise, min=-1.0, max=1.0)  # アクション空間の範囲に収める
        noise_flat = noise.reshape(-1, action_dim)  

        state_expanded = state.unsqueeze(0).expand(k_sample, -1, -1).reshape(-1, state.shape[-1])
        qi_1, qi_2 = self.critic(state, action)
        Q = torch.min(qi_1, qi_2)

        qi_1_noise, qi_2_noise = self.critic(state_expanded, noise_flat)
        qi_noise = self.M *torch.min(qi_1_noise, qi_2_noise)
        logZ = torch.logsumexp(qi_noise.view(k_sample, B, 1), dim=0) - math.log(k_sample)+math.log(4.0)     # shape: [B, 1]
        pi_probs = torch.softmax(Q - logZ,dim=0)
        expected_Q = torch.sum(pi_probs * Q.squeeze(-1), dim=0, keepdim=True)  # (B, 1)

        entropy = 1/self.M *logZ - Q
        target = torch.ones_like(entropy) * 1/self.M * self.target_ent
        loss_ent = F.mse_loss(entropy, target)
        self.entropy_optimizer.zero_grad()
        loss_ent.backward()
        self.entropy_optimizer.step()
        wandb.log({"pi_probs": pi_probs.mean()},step=global_steps)
        wandb.log({"logZ": logZ.mean()}, step=global_steps)
        wandb.log({"Q": Q.mean()}, step=global_steps)
        wandb.log({"expected_Q":expected_Q.mean()},step=global_steps)
        wandb.log({"entropy": entropy.mean()},step=global_steps)
        wandb.log({"loss_ent": loss_ent},step=global_steps)
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
    
    def append_memory(self, state, action, reward, next_state, mask):
        self.memory.append(state, action, reward, next_state, mask)
    
    def step_ema(self):
        if self.step < self.step_start_ema:
            return
        self.ema.update_model_average(self.ema_model, self.actor)

    def train(self, iterations,batch_size=100,global_steps=None):
        for _ in range(iterations):
        # for state, action, reward, next_state, mask,done in data_loader:
            state, action, reward, next_state, mask = self.memory.sample(batch_size)
            state = state.to(self.device)
            next_state = next_state.to(self.device)
            action = (action.view(batch_size,-1)).to(self.device)
            """ Q Training """
            current_q1, current_q2 = self.critic(state, action)

            next_action = self.ema_model(next_state)
            next_action = torch.clamp(next_action, min=-1.0, max=1.0)
            target_q1, target_q2 = self.critic_target(next_state, next_action)
            target_q = torch.min(target_q1, target_q2)

            target_q = (reward.to(self.device) + mask.to(self.device) * target_q).detach()

            critic_loss = F.mse_loss(current_q1, target_q) + F.mse_loss(current_q2, target_q)
            wandb.log({"critic_loss": critic_loss},step=global_steps)
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            if self.grad_norm > 0:
                critic_grad_norms = nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=self.grad_norm, norm_type=2)
            self.critic_optimizer.step()

            """ Policy Training """
            batch_size = len(action)
            t = torch.randint(0, self.n_timesteps, (batch_size,), device=self.device).long()
            noise = torch.randn_like(action)
            noisy_actions = self.actor.q_sample(x_start=action, t=t, noise=noise)
            critic_jacobian=self.compute_jacobian(state,noisy_actions)
            coefficient = self.M.clone().detach()
            wandb.log({"coefficient":self.M.item()},step=global_steps)
            wandb.log({"critic_jacobian":coefficient.mean()},step=global_steps)
            loss = self.actor.p_losses(x_start=-coefficient*critic_jacobian, state=state, t=t, x_noisy=noisy_actions)
            self.actor_optimizer.zero_grad()
            loss.backward()
            if self.grad_norm > 0: 
                actor_grad_norms = nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=self.grad_norm, norm_type=2)
            self.actor_optimizer.step()
            wandb.log({"QSM_loss": loss},step=global_steps)
            self.update_entropy(state,global_steps=global_steps)

            """ Step Target network """
            if self.step % self.update_ema_every == 0:
                self.step_ema()

            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

            self.step += 1

        if self.lr_decay: 
            self.actor_lr_scheduler.step()
            self.critic_lr_scheduler.step()
            

    def sample_action(self, state,eval=False):
        state = torch.FloatTensor(state.reshape(1, -1)).to(self.device)

        # 処理の開始時刻を取得
        start_time = time.time()

        # ここに測りたい処理を書く
        with torch.no_grad():
            action = self.actor.sample(state)
        if not eval:
            action = action + torch.randn_like(action) * 0.1
        action = torch.clamp(action, min=-1.0, max=1.0)
        # 処理の終了時刻を取得
        end_time = time.time()

        # 処理時間を計算
        elapsed_time = end_time - start_time
        # print(action)
        return action.cpu().data.numpy().flatten(),elapsed_time#,action
    def output_entropy(self, action):
        return self.actor.entropy(action*-1)
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


