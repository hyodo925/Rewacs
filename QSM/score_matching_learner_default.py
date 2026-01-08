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
class CoefficientMLP(nn.Module):
    def __init__(self, action_dim, hidden_dim=256):
        super(CoefficientMLP, self).__init__()
        self.co_model = nn.Sequential(nn.Linear(action_dim + action_dim, hidden_dim),
                                      nn.Mish(),
                                      nn.Linear(hidden_dim, hidden_dim),
                                      nn.Mish(),
                                      nn.Linear(hidden_dim, hidden_dim),
                                      nn.Mish(),
                                      nn.Linear(hidden_dim, 1))

    def forward(self, critic_jacobian,action):
        x = torch.cat([critic_jacobian, action], dim=-1)
        x = self.co_model(x)  # 出力は係数
        return x
class ScoreMatchingLearner(object):
    def __init__(self,
                 state_dim,
                 action_dim,
                 max_action,
                 device,
                 discount,
                 tau,
                 memory,
                 max_q_backup=False,
                 beta_schedule='linear',
                 n_timesteps=100,
                 ema_decay=0.995,
                 step_start_ema=1000,
                 update_ema_every=5,
                 lr=3e-4,
                 lr_decay=False,
                 lr_maxt=1000,
                 grad_norm=1.0,
                 use_coefficient=False,
                 eval=False
                 ):

        self.model = MLP(state_dim=state_dim, action_dim=action_dim, device=device)

        self.actor = Diffusion(state_dim=state_dim, action_dim=action_dim, model=self.model, max_action=max_action,
                               beta_schedule=beta_schedule, n_timesteps=n_timesteps,).to(device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=3e-4)
        if use_coefficient:
            self.coefficient = CoefficientMLP(action_dim=action_dim).to(device)
            self.coefficient_optimizer = torch.optim.Adam(self.coefficient.parameters(), lr=3e-4)

        self.lr_decay = lr_decay
        self.grad_norm = grad_norm
        self.use_coefficient = use_coefficient
        self.step = 0
        self.step_start_ema = step_start_ema
        self.ema = EMA(ema_decay)
        self.ema_model = copy.deepcopy(self.actor)
        self.update_ema_every = update_ema_every

        self.critic = Critic(state_dim, action_dim).to(device)
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=3e-4)

        if lr_decay:
            self.actor_lr_scheduler = CosineAnnealingLR(self.actor_optimizer, T_max=lr_maxt, eta_min=0.)
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
                project="Q-Scorematching",
                config={
                "learning_rate": 3e-4,
                "architecture": "MLP",
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
    def append_memory(self, state, action, reward, next_state, mask):
        self.memory.append(state, action, reward, next_state, mask)
    def step_ema(self):
        if self.step < self.step_start_ema:
            return
        self.ema.update_model_average(self.ema_model, self.actor)

    def train(self, iterations, batch_size=100,imitation=False):
        for _ in range(iterations):
            obs, action, reward, next_state, mask = self.memory.sample(batch_size)
            # obs_flat と r_obs を結合して次の状態を作成
            state = obs.to(self.device)
            next_state = next_state.to(self.device)  # dim=1 で結合
            #state, action, next_state, reward, not_done = replay_buffer.sample(batch_size)
            action = (action.view(100,-1)).to(self.device)
            """ Q Training """
            current_q1, current_q2 = self.critic(state, action)

            next_action = self.ema_model(next_state)
            target_q1, target_q2 = self.critic_target(next_state, next_action)
            target_q = torch.min(target_q1, target_q2)

            target_q = (reward.to(self.device) + mask.to(self.device) * target_q).detach()

            critic_loss = F.mse_loss(current_q1, target_q) + F.mse_loss(current_q2, target_q)
            wandb.log({"critic_loss": critic_loss})
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
            
            critic_jacobian = self.compute_jacobian(state, noisy_actions.clone())
            y_pred = critic_jacobian
            # 勾配の形状を確認
            if self.use_coefficient:
                coefficient = self.coefficient(y_pred,action)
                pred_action = coefficient*y_pred
                action_norms = torch.norm(action, dim=1)  
                pred_action_norms = torch.norm(pred_action,dim=1)
                coefficient_loss = nn.functional.mse_loss(pred_action_norms, action_norms)
                wandb.log({"Coefficient_loss": coefficient_loss})
                self.coefficient_optimizer.zero_grad()
                coefficient_loss.backward()
                self.coefficient_optimizer.step()

            else:
                action_norms = torch.norm(action, dim=1)
                jacobian_norms = torch.norm(critic_jacobian,dim=1)
                coefficient = (action_norms/jacobian_norms).unsqueeze(1) 
                

            critic_jacobian = self.compute_jacobian(state, noisy_actions.clone())
            y_pred = critic_jacobian
            # 勾配の形状を確認
            if self.use_coefficient:
                coefficient = self.coefficient(y_pred,action)

            loss = self.actor.p_losses(x_start=-coefficient*critic_jacobian, state=state, t=t, x_noisy=noisy_actions)
            self.actor_optimizer.zero_grad()
            loss.backward()
            if self.grad_norm > 0: 
                actor_grad_norms = nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=self.grad_norm, norm_type=2)
            self.actor_optimizer.step()
            wandb.log({"QSM_loss": loss})

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
        with torch.no_grad():
            action = self.actor.sample(state)
        if not eval:
            action = action + torch.randn_like(action) * 0.1
        action = torch.clamp(action, min=-1.0, max=1.0)
        return action.cpu().data.numpy().flatten()
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


