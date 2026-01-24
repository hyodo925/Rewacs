import argparse
from copy import deepcopy
from typing import List, Optional
import os
import itertools
import math
import random
import time
import json
import pickle
from collections import defaultdict
import warnings
import copy
import higher
import numpy as np
import torch
import torch.autograd as A
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as O
import torch.distributions as D


class MACAW:
    def __init__(
        self,
        model,
        tasks,
        actor_optimizer,
        critic_optimizer,
        value_optimizer,
        batch_size,
        num_tasks=5,
        polyak=0.995,
        gamma=0.9,
        beta=0.3,
    ):
        self.alg_name = "MACAW"
        self.model = model
        self.target = copy.deepcopy(model)
        self.tasks = tasks
        self.actor_optimizer = actor_optimizer
        self.critic_optimizer = critic_optimizer
        self.value_optimizer = value_optimizer
        self.batch_size = batch_size
        self.polyak = polyak
        self.gamma = torch.as_tensor([gamma])
        self.beta = torch.as_tensor([beta])

        self.device = model.device
        self.num_tasks = num_tasks

        self.q = True
        self.huber = True
        self.log_targets = True
        self.no_norm = True
        self.no_bootstrap = True


        self._policy_lrs = None
        self._value_lrs = None
        self._q_lrs = None
        self._adv_coef = None
        # self.advantage_head_coef = None
        self.advantage_head_coef = 0.01


        self.exp_advantage_clip = 20.0
        self._advantage_clamp = np.log(self.exp_advantage_clip)
        self._action_sigma = 0.2
        self._grad_clip = 1e2
        self.maml_steps = 1
        self.value_reg = 0
        self.inner_policy_lr = 0.001
        self.inner_value_lr = 0.001
        self.lrlr = 3e-4
        self.target_vf_alpha = 0.9
        self.adaptation_temperature = 1

        if self._policy_lrs is None:
            self._policy_lrs = [torch.nn.Parameter(torch.tensor(float(np.log(self.inner_policy_lr))).to(self.device))
                                for p in self.model.actor.parameters()]
            self._value_lrs = [torch.nn.Parameter(torch.tensor(float(np.log(self.inner_value_lr))).to(self.device))
                               for p in self.model.value.parameters()]
            self._q_lrs = [torch.nn.Parameter(torch.tensor(float(np.log(self.inner_value_lr))).to(self.device))
                               for p in self.model.critic.parameters()]
        if self.advantage_head_coef is not None:
            self._adv_coef = torch.nn.Parameter(torch.tensor(float(np.log(self.advantage_head_coef))).to(self.device))
                                                                 
        self._policy_lr_optimizer = O.Adam(self._policy_lrs, lr=self.lrlr)
        self._value_lr_optimizer = O.Adam(self._value_lrs, lr=self.lrlr)
        self._q_lr_optimizer = O.Adam(self._q_lrs, lr=self.lrlr)
        if self.advantage_head_coef is not None:
            self._adv_coef_optimizer = O.Adam([self._adv_coef], lr=self.lrlr)

    def safe_exp(self, x):
        return torch.where(x < 50, torch.exp(x), x)

    # def add_task_description(self, task_idx: int):

    #     idx = torch.zeros((self.batch_size, self.num_tasks)).to(self.device)
    #     if task_idx is not None:
    #         idx[:, task_idx] = 1
    #     return idx

    #@profile
    def q_function_loss_on_batch(self, q_function, obs, r_obs, act, mc_returns, inner: bool = False, task_idx: int = None):
        # q_estimates = q_function((obs.to(self.device),r_obs.reshape(self.batch_size, 1, -1).to(self.device),), act, self.add_task_description(task_idx))
        q_1, q_2 = q_function((obs.to(self.device),r_obs.reshape(self.batch_size, 1, -1).to(self.device),), act.squeeze().to(self.device),)
        q_estimates = torch.min(torch.cat((q_1, q_2), 1), dim=1)[0].reshape((-1, 1))
        with torch.no_grad():
            mc_value_estimates = mc_returns.to(self.device)

        return (q_estimates - mc_value_estimates).pow(2).mean()

    #@profile
    def value_function_loss_on_batch(self, value_function, obs, r_obs, mc_returns, inner: bool = False, task_idx: int = None, iweights: torch.tensor = None, target = None):
        # value_estimates = value_function((obs.to(self.device),r_obs.reshape(self.batch_size, 1, -1).to(self.device),), act=None, task_idx=self.add_task_description(task_idx))
        value_estimates = value_function((obs.to(self.device),r_obs.reshape(self.batch_size, 1, -1).to(self.device),), act=None )
        with torch.no_grad():
            if target is None:
                target = value_function
            mc_value_estimates = mc_returns.to(self.device)

            targets = mc_value_estimates
            if self.log_targets:
                targets[torch.logical_and(targets > -1, targets < 1)] = 0
                targets[targets > 1] = targets[targets>1].log()
                targets[targets < -1] = -targets[targets<-1].abs().log()
                targets = targets.clone()

        if self.huber and not inner:
            losses = F.smooth_l1_loss(value_estimates, targets, reduction='none')
        else:
            losses = (value_estimates - targets).pow(2)

        return losses.mean(), value_estimates.mean(), mc_value_estimates.mean(), mc_value_estimates.std()

    #@profile
    def adaptation_policy_loss_on_batch(self, policy, q_function, value_function, obs, r_obs, act, mc_returns, task_idx: int, inner: bool = False,
                                        iweights: torch.tensor = None, online: bool = False):
        with torch.no_grad():
            # value_estimates = value_function((obs.to(self.device),r_obs.reshape(self.batch_size, 1, -1).to(self.device),), act=None, task_idx=self.add_task_description(task_idx))
            value_estimates = value_function((obs.to(self.device),r_obs.reshape(self.batch_size, 1, -1).to(self.device),), act=None)
            if q_function is not None:
                q_1, q_2 = q_function((obs.to(self.device), r_obs.to(self.device)),act.squeeze().to(self.device),)
                action_value_estimates = torch.min(torch.cat((q_1, q_2), 1), dim=1)[0].reshape((-1, 1))
            else:
                action_value_estimates = mc_returns.to(self.device)

            advantages = (action_value_estimates - value_estimates).squeeze(-1)
            if self.no_norm:
                weights = advantages.clamp(min=-self._advantage_clamp, max=self._advantage_clamp).exp()
            else:
                normalized_advantages = (1 / self.adaptation_temperature) * (advantages - advantages.mean()) / advantages.std()
                weights = normalized_advantages.clamp(max=self._advantage_clamp).exp()

        if self.advantage_head_coef is not None:
            # action_mu, advantage_prediction = policy((obs.to(self.device),r_obs.reshape(self.batch_size, 1, -1).to(self.device),), act.squeeze().to(self.device), self.add_task_description(task_idx))
            action_mu, advantage_prediction = policy((obs.to(self.device),r_obs.reshape(self.batch_size, 1, -1).to(self.device),), act.squeeze().to(self.device))
        else:
            # action_mu = policy((obs.to(self.device),r_obs.reshape(self.batch_size, 1, -1).to(self.device),), act=None, task_idx=self.add_task_description(task_idx))
            action_mu, _= policy((obs.to(self.device),r_obs.reshape(self.batch_size, 1, -1).to(self.device),), action=None)
        action_sigma = torch.empty_like(action_mu).fill_(self._action_sigma)
        action_distribution = D.Normal(action_mu, action_sigma)
        action_log_probs = action_distribution.log_prob(act.squeeze().to(self.device)).sum(-1)
        # action_log_probs = policy.get_log_prob((obs.to(self.device),r_obs.reshape(self.batch_size, 1, -1).to(self.device),),act.squeeze().to(self.device),)
        losses = -(action_log_probs * weights)

        if iweights is not None:
            losses = losses * iweights
        
        adv_prediction_loss = None
        if inner:
            if self.advantage_head_coef is not None:
                adv_prediction_loss = F.softplus(self._adv_coef) *  (advantage_prediction.squeeze() - advantages) ** 2
                losses = losses + adv_prediction_loss
                adv_prediction_loss = adv_prediction_loss.mean()

        return losses.mean(), advantages.mean(), weights, adv_prediction_loss

    def update_model(self, model: nn.Module, optimizer: torch.optim.Optimizer, clip: float = None, extra_grad: list = None):
        if clip is not None:
            grad = torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        else:
            grad = None
            
        optimizer.step()
        optimizer.zero_grad()
        
        return grad

    def update_params(self, params: list, optimizer: torch.optim.Optimizer, clip: float = None, extra_grad: list = None):
        optimizer.step()
        optimizer.zero_grad()

    def soft_update(self, source, target):
        for param_source, param_target in zip(source.named_parameters(), target.named_parameters()):
            assert param_source[0] == param_target[0]
            param_target[1].data = self.target_vf_alpha * param_target[1].data + (1 - self.target_vf_alpha) * param_source[1].data


    def update(self, data_for_logging=None):
        for train_task_idx, task in enumerate(self.tasks):
            sample = task.sample(self.batch_size)
            sample_val = task.sample(self.batch_size)           
            # self._env.set_task_idx(train_task_idx)
            obs, next_obs, r_obs, next_r_obs, act, rwd, mc_returns, done = list(sample.values())
            obs_val, next_obs_val, r_obs_val, next_r_obs_val, act_val, rwd_val, mc_returns_val, done_val = list(sample_val.values())

            inner_value_losses = []
            meta_value_losses = []
            inner_q_losses = []
            meta_q_losses = []
            inner_policy_losses = []
            adv_policy_losses = []
            meta_policy_losses = []
            # value_lr_grads = []
            # policy_lr_grads = []
            inner_mc_means, inner_mc_stds = [], []
            outer_mc_means, outer_mc_stds = [], []
            inner_values, outer_values = [], []
            inner_qs, outer_qs = [], []
            inner_weights, outer_weights = [], []
            inner_advantages, outer_advantages = [], []
            
            # iweights_ = None
            # iweights_no_action_ = None

            ##################################################################################################
            # Adapt value function and collect meta-gradients
            ##################################################################################################
            vf = self.model.value
            # vf.train()
            vf_target = deepcopy(vf)
            # opt = O.SGD([{'params': p, 'lr': None} for p in vf.adaptation_parameters()])
            # opt = O.SGD([{'params': p, 'lr': None} for p in vf.parameters()])
            opt = torch.optim.SGD(self.model.value.parameters(), lr=self.inner_value_lr)
            # with higher.innerloop_ctx(vf, opt, override={'lr': [F.softplus(l) for l in self._value_lrs]}, copy_initial_weights=False) as (f_value_function, diff_value_opt):
            with higher.innerloop_ctx(vf, opt, copy_initial_weights=False) as (f_value_function, diff_value_opt):
                if self.num_tasks > 1:
                    for step in range(self.maml_steps):
                        # sub_batch = value_batch.view(self._args.maml_steps, value_batch.shape[0] // self._args.maml_steps, *value_batch.shape[1:])[step]
                        loss, value_inner, mc_inner, mc_std_inner = self.value_function_loss_on_batch(f_value_function, obs, r_obs, mc_returns, inner=True, task_idx=train_task_idx, target=vf_target)#, iweights=iweights_no_action_)

                        inner_values.append(value_inner.item())
                        inner_mc_means.append(mc_inner.item())
                        inner_mc_stds.append(mc_std_inner.item())
                        diff_value_opt.step(loss)
                        inner_value_losses.append(loss.item())

                        # Soft update target value function parameters
                        self.soft_update(f_value_function, vf_target)

                # Collect grads for the value function update in the outer loop [L14],
                #  which is not actually performed here
                meta_value_function_loss, value, mc, mc_std = self.value_function_loss_on_batch(f_value_function, obs_val, r_obs_val, mc_returns_val, task_idx=train_task_idx, target=vf_target)
                total_vf_loss = meta_value_function_loss / self.num_tasks
                if self.value_reg > 0:
                    total_vf_loss = total_vf_loss + self.value_reg * self.model.value(obs, r_obs).pow(2).mean()
                total_vf_loss.backward()

                outer_values.append(value.item())
                outer_mc_means.append(mc.item())
                outer_mc_stds.append(mc_std.item())
                meta_value_losses.append(meta_value_function_loss.item())
                ##################################################################################################

            if self.q:
                qf = self.model.critic
                qf.train()
                qf_target = deepcopy(qf)
                # opt = O.SGD([{'params': p, 'lr': None} for p in qf.adaptation_parameters()])
                # opt = O.SGD([{'params': p, 'lr': None} for p in qf.parameters()])
                opt = torch.optim.SGD(self.model.critic.parameters(), lr=self.inner_value_lr)
                # with higher.innerloop_ctx(qf, opt, override={'lr': [F.softplus(l) for l in self._q_lrs]}, copy_initial_weights=False) as (f_q_function, diff_q_opt):
                with higher.innerloop_ctx(qf, opt, copy_initial_weights=False) as (f_q_function, diff_q_opt):
                    if self.num_tasks > 1:
                        for step in range(self.maml_steps):
                            # sub_batch = value_batch.view(self._args.maml_steps, value_batch.shape[0] // self._args.maml_steps, *value_batch.shape[1:])[step]
                            loss  = self.q_function_loss_on_batch(f_q_function, obs, r_obs, act, mc_returns, inner=True, task_idx=train_task_idx)#, iweights=iweights_no_action_)
                            diff_q_opt.step(loss)
                            inner_q_losses.append(loss.item())

                            # Soft update target value function parameters
                            self.soft_update(f_q_function, qf_target)

                    # Collect grads for the value function update in the outer loop [L14],
                    #  which is not actually performed here
                    meta_q_function_loss = self.q_function_loss_on_batch(f_q_function, obs_val, r_obs_val, act_val, mc_returns_val, task_idx=train_task_idx)
                    total_qf_loss = meta_q_function_loss / self.num_tasks
                    total_qf_loss.backward()
                    meta_value_losses.append(meta_q_function_loss.item())


            ##################################################################################################
            # Adapt policy and collect meta-gradients
            ##################################################################################################
            adapted_value_function = f_value_function
            adapted_q_function = f_q_function if self.q else None
            
            # opt = O.SGD([{'params': p, 'lr': None} for p in self.model.actor.adaptation_parameters()])
            # opt = O.SGD([{'params': p, 'lr': None} for p in self.model.actor.parameters()])
            opt = torch.optim.SGD(self.model.actor.parameters(), lr=self.inner_policy_lr)
            # self.model.actor.train()
            # with higher.innerloop_ctx(self.model.actor, opt, override={'lr': [F.softplus(l) for l in self._policy_lrs]}, copy_initial_weights=False) as (f_adaptation, diff_policy_opt):
            with higher.innerloop_ctx(self.model.actor, opt, copy_initial_weights=False) as (f_adaptation, diff_policy_opt):
                if self.num_tasks > 1:
                    for step in range(self.maml_steps):
                        # sub_batch = policy_batch.view(self._args.maml_steps, policy_batch.shape[0] // self._args.maml_steps, *policy_batch.shape[1:])[step]
                        loss, adv, weights, adv_loss = self.adaptation_policy_loss_on_batch(f_adaptation, adapted_q_function,
                                                                                                adapted_value_function, obs, r_obs, act, mc_returns, train_task_idx, inner=True)
                        if adv_loss is not None:
                            adv_policy_losses.append(adv_loss.item())
                        inner_advantages.append(adv.item())
                        inner_weights.append(weights.mean().item())

                        diff_policy_opt.step(loss)
                        inner_policy_losses.append(loss.item())

                meta_policy_loss, outer_adv, outer_weights_, _ = self.adaptation_policy_loss_on_batch(f_adaptation, adapted_q_function,
                                                                                                        adapted_value_function, obs_val, r_obs_val, act_val, mc_returns_val, train_task_idx)
                outer_weights.append(outer_weights_.mean().item())
                outer_advantages.append(outer_adv.item())

                (meta_policy_loss / self.num_tasks).backward()
                meta_policy_losses.append(meta_policy_loss.item())

            # Meta-update value function [L14]
            grad_value = self.update_model(self.model.value, self.value_optimizer, clip=self._grad_clip)

            # Meta-update Q function [L14]
            if self.q:
                grad_q = self.update_model(self.model.critic, self.critic_optimizer, clip=self._grad_clip)
                if data_for_logging is not None:
                    data_for_logging[0].log(
                        {
                            "loss/q": total_qf_loss,
                            "grad_critic": grad_q,
                        },
                        step=data_for_logging[1],
                    )

        # Meta-update adaptation policy [L15]
        grad_actor = self.update_model(self.model.actor, self.actor_optimizer, clip=self._grad_clip)

        if self.lrlr > 0:
            self.update_params(self._value_lrs, self._value_lr_optimizer)
            self.update_params(self._q_lrs, self._q_lr_optimizer)
            self.update_params(self._policy_lrs, self._policy_lr_optimizer)
            if self.advantage_head_coef is not None:
                self.update_params([self._adv_coef], self._adv_coef_optimizer)
        # return rollouts, test_rewards, train_rewards, meta_value_losses, meta_policy_losses, None, successes

        if data_for_logging is not None:
            data_for_logging[0].log(
                {
                    "loss/actor": (meta_policy_loss /self.num_tasks),
                    "loss/value": total_vf_loss,
                    "grad_actor": grad_actor,
                    "grad_value": grad_value,
                },
                step=data_for_logging[1],
            )

                
