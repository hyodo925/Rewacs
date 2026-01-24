import copy
import higher
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
from tqdm import tqdm
import numpy as np


class MAMLAWAC:
    def __init__(
        self,
        model,
        tasks,
        actor_optimizer,
        critic_optimizer,
        batch_size,
        num_tasks=5,
        polyak=0.995,
        gamma=0.9,
        beta=0.3,
        outer_policy_lr=3e-4,
        inner_policy_lr=3e-4,
        outer_value_lr=3e-4,
        inner_value_lr=3e-4,
    ):
        self.alg_name = "MAMLAWAC"
        self.model = model
        self.target = copy.deepcopy(model)
        self.tasks = tasks
        self.actor_optimizer = actor_optimizer
        self.critic_optimizer = critic_optimizer
        self.outer_policy_lr = outer_policy_lr
        self.inner_policy_lr = inner_policy_lr
        self.outer_value_lr = outer_value_lr
        self.inner_value_lr = inner_value_lr
        self.batch_size = batch_size
        self.polyak = polyak
        self.gamma = torch.as_tensor([gamma])
        self.beta = torch.as_tensor([beta])

        self.device = model.device

        self.num_tasks = num_tasks
        self.itr_inner_loop = 1
        self.target_vf_alpha = 0.9
        self.grad_clip = 1e9
        self.advantage_head_coef = None
        self.policy_lrs = None
        self.value_lrs = None
        self.q_lrs = None
        self.adv_coef = None
        self.learn_lr = False
        if self.learn_lr:
            self.lrlr = 1e-4
            self.policy_lrs = [torch.nn.Parameter(torch.tensor(float(np.log(inner_policy_lr))).to(self.device))
                                for p in self.model.actor.parameters()]
            self.q_lrs = [torch.nn.Parameter(torch.tensor(float(np.log(inner_value_lr))).to(self.device))
                                for p in self.model.critic.parameters()]
            if self.advantage_head_coef is not None:
                self.adv_coef = torch.nn.Parameter(torch.tensor(float(np.log(self.advantage_head_coef))).to(self.device))
            
            self.policy_lr_optimizer = torch.optim.Adam(self.policy_lrs, lr=self.lrlr)
            self.q_lr_optimizer = torch.optim.Adam(self.q_lrs, lr=self.lrlr)

    def safe_exp(self, x):
        return torch.where(x < 50, torch.exp(x), x)

    def soft_update(self, source, target):
        for param_source, param_target in zip(source.named_parameters(), target.named_parameters()):
            assert param_source[0] == param_target[0]
            param_target[1].data = self.target_vf_alpha * param_target[1].data + (1 - self.target_vf_alpha) * param_source[1].data

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

    def advantage_loss(self, policy, q_function, obs, r_obs, act):
        Q1, Q2 = q_function(
            (obs.to(self.device), r_obs.to(self.device)),
            act=act.squeeze().to(self.device),
        )

        action_gen, _, _ = policy.sample(
            (
                obs.to(self.device),
                r_obs.reshape(self.batch_size, 1, -1).to(self.device),
            )
        )
        qw_ref = torch.min(torch.cat((Q1, Q2), 1), dim=1)[0].reshape(
            (-1, 1)
        )

        v_act1, v_act2 = q_function(
            (obs.to(self.device), r_obs.to(self.device)),
            act=action_gen,
        )

        qw_gen = torch.min(torch.cat((v_act1, v_act2), 1), dim=1)[
            0
        ].reshape((-1, 1))

        adv = torch.max(torch.zeros_like(qw_ref), qw_ref - qw_gen).detach()
        adv = (adv - adv.mean()) / (adv.std() + 1e-8) 
        weights = self.safe_exp(adv / self.beta)
        weights = torch.clamp(weights, max=100.0)
        weights = weights / (weights.mean() + 1e-8)
        
        loss_act = -(
                policy.get_log_prob(
                    (
                        obs.to(self.device),
                        r_obs.reshape(self.batch_size, 1, -1).to(
                            self.device
                        ),
                    ),
                    act.squeeze().to(self.device),
                )
                * weights
            ).mean()
        return loss_act
    def value_loss(self,q_function, target, prev_obs, obs, prev_r_obs, r_obs, act, rwd, done):
        next_act_target, next_log_prob, _ = self.model.actor.sample(
            (
                obs.to(self.device),
                r_obs.reshape(self.batch_size, 1, -1).to(self.device),
            )
        )
        Q_target_1, Q_target_2 = target(
            (obs.to(self.device), r_obs.to(self.device)), 
            act=next_act_target
        )
        Q_target_min = torch.min(
            torch.cat((Q_target_1, Q_target_2), 1), dim=1
        )[0].unsqueeze(-1)

        Q_target = rwd.to(self.device) + (
            self.gamma * Q_target_min
        ) * done.to(self.device)

        Q1, Q2 = q_function(
            (prev_obs.to(self.device), prev_r_obs.to(self.device)),
            act=act.squeeze().to(self.device),
        )

        loss_value = F.mse_loss(Q_target, Q1) + F.mse_loss(Q_target, Q2)

        return loss_value   
    
    def step(self, sample, sample_val, update_actor=False,):
        obs, next_obs, r_obs, next_r_obs, act, rwd, done = list(sample.values())
        obs_val, next_obs_val, r_obs_val, next_r_obs_val, act_val, rwd_val, done_val = list(sample_val.values())

        qf = self.model.critic
        qf.train()
        qf_target = copy.deepcopy(qf)
        # opt = torch.optim.SGD([{'params': p, 'lr': None} for p in qf.parameters()])
        opt = torch.optim.SGD(self.model.critic.parameters(), lr=self.inner_value_lr)
        # with higher.innerloop_ctx(qf, opt, override={'lr': [F.softplus(l) for l in self.q_lrs]}, copy_initial_weights=False) as (f_q_function, diff_q_opt):
        with higher.innerloop_ctx(qf, opt, copy_initial_weights=False) as (f_q_function, diff_q_opt):    
            for _ in range(self.itr_inner_loop):
                inner_value_loss = self.value_loss(f_q_function, qf_target, obs.to(self.device), next_obs.to(self.device), r_obs.to(self.device), next_r_obs.to(self.device), act.to(self.device), rwd.to(self.device), done.to(self.device))
                diff_q_opt.step(inner_value_loss)
                self.soft_update(f_q_function, qf_target)

            meta_value_loss = self.value_loss(f_q_function, qf_target, obs_val.to(self.device), next_obs_val.to(self.device), r_obs_val.to(self.device), next_r_obs_val.to(self.device), act_val.to(self.device), rwd_val.to(self.device), done_val.to(self.device))
            # (meta_value_loss / self.num_tasks).backward()
        # value_loss = self.value_loss(self.model.critic, self.target.critic, obs_val.to(self.device), next_obs_val.to(self.device), r_obs_val.to(self.device), next_r_obs_val.to(self.device), act_val.to(self.device), rwd_val.to(self.device), done_val.to(self.device))
        adapted_value_function = f_q_function
        # opt = torch.optim.SGD([{'params': p, 'lr': None} for p in self.model.actor.parameters()])
        opt = torch.optim.SGD(self.model.actor.parameters(), lr=self.inner_policy_lr)
        self.model.actor.train()
        # with higher.innerloop_ctx(self.model.actor, opt, override={'lr': [F.softplus(l) for l in self.policy_lrs]}, copy_initial_weights=False) as (f_policy, diff_policy_opt):
        with higher.innerloop_ctx(self.model.actor, opt, copy_initial_weights=False) as (f_policy, diff_policy_opt):
            for _ in range(self.itr_inner_loop):
                inner_policy_loss = self.advantage_loss(f_policy, adapted_value_function, obs.to(self.device), r_obs.to(self.device), act.to(self.device))
                diff_policy_opt.step(inner_policy_loss)
            meta_policy_loss = self.advantage_loss(f_policy, adapted_value_function, obs_val.to(self.device), r_obs_val.to(self.device), act_val.to(self.device))
            # (meta_policy_loss / self.num_tasks).backward()
        # policy_loss = self.advantage_loss(self.model.actor, self.model.critic, obs_val.to(self.device), r_obs_val.to(self.device), act_val.to(self.device))
        return  meta_value_loss, meta_policy_loss, inner_value_loss, inner_policy_loss
        # return value_loss, policy_loss
    
    
    def update(self, data_for_logging=None):
        meta_policy_losses = []
        meta_value_losses = []
        total_meta_value_loss = 0
        total_meta_policy_loss = 0
        # sampled_tasks = random.sample(self.tasks, self.num_tasks) 
        # task = self.tasks[0]
        for task in self.tasks:
            # for batch in task:
            sample = task.sample(self.batch_size)
            sample_val = task.sample(self.batch_size)
            meta_value_loss, meta_policy_loss, inner_value_loss, inner_policy_loss = self.step(sample, sample_val)
            meta_policy_losses.append(meta_policy_loss)
            meta_value_losses.append(meta_value_loss)
            total_meta_value_loss += meta_value_loss
            total_meta_policy_loss += meta_policy_loss

        (total_meta_value_loss / self.num_tasks).backward()
        (total_meta_policy_loss / self.num_tasks).backward()
        meta_policy_grad = self.update_model(self.model.actor, self.actor_optimizer, clip=self.grad_clip)
        meta_critic_grad = self.update_model(self.model.critic, self.critic_optimizer, clip=self.grad_clip)
        # if self.lrlr > 0:
        #     self.update_params(self.q_lrs, self.q_lr_optimizer)
        #     self.update_params(self.policy_lrs, self.policy_lr_optimizer)

        if data_for_logging is not None:
            data_for_logging[0].log(
                {
                    "meta_value_losses_mean": meta_value_loss,
                    "meta_policy_losses_mean": meta_policy_loss,
                    "meta_actor_grad":meta_policy_grad,
                    "meta_critic_grad":meta_critic_grad,
                    "inner_value_loss":inner_value_loss,
                    "inner_policy_loss":inner_policy_loss,
                },
                step=data_for_logging[1],
            )

    def update_target(self):
        for param, target_param in zip(
            self.model.parameters(), self.target.parameters()
        ):
            target_param.data.mul_(self.polyak)
            target_param.data.add_((1 - self.polyak) * param.data)
