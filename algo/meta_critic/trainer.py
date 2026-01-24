import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from .virtual_updater import VirtualActorUpdater
import higher


def l1_penalty(var):
    return torch.abs(var).sum()
class MetaCriticAWAC:
    def __init__(
        self,
        model,
        replay_buffer,
        replay_buffer_val,
        actor_optimizer,
        critic_optimizer,
        meta_critic_optimizer,
        batch_size,
        lr=3e-4,
        polyak=0.995,
        gamma=0.9,
        beta=0.3,
    ):
        self.alg_name = "MetaCriticAWAC"
        self.model = model
        self.target = copy.deepcopy(model)
        self.replay_buffer = replay_buffer
        self.replay_buffer_val = replay_buffer_val
        self.actor_optimizer = actor_optimizer
        self.critic_optimizer = critic_optimizer
        self.meta_critic_optimizer = meta_critic_optimizer
        self.batch_size = batch_size
        self.polyak = polyak
        self.gamma = torch.as_tensor([gamma])
        self.beta = torch.as_tensor([beta])
        self.lr =lr
        self.device = model.device

        feature_net = nn.Sequential(*list(self.model.actor.children())[:-2])

        def get_layer(model):
            count = 0
            para_optim = []
            for k in model.children():
                count += 1
                # 6 should be changed properly
                for param in k.parameters():
                    para_optim.append(param)
            return para_optim
        self.param_optim_theta = get_layer(self.model.actor)

    def safe_exp(self, x):
        return torch.where(x < 50, torch.exp(x), x)

    def update(self, data_for_logging=None):
        sample = self.replay_buffer.sample(self.batch_size)
        obs, next_obs, r_obs, next_r_obs, act, rwd, done = list(sample.values())

        sample_val = self.replay_buffer.sample(self.batch_size)
        obs_val, next_obs_val, r_obs_val, next_r_obs_val, act_val, rwd_val, done_val = list(sample_val.values())

        with torch.no_grad():
            next_act_target, next_log_prob, *_ = self.model.actor.sample(
                (
                    next_obs.to(self.device),
                    next_r_obs.reshape(self.batch_size, 1, -1).to(self.device),
                )
            )
            Q_target_1, Q_target_2 = self.target.critic(
                (next_obs.to(self.device), next_r_obs.to(self.device)), next_act_target
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
        )

        loss_critic = F.mse_loss(Q_target, Q1) + F.mse_loss(Q_target, Q2)
        self.critic_optimizer.zero_grad()
        loss_critic.backward()
        self.critic_optimizer.step()
        lc = loss_critic.data.item()

        if data_for_logging is not None:
            data_for_logging[0].log(
                {
                    "loss/critic": lc,
                },
                step=data_for_logging[1],
            )
        ###############################
        ### compute actor loss      ###
        ###############################
        with higher.innerloop_ctx(self.model.actor,self.actor_optimizer,copy_initial_weights=False) as (f_actor, diffopt):
            action_gen, log_prob, _, other_output = f_actor.sample((obs.to(self.device),r_obs.reshape(self.batch_size, 1, -1).to(self.device)))
            with torch.no_grad():
                qw_ref = torch.min(torch.cat((Q1, Q2), 1), dim=1)[0].reshape((-1, 1))
                v_act1, v_act2 = self.model.critic((obs.to(self.device), r_obs.to(self.device)),act=action_gen.detach(),)
                qw_gen = torch.min(torch.cat((v_act1, v_act2), 1), dim=1)[0].reshape((-1, 1))
                adv = torch.max(torch.zeros_like(qw_ref), qw_ref - qw_gen)
                weights = self.safe_exp(adv / self.beta)

            loss_act = -(f_actor.get_log_prob((obs.to(self.device),r_obs.reshape(self.batch_size, 1, -1).to(self.device),),act.squeeze().to(self.device),)* weights).mean()
            diffopt.step(loss_act)

            with torch.no_grad():
                pi_val, log_pi_val, *_ = f_actor.sample((obs_val.to(self.device),r_obs_val.reshape(self.batch_size, 1, -1).to(self.device),))
                v_act1, v_act2 = self.model.critic((obs_val.to(self.device), r_obs_val.to(self.device)),act=pi_val)
                qw_gen = torch.min(torch.cat((v_act1, v_act2), 1), dim=1)[0].reshape((-1, 1))
                adv = torch.max(torch.zeros_like(qw_ref), qw_ref - qw_gen)
                weights = self.safe_exp(adv / self.beta)
            policy_loss_val = -(f_actor.get_log_prob((obs_val.to(self.device),r_obs_val.reshape(self.batch_size, 1, -1).to(self.device),),pi_val.squeeze().to(self.device),)* weights).mean()

            loss_auxiliary = self.model.meta_critic(
                (
                    obs.reshape(self.batch_size, -1).to(self.device),
                    r_obs.reshape(self.batch_size, -1).to(self.device),
                ),
                act.squeeze().to(self.device),
                other_output.reshape(self.batch_size, -1).to(self.device),
            )
            # concat = torch.concat([obs.reshape(self.batch_size, -1).to(self.device), r_obs.reshape(self.batch_size, -1).to(self.device), act.squeeze().to(self.device),other_output.reshape(self.batch_size, -1).to(self.device),], dim=1)
            # loss_auxiliary = self.model.meta_critic(concat)

            diffopt.step(loss_auxiliary)

            with torch.no_grad():
                pi_val_new, log_pi_val_new, *_ = f_actor.sample((obs_val.to(self.device),r_obs_val.reshape(self.batch_size, 1, -1).to(self.device),))
                v_act1, v_act2 = self.model.critic((obs_val.to(self.device), r_obs_val.to(self.device)),act=pi_val_new)
                qw_gen = torch.min(torch.cat((v_act1, v_act2), 1), dim=1)[0].reshape((-1, 1))
                adv = torch.max(torch.zeros_like(qw_ref), qw_ref - qw_gen)
                weights = self.safe_exp(adv / self.beta)
            policy_loss_val_new = -(f_actor.get_log_prob((obs_val.to(self.device),r_obs_val.reshape(self.batch_size, 1, -1).to(self.device)),pi_val_new.squeeze().to(self.device),)* weights).mean()

            total_actor_loss = loss_act + loss_auxiliary
        utility = policy_loss_val - policy_loss_val_new
        utility = torch.tanh(utility)
        loss_meta = -utility
        self.meta_critic_optimizer.zero_grad()
        # loss_meta.backward()
        grad_omega = torch.autograd.grad(loss_meta, self.model.meta_critic.parameters(),allow_unused=True)
        for gradient, variable in zip(grad_omega, self.model.meta_critic.parameters()):
            variable.grad = gradient
        # loss_meta.backward()
        self.meta_critic_optimizer.step()
        lm = loss_meta.data.item()

        self.actor_optimizer.zero_grad()
        total_actor_loss.backward()
        self.actor_optimizer.step()
        la = loss_act.data.item()

        if data_for_logging is not None:
            data_for_logging[0].log(
                {
                    "loss/actor": la,
                    "loss/meta": lm,
                    # "abs_theta": abs_theta,
                },
                step=data_for_logging[1],
            )

    def update_target(self):
        for param, target_param in zip(
            self.model.parameters(), self.target.parameters()
        ):
            target_param.data.mul_(self.polyak)
            target_param.data.add_((1 - self.polyak) * param.data)


class Scalar(nn.Module):
    def __init__(self, init_value: float):
        super().__init__()
        self.constant = nn.Parameter(torch.tensor(init_value, dtype=torch.float32))

    def forward(self) -> nn.Parameter:
        return self.constant
    
class MetaCriticCalQL:
    def __init__(
        self,
        model,
        replay_buffer,
        replay_buffer_val,
        action_dim,
        actor_optimizer,
        critic_optimizer,
        meta_critic_optimizer,
        batch_size,
        lr=3e-4,
        polyak=0.995,
        gamma=0.9,
        beta=0.3,
    ):
        self.alg_name = "MetaCriticCalQL"
        self.model = model
        self.action_dim = action_dim
        self.target = copy.deepcopy(model)
        self.replay_buffer = replay_buffer
        self.replay_buffer_val = replay_buffer_val
        self.actor_optimizer = actor_optimizer
        self.critic_optimizer = critic_optimizer
        self.meta_critic_optimizer = meta_critic_optimizer
        self.batch_size = batch_size
        self.polyak = polyak
        self.gamma = torch.as_tensor([gamma])
        self.beta = torch.as_tensor([beta])
        self.lr =lr
        self.device = model.device

        self.updater = VirtualActorUpdater()

        self.cql_alpha = 10.0
        self.cql_temp = 1.0
        self.cql_lagrange = True
        self.use_automatic_entropy_tuning = True
        self.cql_importance_sample = True
        self.cql_target_action_gap = -1.0
        self.target_entropy = -np.prod(self.action_dim).item()
        self.cql_n_actions = 10
        self.alpha_multiplier = 1.0
        self.cql_clip_diff_min = float("-inf")
        self.cql_clip_diff_max = float("inf")
        self._calibration_enabled = True

        if self.use_automatic_entropy_tuning:
            self.log_alpha = Scalar(0.0)
            self.alpha_optimizer = torch.optim.Adam(
                self.log_alpha.parameters(),
                lr=self.lr,
            )
        else:
            self.log_alpha = None

        self.log_alpha_prime = Scalar(1.0)
        self.alpha_prime_optimizer = torch.optim.Adam(
            self.log_alpha_prime.parameters(),
            lr=self.lr,
        )

    def safe_exp(self, x):
        return torch.where(x < 50, torch.exp(x), x)

    def update(self, current_it, total_it, data_for_logging=None):
        sample = self.replay_buffer.sample(self.batch_size)
        obs, next_obs, r_obs, next_r_obs, act, rwd, mc_returns, done = list(sample.values())

        sample_val = self.replay_buffer_val.sample(self.batch_size)
        obs_val, next_obs_val, r_obs_val, next_r_obs_val, act_val, rwd_val, mc_returns_val, done_val = list(sample_val.values())

        act_target, log_prob, _, other_output= self.model.actor.sample(
            (
                obs.to(self.device),
                r_obs.reshape(self.batch_size, 1, -1).to(self.device),
            )
        )
        alpha_loss = -(
            self.log_alpha() * (log_prob + self.target_entropy).detach()
        ).mean()
        alpha = self.log_alpha().exp() * self.alpha_multiplier

        if current_it <= total_it:
            get_log_prob = self.model.actor.get_log_prob(
                    (
                        obs.to(self.device),
                        r_obs.reshape(self.batch_size, 1, -1).to(self.device),
                    ),
                    act.squeeze().to(self.device),
                )
            loss_act = (alpha * log_prob - get_log_prob).mean()
        else:
            v_act1, v_act2 = self.model.critic(
                (obs.to(self.device), r_obs.to(self.device)),
                act_target.detach(),
            )
            q_new_actions= torch.min(torch.cat((v_act1, v_act2), 1), dim=1)[0].reshape((-1, 1))
            loss_act = (alpha * log_prob - q_new_actions).mean()


        with torch.no_grad():
            next_act_target, _, _, _= self.model.actor.sample(
                (
                    obs.to(self.device),
                    r_obs.reshape(self.batch_size, 1, -1).to(self.device),
                )
            )
            Q_target_1, Q_target_2 = self.target.critic(
                (next_obs.to(self.device), next_r_obs.to(self.device)), next_act_target
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
        )

        td_loss = F.mse_loss(Q_target, Q1) + F.mse_loss(Q_target, Q2)

        batch_size, action_dim = act.shape[0], act.shape[-1]
        cql_random_actions = act.squeeze().new_empty(
            (batch_size, self.cql_n_actions, action_dim), requires_grad=False
        ).uniform_(-1, 1)

        cql_current_actions, cql_current_log_pis, _, _ = self.model.actor.sample(
            (
                obs.to(self.device),
                r_obs.reshape(self.batch_size, 1, -1).to(self.device),
            ), 
            repeat=self.cql_n_actions
        )
        cql_next_actions, cql_next_log_pis, _, _ = self.model.actor.sample(
            (
                next_obs.to(self.device),
                next_r_obs.reshape(self.batch_size, 1, -1).to(self.device),
            ), 
            repeat=self.cql_n_actions
        )
        cql_current_actions, cql_current_log_pis = (
            cql_current_actions.detach(),
            cql_current_log_pis.detach(),
        )
        cql_next_actions, cql_next_log_pis = (
            cql_next_actions.detach(),
            cql_next_log_pis.detach(),
        )

        cql_q1_rand, cql_q2_rand = self.model.critic(
            (
                obs.to(self.device),
                r_obs.reshape(self.batch_size, 1, -1).to(self.device),
            ), 
            cql_random_actions
        )
        cql_q1_current_actions, cql_q2_current_actions = self.model.critic(
            (
                obs.to(self.device),
                r_obs.reshape(self.batch_size, 1, -1).to(self.device),
            ), 
            cql_current_actions
        )
        cql_q1_next_actions, cql_q2_next_actions = self.model.critic(
            (
                obs.to(self.device),
                r_obs.reshape(self.batch_size, 1, -1).to(self.device),
            ), 
            cql_next_actions
        )

        # Calibration
        lower_bounds = mc_returns.reshape(-1, 1).repeat(
            1, cql_q1_current_actions.shape[1]
        )

        # num_vals = torch.sum(lower_bounds == lower_bounds)
        # bound_rate_cql_q1_current_actions = (
        #     torch.sum(cql_q1_current_actions < lower_bounds) / num_vals
        # )
        # bound_rate_cql_q2_current_actions = (
        #     torch.sum(cql_q2_current_actions < lower_bounds) / num_vals
        # )
        # bound_rate_cql_q1_next_actions = (
        #     torch.sum(cql_q1_next_actions < lower_bounds) / num_vals
        # )
        # bound_rate_cql_q2_next_actions = (
        #     torch.sum(cql_q2_next_actions < lower_bounds) / num_vals
        # )

        """ Cal-QL: bound Q-values with MC return-to-go """
        if self._calibration_enabled:
            cql_q1_current_actions = torch.maximum(cql_q1_current_actions, lower_bounds.unsqueeze(-1))
            cql_q2_current_actions = torch.maximum(cql_q2_current_actions, lower_bounds.unsqueeze(-1))
            cql_q1_next_actions = torch.maximum(cql_q1_next_actions, lower_bounds.unsqueeze(-1))
            cql_q2_next_actions = torch.maximum(cql_q2_next_actions, lower_bounds.unsqueeze(-1))

        cql_cat_q1 = torch.cat(
            [
                cql_q1_rand,
                torch.unsqueeze(Q1, 1),
                cql_q1_next_actions,
                cql_q1_current_actions,
            ],
            dim=1,
        )
        cql_cat_q2 = torch.cat(
            [
                cql_q2_rand,
                torch.unsqueeze(Q2, 1),
                cql_q2_next_actions,
                cql_q2_current_actions,
            ],
            dim=1,
        )
        cql_std_q1 = torch.std(cql_cat_q1, dim=1)
        cql_std_q2 = torch.std(cql_cat_q2, dim=1)

        if self.cql_importance_sample:
            random_density = np.log(0.5**action_dim)
            cql_cat_q1 = torch.cat(
                [
                    cql_q1_rand - random_density,
                    cql_q1_next_actions - cql_next_log_pis.detach(),
                    cql_q1_current_actions - cql_current_log_pis.detach(),
                ],
                dim=1,
            )
            cql_cat_q2 = torch.cat(
                [
                    cql_q2_rand - random_density,
                    cql_q2_next_actions - cql_next_log_pis.detach(),
                    cql_q2_current_actions - cql_current_log_pis.detach(),
                ],
                dim=1,
            )

        cql_qf1_ood = torch.logsumexp(cql_cat_q1 / self.cql_temp, dim=1) * self.cql_temp
        cql_qf2_ood = torch.logsumexp(cql_cat_q2 / self.cql_temp, dim=1) * self.cql_temp

        """Subtract the log likelihood of data"""
        cql_qf1_diff = torch.clamp(
            cql_qf1_ood - Q1,
            self.cql_clip_diff_min,
            self.cql_clip_diff_max,
        ).mean()
        cql_qf2_diff = torch.clamp(
            cql_qf2_ood - Q2,
            self.cql_clip_diff_min,
            self.cql_clip_diff_max,
        ).mean()

        if self.cql_lagrange:
            alpha_prime = torch.clamp(
                torch.exp(self.log_alpha_prime()), min=0.0, max=1000000.0
            )
            cql_min_qf1_loss = (
                alpha_prime
                * self.cql_alpha
                * (cql_qf1_diff - self.cql_target_action_gap)
            )
            cql_min_qf2_loss = (
                alpha_prime
                * self.cql_alpha
                * (cql_qf2_diff - self.cql_target_action_gap)
            )

            self.alpha_prime_optimizer.zero_grad()
            alpha_prime_loss = (-cql_min_qf1_loss - cql_min_qf2_loss) * 0.5
            alpha_prime_loss.backward(retain_graph=True)
            self.alpha_prime_optimizer.step()
        else:
            cql_min_qf1_loss = cql_qf1_diff * self.cql_alpha
            cql_min_qf2_loss = cql_qf2_diff * self.cql_alpha

        loss_critic = td_loss+ cql_min_qf1_loss + cql_min_qf2_loss

        self.critic_optimizer.zero_grad()
        loss_critic.backward()
        self.critic_optimizer.step()
        lc = loss_critic.data.item()

        if data_for_logging is not None:
            data_for_logging[0].log(
                {
                    "loss/critic": lc,
                },
                step=data_for_logging[1],
            )

        if self.use_automatic_entropy_tuning:
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()
            lalpha = alpha_loss.data.item()

        with higher.innerloop_ctx(self.model.actor,self.actor_optimizer,copy_initial_weights=False) as (f_actor, diffopt):
            act_target, log_prob, _, other_output= f_actor.sample((obs.to(self.device),r_obs.reshape(self.batch_size, 1, -1).to(self.device),))
            get_log_prob = f_actor.get_log_prob((obs.to(self.device),r_obs.reshape(self.batch_size, 1, -1).to(self.device),),act.squeeze().to(self.device),)
            loss_act = (alpha * log_prob - get_log_prob).mean()
            diffopt.step(loss_act)
            with torch.no_grad():
                act_target, log_prob, _, _= f_actor.sample((obs_val.to(self.device),r_obs_val.reshape(self.batch_size, 1, -1).to(self.device),))
            get_log_prob = f_actor.get_log_prob((obs_val.to(self.device),r_obs_val.reshape(self.batch_size, 1, -1).to(self.device),),act_val.squeeze().to(self.device),)
            policy_loss_val = (alpha * log_prob - get_log_prob).mean()

            loss_auxiliary = self.model.meta_critic(
                (
                    obs.reshape(self.batch_size, -1).to(self.device),
                    r_obs.reshape(self.batch_size, -1).to(self.device),
                ),
                act.squeeze().to(self.device),
                other_output.reshape(self.batch_size, -1).to(self.device),
            )
            # concat = torch.concat([obs.reshape(self.batch_size, -1).to(self.device), r_obs.reshape(self.batch_size, -1).to(self.device), act.squeeze().to(self.device),other_output.reshape(self.batch_size, -1).to(self.device),], dim=1)
            # loss_auxiliary = self.model.meta_critic(concat)

            diffopt.step(loss_auxiliary)
            with torch.no_grad():
                act_target, log_prob, _, _= f_actor.sample((obs_val.to(self.device),r_obs_val.reshape(self.batch_size, 1, -1).to(self.device),),)
            get_log_prob = f_actor.get_log_prob((obs_val.to(self.device),r_obs_val.reshape(self.batch_size, 1, -1).to(self.device),),act_val.squeeze().to(self.device),)
            policy_loss_val_new = (alpha * log_prob - get_log_prob).mean()

            total_actor_loss = loss_act + loss_auxiliary
            # total_actor_loss = loss_act
            # total_actor_loss = loss_auxiliary


        utility = policy_loss_val - policy_loss_val_new
        utility = torch.tanh(utility)
        loss_meta = -utility
        self.meta_critic_optimizer.zero_grad()
        grad_omega = torch.autograd.grad(loss_meta, self.model.meta_critic.parameters(),allow_unused=True)
        for gradient, variable in zip(grad_omega, self.model.meta_critic.parameters()):
            variable.grad = gradient
        self.meta_critic_optimizer.step()
        lm = loss_meta.data.item()

        self.actor_optimizer.zero_grad()
        total_actor_loss.backward()
        self.actor_optimizer.step()
        la = loss_act.data.item()

        if data_for_logging is not None:
            data_for_logging[0].log(
                {
                    "loss/actor": la,
                    "loss/meta": lm,
                },
                step=data_for_logging[1],
            )

    
    def finetune(self, current_it, total_it, data_for_logging=None):
        sample = self.replay_buffer.sample(self.batch_size)
        obs, next_obs, r_obs, next_r_obs, act, rwd, mc_returns, done = list(sample.values())

        sample_val = self.replay_buffer_val.sample(self.batch_size)
        obs_val, next_obs_val, r_obs_val, next_r_obs_val, act_val, rwd_val, mc_returns_val, done_val = list(sample_val.values())

        act_target, log_prob, _, other_output= self.model.actor.sample(
            (
                obs.to(self.device),
                r_obs.reshape(self.batch_size, 1, -1).to(self.device),
            )
        )
        alpha_loss = -(
            self.log_alpha() * (log_prob + self.target_entropy).detach()
        ).mean()
        alpha = self.log_alpha().exp() * self.alpha_multiplier

        if current_it <= total_it:
            get_log_prob = self.model.actor.get_log_prob(
                    (
                        obs.to(self.device),
                        r_obs.reshape(self.batch_size, 1, -1).to(self.device),
                    ),
                    act.squeeze().to(self.device),
                )
            loss_act = (alpha * log_prob - get_log_prob).mean()
        else:
            v_act1, v_act2 = self.model.critic(
                (obs.to(self.device), r_obs.to(self.device)),
                act_target.detach(),
            )
            q_new_actions= torch.min(torch.cat((v_act1, v_act2), 1), dim=1)[0].reshape((-1, 1))
            loss_act = (alpha * log_prob - q_new_actions).mean()


        with torch.no_grad():
            next_act_target, _, _, _= self.model.actor.sample(
                (
                    obs.to(self.device),
                    r_obs.reshape(self.batch_size, 1, -1).to(self.device),
                )
            )
            Q_target_1, Q_target_2 = self.target.critic(
                (next_obs.to(self.device), next_r_obs.to(self.device)), next_act_target
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
        )

        td_loss = F.mse_loss(Q_target, Q1) + F.mse_loss(Q_target, Q2)

        batch_size, action_dim = act.shape[0], act.shape[-1]
        cql_random_actions = act.squeeze().new_empty(
            (batch_size, self.cql_n_actions, action_dim), requires_grad=False
        ).uniform_(-1, 1)

        cql_current_actions, cql_current_log_pis, _, _ = self.model.actor.sample(
            (
                obs.to(self.device),
                r_obs.reshape(self.batch_size, 1, -1).to(self.device),
            ), 
            repeat=self.cql_n_actions
        )
        cql_next_actions, cql_next_log_pis, _, _ = self.model.actor.sample(
            (
                next_obs.to(self.device),
                next_r_obs.reshape(self.batch_size, 1, -1).to(self.device),
            ), 
            repeat=self.cql_n_actions
        )
        cql_current_actions, cql_current_log_pis = (
            cql_current_actions.detach(),
            cql_current_log_pis.detach(),
        )
        cql_next_actions, cql_next_log_pis = (
            cql_next_actions.detach(),
            cql_next_log_pis.detach(),
        )

        cql_q1_rand, cql_q2_rand = self.model.critic(
            (
                obs.to(self.device),
                r_obs.reshape(self.batch_size, 1, -1).to(self.device),
            ), 
            cql_random_actions
        )
        cql_q1_current_actions, cql_q2_current_actions = self.model.critic(
            (
                obs.to(self.device),
                r_obs.reshape(self.batch_size, 1, -1).to(self.device),
            ), 
            cql_current_actions
        )
        cql_q1_next_actions, cql_q2_next_actions = self.model.critic(
            (
                obs.to(self.device),
                r_obs.reshape(self.batch_size, 1, -1).to(self.device),
            ), 
            cql_next_actions
        )

        # Calibration
        lower_bounds = mc_returns.reshape(-1, 1).repeat(
            1, cql_q1_current_actions.shape[1]
        )

        # num_vals = torch.sum(lower_bounds == lower_bounds)
        # bound_rate_cql_q1_current_actions = (
        #     torch.sum(cql_q1_current_actions < lower_bounds) / num_vals
        # )
        # bound_rate_cql_q2_current_actions = (
        #     torch.sum(cql_q2_current_actions < lower_bounds) / num_vals
        # )
        # bound_rate_cql_q1_next_actions = (
        #     torch.sum(cql_q1_next_actions < lower_bounds) / num_vals
        # )
        # bound_rate_cql_q2_next_actions = (
        #     torch.sum(cql_q2_next_actions < lower_bounds) / num_vals
        # )

        """ Cal-QL: bound Q-values with MC return-to-go """
        if self._calibration_enabled:
            cql_q1_current_actions = torch.maximum(cql_q1_current_actions, lower_bounds.unsqueeze(-1))
            cql_q2_current_actions = torch.maximum(cql_q2_current_actions, lower_bounds.unsqueeze(-1))
            cql_q1_next_actions = torch.maximum(cql_q1_next_actions, lower_bounds.unsqueeze(-1))
            cql_q2_next_actions = torch.maximum(cql_q2_next_actions, lower_bounds.unsqueeze(-1))

        cql_cat_q1 = torch.cat(
            [
                cql_q1_rand,
                torch.unsqueeze(Q1, 1),
                cql_q1_next_actions,
                cql_q1_current_actions,
            ],
            dim=1,
        )
        cql_cat_q2 = torch.cat(
            [
                cql_q2_rand,
                torch.unsqueeze(Q2, 1),
                cql_q2_next_actions,
                cql_q2_current_actions,
            ],
            dim=1,
        )
        cql_std_q1 = torch.std(cql_cat_q1, dim=1)
        cql_std_q2 = torch.std(cql_cat_q2, dim=1)

        if self.cql_importance_sample:
            random_density = np.log(0.5**action_dim)
            cql_cat_q1 = torch.cat(
                [
                    cql_q1_rand - random_density,
                    cql_q1_next_actions - cql_next_log_pis.detach(),
                    cql_q1_current_actions - cql_current_log_pis.detach(),
                ],
                dim=1,
            )
            cql_cat_q2 = torch.cat(
                [
                    cql_q2_rand - random_density,
                    cql_q2_next_actions - cql_next_log_pis.detach(),
                    cql_q2_current_actions - cql_current_log_pis.detach(),
                ],
                dim=1,
            )

        cql_qf1_ood = torch.logsumexp(cql_cat_q1 / self.cql_temp, dim=1) * self.cql_temp
        cql_qf2_ood = torch.logsumexp(cql_cat_q2 / self.cql_temp, dim=1) * self.cql_temp

        """Subtract the log likelihood of data"""
        cql_qf1_diff = torch.clamp(
            cql_qf1_ood - Q1,
            self.cql_clip_diff_min,
            self.cql_clip_diff_max,
        ).mean()
        cql_qf2_diff = torch.clamp(
            cql_qf2_ood - Q2,
            self.cql_clip_diff_min,
            self.cql_clip_diff_max,
        ).mean()

        if self.cql_lagrange:
            alpha_prime = torch.clamp(
                torch.exp(self.log_alpha_prime()), min=0.0, max=1000000.0
            )
            cql_min_qf1_loss = (
                alpha_prime
                * self.cql_alpha
                * (cql_qf1_diff - self.cql_target_action_gap)
            )
            cql_min_qf2_loss = (
                alpha_prime
                * self.cql_alpha
                * (cql_qf2_diff - self.cql_target_action_gap)
            )

            self.alpha_prime_optimizer.zero_grad()
            alpha_prime_loss = (-cql_min_qf1_loss - cql_min_qf2_loss) * 0.5
            alpha_prime_loss.backward(retain_graph=True)
            self.alpha_prime_optimizer.step()
        else:
            cql_min_qf1_loss = cql_qf1_diff * self.cql_alpha
            cql_min_qf2_loss = cql_qf2_diff * self.cql_alpha

        loss_critic = td_loss+ cql_min_qf1_loss + cql_min_qf2_loss

        self.critic_optimizer.zero_grad()
        loss_critic.backward()
        self.critic_optimizer.step()
        lc = loss_critic.data.item()

        if data_for_logging is not None:
            data_for_logging[0].log(
                {
                    "loss/critic": lc,
                },
                step=data_for_logging[1],
            )

        if self.use_automatic_entropy_tuning:
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()
            lalpha = alpha_loss.data.item()

        with higher.innerloop_ctx(self.model.actor,self.actor_optimizer,copy_initial_weights=False) as (f_actor, diffopt):
            act_target, log_prob, _, other_output= f_actor.sample((obs.to(self.device),r_obs.reshape(self.batch_size, 1, -1).to(self.device),))
            get_log_prob = f_actor.get_log_prob((obs.to(self.device),r_obs.reshape(self.batch_size, 1, -1).to(self.device),),act.squeeze().to(self.device),)
            loss_act = (alpha * log_prob - get_log_prob).mean()
            diffopt.step(loss_act)
            with torch.no_grad():
                act_target, log_prob, _, _= f_actor.sample((obs_val.to(self.device),r_obs_val.reshape(self.batch_size, 1, -1).to(self.device),))
            get_log_prob = f_actor.get_log_prob((obs_val.to(self.device),r_obs_val.reshape(self.batch_size, 1, -1).to(self.device),),act_val.squeeze().to(self.device),)
            policy_loss_val = (alpha * log_prob - get_log_prob).mean()

            loss_auxiliary = self.model.meta_critic(
                (
                    obs.reshape(self.batch_size, -1).to(self.device),
                    r_obs.reshape(self.batch_size, -1).to(self.device),
                ),
                act.squeeze().to(self.device),
                other_output.reshape(self.batch_size, -1).to(self.device),
            )
            # concat = torch.concat([obs.reshape(self.batch_size, -1).to(self.device), r_obs.reshape(self.batch_size, -1).to(self.device), act.squeeze().to(self.device),other_output.reshape(self.batch_size, -1).to(self.device),], dim=1)
            # loss_auxiliary = self.model.meta_critic(concat)

            diffopt.step(loss_auxiliary)
            with torch.no_grad():
                act_target, log_prob, _, _= f_actor.sample((obs_val.to(self.device),r_obs_val.reshape(self.batch_size, 1, -1).to(self.device),),)
            get_log_prob = f_actor.get_log_prob((obs_val.to(self.device),r_obs_val.reshape(self.batch_size, 1, -1).to(self.device),),act_val.squeeze().to(self.device),)
            policy_loss_val_new = (alpha * log_prob - get_log_prob).mean()

            total_actor_loss = loss_act + loss_auxiliary
            # total_actor_loss = loss_act
            # total_actor_loss = loss_auxiliary


        utility = policy_loss_val - policy_loss_val_new
        utility = torch.tanh(utility)
        loss_meta = -utility
        self.meta_critic_optimizer.zero_grad()
        grad_omega = torch.autograd.grad(loss_meta, self.model.meta_critic.parameters(),allow_unused=True)
        for gradient, variable in zip(grad_omega, self.model.meta_critic.parameters()):
            variable.grad = gradient
        self.meta_critic_optimizer.step()
        lm = loss_meta.data.item()

        self.actor_optimizer.zero_grad()
        total_actor_loss.backward()
        self.actor_optimizer.step()
        la = loss_act.data.item()

        if data_for_logging is not None:
            data_for_logging[0].log(
                {
                    "loss/actor": la,
                    "loss/meta": lm,
                },
                step=data_for_logging[1],
            )


    def update_target(self):
        for param, target_param in zip(
            self.model.parameters(), self.target.parameters()
        ):
            target_param.data.mul_(self.polyak)
            target_param.data.add_((1 - self.polyak) * param.data)



class MetaCriticFQL:
    def __init__(
        self,
        model,
        replay_buffer,
        replay_buffer_val,
        actor_optimizer,
        critic_optimizer,
        meta_critic_optimizer,
        bc_flow_optimizer,
        batch_size,
        flow_steps=10,
        lr=3e-4,
        polyak=0.995,
        gamma=0.9,
        beta=0.3,
        alpha=1.0,
    ):
        self.alg_name = "MetaCriticFQL"
        self.model = model
        self.target = copy.deepcopy(model)
        self.replay_buffer = replay_buffer
        self.replay_buffer_val = replay_buffer_val
        self.actor_optimizer = actor_optimizer
        self.critic_optimizer = critic_optimizer
        self.meta_critic_optimizer = meta_critic_optimizer
        self.bc_flow_optimizer = bc_flow_optimizer
        self.batch_size = batch_size
        self.polyak = polyak
        self.gamma = torch.as_tensor([gamma])
        self.beta = torch.as_tensor([beta])
        self.alpha = torch.as_tensor([alpha])
        self.lr =lr
        self.device = model.device

        self.updater = VirtualActorUpdater()
        self.flow_steps = flow_steps
        self.distill_only = False

    def safe_exp(self, x):
        return torch.where(x < 50, torch.exp(x), x)

    def update(self, data_for_logging=None):
        sample = self.replay_buffer.sample(self.batch_size)
        obs, next_obs, r_obs, next_r_obs, act, rwd, done = list(sample.values())

        sample_val = self.replay_buffer_val.sample(self.batch_size)
        obs_val, next_obs_val, r_obs_val, next_r_obs_val, act_val, rwd_val, done_val = list(sample_val.values())
        
        z = torch.randn((act.squeeze().shape[0], act.squeeze().shape[1]), device=self.device) 
        next_act_target, other_output= self.model.actor.sample_one_step_action(
            (
                next_obs.to(self.device),
                next_r_obs.reshape(self.batch_size, 1, -1).to(self.device),
            ),
            noise=z,
        )

        with torch.no_grad():
            Q_target_1, Q_target_2 = self.target.critic(
                (next_obs.to(self.device), next_r_obs.to(self.device)), next_act_target.detach()
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
        )

        loss_critic = F.mse_loss(Q_target, Q1) + F.mse_loss(Q_target, Q2)
        self.critic_optimizer.zero_grad()
        loss_critic.backward()
        self.critic_optimizer.step()
        lc = loss_critic.data.item()

        if data_for_logging is not None:
            data_for_logging[0].log(
                {
                    "loss/critic": lc,
                },
                step=data_for_logging[1],
            )
        ###############################
        ### compute actor loss      ###
        ###############################
        x_0 = torch.randn((act.squeeze().shape[0], act.squeeze().shape[1]), device=self.device)
        x_1 = act.squeeze().to(self.device)
        t = torch.rand((act.shape[0], 1), device=self.device)
        x_t = (1 - t) * x_0 + t * x_1
        vel = x_1 - x_0
        pred = self.model.bc_flow((obs.to(self.device), r_obs.to(self.device)), t, x_t)
        bc_flow_loss = F.mse_loss(pred, vel)
        self.bc_flow_optimizer.zero_grad()
        bc_flow_loss.backward()
        self.bc_flow_optimizer.step()



        with higher.innerloop_ctx(self.model.actor,self.actor_optimizer,copy_initial_weights=False) as (f_actor, diffopt):
            z = torch.randn((act.squeeze().shape[0], act.squeeze().shape[1]), device=self.device)
            target_flow_actions = f_actor.sample_flow_step_action((obs.to(self.device),r_obs.reshape(self.batch_size, 1, -1).to(self.device),),bc_flow=self.model.bc_flow, noise=z,flow_steps=self.flow_steps)
            actor_actions, _ = f_actor.sample_one_step_action((obs.to(self.device),r_obs.reshape(self.batch_size, 1, -1).to(self.device),),noise=z)
            distill_loss = F.mse_loss(actor_actions, target_flow_actions)
            ld = distill_loss.data.item()

            q1, q2 = self.model.critic((obs.to(self.device), r_obs.to(self.device)),act=actor_actions)
            q_min = torch.min(torch.cat((q1, q2), 1), dim=1)[0].reshape((-1, 1))

            if self.distill_only:
                loss_act = (self.alpha.to(self.device) * distill_loss).mean()
            else:
                loss_act = (- q_min + self.alpha.to(self.device) * distill_loss).mean()

            diffopt.step(loss_act)
            
            z = torch.randn((act_val.squeeze().shape[0], act_val.squeeze().shape[1]), device=self.device)
            target_flow_actions = f_actor.sample_flow_step_action((obs_val.to(self.device),r_obs_val.reshape(self.batch_size, 1, -1).to(self.device),),bc_flow=self.model.bc_flow, noise=z,flow_steps=self.flow_steps)
            actor_actions_old, _ = f_actor.sample_one_step_action((obs_val.to(self.device),r_obs_val.reshape(self.batch_size, 1, -1).to(self.device),),noise=z)
            distill_loss = F.mse_loss(actor_actions_old, target_flow_actions)
            q1, q2 = self.model.critic((obs_val.to(self.device), r_obs_val.to(self.device)),act=actor_actions_old,)
            q_min = torch.min(torch.cat((q1, q2), 1), dim=1)[0].reshape((-1, 1))
            policy_loss_val = (- q_min + self.alpha.to(self.device) * distill_loss).mean()

            loss_auxiliary = self.model.meta_critic(
                (
                    obs.reshape(self.batch_size, -1).to(self.device),
                    r_obs.reshape(self.batch_size, -1).to(self.device),
                ),
                act.squeeze().to(self.device),
                other_output.reshape(self.batch_size, -1).to(self.device),
            )

            diffopt.step(loss_auxiliary)

            z = torch.randn((act_val.squeeze().shape[0], act_val.squeeze().shape[1]), device=self.device)
            target_flow_actions = f_actor.sample_flow_step_action((obs_val.to(self.device),r_obs_val.reshape(self.batch_size, 1, -1).to(self.device),),bc_flow=self.model.bc_flow, noise=z,flow_steps=self.flow_steps)
            actor_actions_new, _ = f_actor.sample_one_step_action((obs_val.to(self.device),r_obs_val.reshape(self.batch_size, 1, -1).to(self.device),),noise=z)
            distill_loss = F.mse_loss(actor_actions_new, target_flow_actions)
            q1, q2 = self.model.critic((obs_val.to(self.device), r_obs_val.to(self.device)),act=actor_actions_new)
            q_min = torch.min(torch.cat((q1, q2), 1), dim=1)[0].reshape((-1, 1))
            policy_loss_val_new = (- q_min + self.alpha.to(self.device) * distill_loss).mean()

            total_actor_loss = loss_act + loss_auxiliary
            
        utility = policy_loss_val - policy_loss_val_new
        utility = torch.tanh(utility)
        loss_meta = -utility
        self.meta_critic_optimizer.zero_grad()
        grad_omega = torch.autograd.grad(loss_meta, self.model.meta_critic.parameters(),allow_unused=True)
        for gradient, variable in zip(grad_omega, self.model.meta_critic.parameters()):
            variable.grad = gradient
        self.meta_critic_optimizer.step()
        lm = loss_meta.data.item()
            
        loss_act = loss_act + loss_auxiliary
        self.actor_optimizer.zero_grad()
        total_actor_loss.backward()
        self.actor_optimizer.step()
        la = loss_act.data.item()

        if data_for_logging is not None:
            data_for_logging[0].log(
                {
                    "loss/actor": la,
                    "loss/meta": lm,
                },
                step=data_for_logging[1],
            )

    def update_target(self):
        for param, target_param in zip(
            self.model.parameters(), self.target.parameters()
        ):
            target_param.data.mul_(self.polyak)
            target_param.data.add_((1 - self.polyak) * param.data)
