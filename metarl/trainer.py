import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from metarl.utils import Hot_Plug, l1_penalty

from .weight_clipping import WeightClippingSGD, WeightClippingAdam

from .utils import (
    get_grad_norm, 
    get_weight_norm, 
    get_abs_approximate_rank, 
    get_approximate_rank, 
    get_dormant_units_ratio, 
    get_effective_rank,
    get_kl_divergence,
    get_mmd,
    get_wasserstein_dist,
    get_grad_direction_stats
    )

class MetaAWAC:
    def __init__(
        self,
        model,
        replay_buffer,
        actor_optimizer,
        critic_optimizer,
        meta_critic_optimizer,
        batch_size,
        polyak=0.995,
        alpha=0.2,
        beta=0.3,
        gamma=0.9,
    ):
        self.alg_name = "MetaAWAC"
        self.model = model
        self.target = copy.deepcopy(model)
        self.replay_buffer = replay_buffer
        self.actor_optimizer = actor_optimizer
        self.critic_optimizer = critic_optimizer
        self.meta_critic_optimizer = meta_critic_optimizer
        self.batch_size = batch_size
        self.polyak = polyak
        self.gamma = gamma
        self.beta = beta

        self.alpha = alpha

        feature_net = nn.Sequential(*list(self.model.actor.children())[:-2])

        self.hotplug = Hot_Plug(feature_net)

        # Weight Clipping
        self.lr = 3e-4
        weight_clipping = 0.5
        clip_last_layer = 1
        self.actor_optimizer = WeightClippingSGD(self.model.actor.parameters(), lr=self.lr, zeta=weight_clipping, clip_last_layer=clip_last_layer)
        # self.actor_optimizer = WeightClippingAdam(self.model.actor.parameters(), lr=self.lr, eps=1e-5, zeta=weight_clipping, clip_last_layer=clip_last_layer)

        # self.critic_optimizer = WeightClippingAdam(self.model.critic.parameters(), lr=self.lr, eps=1e-5, zeta=weight_clipping, clip_last_layer=clip_last_layer)


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

        self.device = model.device

    def safe_exp(self, x):
        return torch.where(x < 50, torch.exp(x), x)

    def calc_weights(self, act, obs, r_obs, Q1, Q2):
        with torch.no_grad():
            qw_ref = torch.min(torch.cat((Q1, Q2), 1), dim=1)[0].reshape((-1, 1))

            v_act1, v_act2 = self.model.critic(
                (obs.to(self.device), r_obs.to(self.device)),
                act.detach(),
            )

            qw_gen = torch.min(torch.cat((v_act1, v_act2), 1), dim=1)[0].reshape(
                (-1, 1)
            )

            adv = torch.max(torch.zeros_like(qw_ref), qw_ref - qw_gen)
            # adv = qw_ref - qw_gen
            # weights = F.softmax(adv / beta, dim=0)
            weights = self.safe_exp(adv / self.beta)

        return weights

    def update(self, update_actor=True, data_for_logging=None):
        sample = self.replay_buffer.sample(self.batch_size)
        obs, next_obs, r_obs, next_r_obs, act, rwd, done = list(sample.values())

        sample_val = self.replay_buffer.sample(self.batch_size)
        obs_val, next_obs_val, r_obs_val, next_r_obs_val, _, _, _ = list(
            sample_val.values()
        )

        with torch.no_grad():
            next_act_target, next_log_prob, _, _ = self.model.actor.sample(
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
        critic_weight_norm = get_weight_norm(self.model.critic)
        self.critic_optimizer.step()
        lc = loss_critic.data.item()

        if data_for_logging is not None:
            data_for_logging[0].log(
                {
                    "loss/critic": lc,
                },
                step=data_for_logging[1],
            )

        if update_actor:
            action_gen, log_pi_gen, _, pi_bar = self.model.actor.sample(
                (
                    obs.to(self.device),
                    r_obs.reshape(self.batch_size, 1, -1).to(self.device),
                )
            )
            with torch.no_grad():
                Q1_a, Q2_a = self.model.critic(
                    (obs.to(self.device), r_obs.to(self.device)),
                    act.squeeze().to(self.device),
                )
                weights = self.calc_weights(action_gen, obs, r_obs, Q1_a, Q2_a)

            get_log_prob = self.model.actor.get_log_prob(
                    (
                        obs.to(self.device),
                        r_obs.reshape(self.batch_size, 1, -1).to(self.device),
                    ),
                    act.squeeze().to(self.device),
                )
            loss_act = -(get_log_prob* weights).mean()
        
            # self.actor_optimizer.zero_grad()
            # loss_act.backward()

            # self.actor_optimizer.step()

            # qf1_pi, qf2_pi = self.model.critic(
            #     (obs.to(self.device), r_obs.to(self.device)),
            #     action_gen.squeeze().to(self.device),
            # )
            # min_qf_pi = torch.min(qf1_pi, qf2_pi)

            # bc_loss = -(
            #     self.model.actor.get_log_prob(
            #         (
            #             obs.to(self.device),
            #             r_obs.reshape(self.batch_size, 1, -1).to(self.device),
            #         ),
            #         act.squeeze().to(self.device),
            #     )
            # ).mean()

            # loss_act = ((self.alpha * log_pi_gen) - min_qf_pi).mean() + bc_loss

            ###############################################################################
            ##############################    PART 1        ###############################
            ###############################################################################
            ###############################################################################
            # concat_output = torch.cat([other_output, state_batch, action_batch], 1)
            # loss_auxiliary = self.feature_critic(concat_output)
            loss_auxiliary = self.model.meta_critic(
                pi_bar,
                (obs_val.to(self.device), r_obs_val.to(self.device)),
                act.squeeze().to(self.device),
            )

            self.actor_optimizer.zero_grad()
            loss_act.backward(retain_graph=True)
            self.hotplug.update(self.actor_optimizer.param_groups[0]["lr"])

            # pi_val, log_pi_val, *_ = self.policy.sample(state_batch_val)

            pi_val, log_pi_val, *_ = self.model.actor.sample(
                (
                    obs_val.to(self.device),
                    r_obs_val.reshape(self.batch_size, 1, -1).to(self.device),
                )
            )

            # qf1_pi_val, qf2_pi_val = self.critic(state_batch_val, pi_val)
            qf1_pi_val, qf2_pi_val = self.model.critic(
                (obs_val.to(self.device), r_obs_val.to(self.device)),
                pi_val.squeeze().to(self.device),
            )

            min_qf_pi_val = torch.min(qf1_pi_val, qf2_pi_val)

            policy_loss_val = ((self.alpha * log_pi_val) - min_qf_pi_val).mean()
            # policy_loss_val = -min_qf_pi_val.mean()

            ###############################################################################
            ##############################    PART 2        ###############################
            ###############################################################################
            ###############################################################################
            loss_auxiliary.backward(create_graph=True)
            # loss_auxiliary.backward()
            laux = loss_auxiliary.data.item()
            abs_theta = 0.0
            for p in self.param_optim_theta:
                abs_theta += l1_penalty(p._grad.data).item()

            self.hotplug.update(self.actor_optimizer.param_groups[0]["lr"])

            # pi_val_new, log_pi_val_new, *_ = self.policy.sample(state_batch_val)
            pi_val_new, log_pi_val_new, *_ = self.model.actor.sample(
                (
                    obs_val.to(self.device),
                    r_obs_val.reshape(self.batch_size, 1, -1).to(self.device),
                )
            )

            # qf1_pi_val_new, qf2_pi_val_new = self.critic(state_batch_val, pi_val_new)
            qf1_pi_val_new, qf2_pi_val_new = self.model.critic(
                (obs_val.to(self.device), r_obs_val.to(self.device)),
                pi_val_new.squeeze().to(self.device),
            )

            min_qf_pi_val_new = torch.min(qf1_pi_val_new, qf2_pi_val_new)

            policy_loss_val_new = (
                (self.alpha * log_pi_val_new) - min_qf_pi_val_new
            ).mean()

            # policy_loss_val_new = -min_qf_pi_val_new.mean()

            utility = policy_loss_val - policy_loss_val_new
            utility = torch.tanh(utility)
            loss_meta = -utility

            self.meta_critic_optimizer.zero_grad(set_to_none=False)
            grad_omega = torch.autograd.grad(
                loss_meta,
                self.model.meta_critic.parameters(),
            )
            for gradient, variable in zip(
                grad_omega, self.model.meta_critic.parameters()
            ):
                variable.grad.data = gradient
            
            self.meta_critic_optimizer.step()
            meta_critic_weight_norm = get_weight_norm(self.model.meta_critic)
            lm = loss_meta.data.item()
            actor_weight_norm = get_weight_norm(self.model.actor)
            self.actor_optimizer.step()
            self.hotplug.restore()

            la = loss_act.data.item()

        with torch.no_grad(): 
            if data_for_logging is not None:
                data_for_logging[0].log(
                    {
                        "loss/actor": la,
                        "loss/critic": lc,
                        "loss/auxiliary": laux,
                        "loss/meta": lm,
                        "diag/Q": ((Q1 + Q2) /2).mean().item(),
                        "diag/log_prob": get_log_prob.mean().data.item(),
                        "diag/log_prob_old": log_pi_val.mean().data.item(),
                        "diag/log_prob_new": log_pi_val_new.mean().data.item(),
                        # "diag/actor_grad_norm":actor_grad_norm,
                        "diag/actor_weight_norm":actor_weight_norm,
                        # "diag/critic_grad_norm":critic_grad_norm,
                        "diag/critic_weight_norm":critic_weight_norm,
                        # "diag/meta_critic_grad_norm":meta_critic_grad_norm,
                        "diag/meta_critic_weight_norm":meta_critic_weight_norm,
                        # "diag/actor_kl_divergence": kl_div,
                        # "diag/critic_wasserstein_dist": wass_dist,
                        # "diag/critic_mmd": mmd_val,
                        # "diag/grad_diversity_actor": grad_diversity_actor,
                        # "diag/grad_diversity_critic": grad_diversity_critic,
                        # "diag/grad_diversity_meta_critic": grad_diversity_meta_critic,
                        # "plasticity/critic_effective_rank": get_effective_rank(c_feats),
                        # "plasticity/critic_approx_rank": get_approximate_rank(c_feats),
                        # "plasticity/critic_abs_approx_rank": get_abs_approximate_rank(c_feats),
                        # "plasticity/critic_dormant_ratio": get_dormant_units_ratio(c_feats),
                        # "plasticity/actor_effective_rank": get_effective_rank(a_feats),
                        # "plasticity/actor_approx_rank": get_approximate_rank(a_feats),
                        # "plasticity/actor_abs_approx_rank": get_abs_approximate_rank(a_feats),
                        # "plasticity/actor_dormant_ratio": get_dormant_units_ratio(a_feats),
                    },
                    step=data_for_logging[1],
                )

    def update_target(self):
        for param, target_param in zip(
            self.model.parameters(), self.target.parameters()
        ):
            target_param.data.mul_(self.polyak)
            target_param.data.add_((1 - self.polyak) * param.data)
