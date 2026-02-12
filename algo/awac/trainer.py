import copy

import torch
import torch.nn.functional as F
from tqdm import tqdm
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
    get_grad_direction_stats,
    )
from .weight_clipping import WeightClippingAdam
from torchviz import make_dot

class AWAC:
    def __init__(
        self,
        model,
        replay_buffer,
        actor_optimizer,
        critic_optimizer,
        batch_size,
        polyak=0.995,
        gamma=0.9,
        beta=0.3,
    ):
        self.alg_name = "AWAC"
        self.model = model
        self.target = copy.deepcopy(model)
        self.replay_buffer = replay_buffer
        self.actor_optimizer = actor_optimizer
        self.critic_optimizer = critic_optimizer
        self.batch_size = batch_size
        self.polyak = polyak
        self.grad_clip = 1e9
        self.device = model.device
        self.gamma = torch.as_tensor([gamma]).to(self.device)
        self.beta = torch.as_tensor([beta]).to(self.device)


        #Weight Clipping
        # self.lr = 3e-4
        # weight_clipping = 0.5
        # clip_last_layer = 1
        # # self.weight_clipping = WeightClippingAdam()
        # self.actor_optimizer = WeightClippingAdam(self.model.actor.parameters(), lr=self.lr, eps=1e-5, zeta=weight_clipping, clip_last_layer=clip_last_layer)
        # self.critic_optimizer = WeightClippingAdam(self.model.critic.parameters(), lr=self.lr, eps=1e-5, zeta=weight_clipping, clip_last_layer=clip_last_layer)

    def safe_exp(self, x):
        return torch.where(x < 50, torch.exp(x), x)

    def update(self, update_actor=False, data_for_logging=None):
        sample = self.replay_buffer.sample(self.batch_size)
        obs = sample["humans_obs"]
        next_obs = sample["next_humans_obs"] # または prev_obs
        r_obs = sample["robot_obs"]
        next_r_obs = sample["next_robot_obs"]
        act = sample["action"]
        rwd = sample["reward"]
        done = sample["done"]
        # print(sample["_weight"])
        with torch.no_grad():
            old_mean, old_log_std = self.model.actor((obs.to(self.device), r_obs.reshape(self.batch_size, 1, -1).to(self.device)))
            c_feats = self.model.critic.integrator(obs.to(self.device), r_obs.reshape(self.batch_size, 1, -1).to(self.device))
            if not self.model.critic.single:
                old_c_feats = torch.cat([c_feats, act.squeeze().to(self.device)], -1)

        with torch.no_grad():
            next_act_target, next_log_prob, _ = self.model.actor.sample((next_obs.to(self.device),next_r_obs.reshape(self.batch_size, 1, -1).to(self.device),))
            Q_target_1, Q_target_2 = self.target.critic((next_obs.to(self.device), next_r_obs.to(self.device)), next_act_target)
            Q_target_min = torch.min(torch.cat((Q_target_1, Q_target_2), 1), dim=1)[0].unsqueeze(-1)
            Q_target = rwd.to(self.device) + (self.gamma * Q_target_min) * done.to(self.device)
        Q1, Q2 = self.model.critic((obs.to(self.device), r_obs.to(self.device)),act.squeeze().to(self.device),)
        loss_critic = F.mse_loss(Q_target, Q1) + F.mse_loss(Q_target, Q2)
        self.critic_optimizer.zero_grad()
        loss_critic.backward()
        grad_diversity_critic = get_grad_direction_stats(self.model.critic)
        critic_grad_norm = get_grad_norm(self.model.critic)
        critic_weight_norm = get_weight_norm(self.model.critic)
        self.critic_optimizer.step()
        lc = loss_critic.data.item()

        if data_for_logging is not None:
            data_for_logging[0].log(
                {
                    "loss/critic": lc,
                    "diag/critic_grad_norm":critic_grad_norm,
                    "diag/critic_weight_norm":critic_weight_norm,
                },
                step=data_for_logging[1],
            )

        if update_actor:
            with torch.no_grad():
                action_gen, _, _ = self.model.actor.sample((obs.to(self.device),r_obs.reshape(self.batch_size, 1, -1).to(self.device),))
                qw_ref = torch.min(torch.cat((Q1, Q2), 1), dim=1)[0].reshape((-1, 1))
                v_act1, v_act2 = self.model.critic((obs.to(self.device), r_obs.to(self.device)),action_gen.detach(),)
                qw_gen = torch.min(torch.cat((v_act1, v_act2), 1), dim=1)[0].reshape((-1, 1))
                adv = torch.max(torch.zeros_like(qw_ref), qw_ref - qw_gen)
                # adv = qw_ref - qw_gen
                # weights = F.softmax(adv / beta, dim=0)
                weights = self.safe_exp(adv / self.beta)
            
            log_prob = self.model.actor.get_log_prob((obs.to(self.device),r_obs.reshape(self.batch_size, 1, -1).to(self.device),),act.squeeze().to(self.device),)
            loss_act = -(log_prob* weights).mean()
            self.actor_optimizer.zero_grad()
            loss_act.backward()
            grad_diversity_actor = get_grad_direction_stats(self.model.actor)
            actor_grad_norm = get_grad_norm(self.model.actor)
            actor_weight_norm = get_weight_norm(self.model.actor)

            self.actor_optimizer.step()
            la = loss_act.data.item()


            with torch.no_grad():
                new_mean, new_log_std = self.model.actor((obs.to(self.device), r_obs.reshape(self.batch_size, 1, -1).to(self.device)))
                a_feats = self.model.actor.integrator(obs.to(self.device), r_obs.reshape(self.batch_size, 1, -1).to(self.device))
                c_feats = self.model.critic.integrator(obs.to(self.device), r_obs.reshape(self.batch_size, 1, -1).to(self.device))
                if not self.model.critic.single:
                    new_c_feats = torch.cat([c_feats, act.squeeze().to(self.device)], -1)
            
                kl_div = get_kl_divergence((old_mean, old_log_std), (new_mean, new_log_std))
                wass_dist = get_wasserstein_dist(old_c_feats, new_c_feats)
                mmd_val = get_mmd(old_c_feats, new_c_feats)
                if data_for_logging is not None:
                    data_for_logging[0].log(
                        {
                            "loss/actor": la,
                            "diag/log_prob": log_prob.mean().data.item(),
                            "diag/Q": ((Q1 + Q2) /2).mean().item(),
                            "diag/actor_grad_norm":actor_grad_norm,
                            "diag/actor_weight_norm":actor_weight_norm,
                            "diag/actor_kl_divergence": kl_div,
                            "diag/critic_wasserstein_dist": wass_dist,
                            "diag/critic_mmd": mmd_val,
                            "diag/grad_diversity_actor": grad_diversity_actor,
                            "diag/grad_diversity_critic": grad_diversity_critic,
                            "plasticity/critic_effective_rank": get_effective_rank(c_feats),
                            "plasticity/critic_approx_rank": get_approximate_rank(c_feats),
                            "plasticity/critic_abs_approx_rank": get_abs_approximate_rank(c_feats),
                            "plasticity/critic_dormant_ratio": get_dormant_units_ratio(c_feats),
                            "plasticity/actor_effective_rank": get_effective_rank(a_feats),
                            "plasticity/actor_approx_rank": get_approximate_rank(a_feats),
                            "plasticity/actor_abs_approx_rank": get_abs_approximate_rank(a_feats),
                            "plasticity/actor_dormant_ratio": get_dormant_units_ratio(a_feats),
                        },
                        step=data_for_logging[1],
                    )


    def update_target(self):
        for param, target_param in zip(
            self.model.parameters(), self.target.parameters()
        ):
            target_param.data.mul_(self.polyak)
            target_param.data.add_((1 - self.polyak) * param.data)


class AWACMultiTask:
    def __init__(
        self,
        model,
        tasks,
        actor_optimizer,
        critic_optimizer,
        batch_size,
        polyak=0.995,
        gamma=0.9,
        beta=0.3,
    ):
        self.alg_name = "AWACMultiTask"
        self.model = model
        self.tasks = tasks
        self.target = copy.deepcopy(model)
        self.actor_optimizer = actor_optimizer
        self.critic_optimizer = critic_optimizer
        self.batch_size = batch_size
        self.polyak = polyak
        self.grad_clip = 1e9
        self.device = model.device
        self.gamma = torch.as_tensor([gamma]).to(self.device)
        self.beta = torch.as_tensor([beta]).to(self.device)

        

    def safe_exp(self, x):
        return torch.where(x < 50, torch.exp(x), x)

    def step(self, buffer, update_actor=False, data_for_logging=None):
        sample = buffer.sample(self.batch_size)
        obs, next_obs, r_obs, next_r_obs, act, rwd, done = list(sample.values())

        with torch.no_grad():
            next_act_target, next_log_prob, _ = self.model.actor.sample(
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

        if update_actor:
            with torch.no_grad():
                action_gen, _, _ = self.model.actor.sample(
                    (
                        obs.to(self.device),
                        r_obs.reshape(self.batch_size, 1, -1).to(self.device),
                    )
                )

                qw_ref = torch.min(torch.cat((Q1, Q2), 1), dim=1)[0].reshape((-1, 1))

                v_act1, v_act2 = self.model.critic(
                    (obs.to(self.device), r_obs.to(self.device)),
                    action_gen.detach(),
                )

                qw_gen = torch.min(torch.cat((v_act1, v_act2), 1), dim=1)[0].reshape(
                    (-1, 1)
                )

                adv = torch.max(torch.zeros_like(qw_ref), qw_ref - qw_gen)
                # adv = qw_ref - qw_gen
                # weights = F.softmax(adv / beta, dim=0)
                weights = self.safe_exp(adv / self.beta)
            
            log_prob = self.model.actor.get_log_prob(
                    (
                        obs.to(self.device),
                        r_obs.reshape(self.batch_size, 1, -1).to(self.device),
                    ),
                    act.squeeze().to(self.device),
                )
            loss_act = -(log_prob* weights).mean()
            if data_for_logging is not None:
                data_for_logging[0].log(
                    {
                        "log_prob": log_prob.mean().data.item(),
                    },
                    step=data_for_logging[1],
                )

        
        return loss_critic, loss_act
    
    def update(self, update_actor=False, data_for_logging=None):
        loss_critic = 0
        loss_act = 0
        for task in self.tasks:
            lc, la = self.step(task, update_actor, data_for_logging)
            loss_critic += lc
            loss_act += la 

        self.critic_optimizer.zero_grad()
        (loss_critic / len(self.tasks)).backward()
        grad_critic = torch.nn.utils.clip_grad_norm_(self.model.critic.parameters(), self.grad_clip)
        self.critic_optimizer.step()
        lc = loss_critic.data.item()

        self.actor_optimizer.zero_grad()
        (loss_act / len(self.tasks)).backward()
        grad_actor = torch.nn.utils.clip_grad_norm_(self.model.actor.parameters(), self.grad_clip)
        self.actor_optimizer.step()
        la = loss_act.data.item()

        if data_for_logging is not None:
            data_for_logging[0].log(
                {
                    "loss/critic": lc,
                    "loss/actor": la,
                    "grad/actor":grad_actor,
                    "grad/critic":grad_critic,
                },
                step=data_for_logging[1],
            )

        return loss_critic, loss_act


    def update_target(self):
        for param, target_param in zip(
            self.model.parameters(), self.target.parameters()
        ):
            target_param.data.mul_(self.polyak)
            target_param.data.add_((1 - self.polyak) * param.data)
