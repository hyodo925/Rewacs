import copy

import torch
import torch.nn.functional as F
import torch.nn as nn
from tqdm import tqdm
from .virtual_updater import Hot_Plug
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
        self.hotplug = Hot_Plug(feature_net)

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

            # loss_auxiliary = self.model.meta_critic(
            #     (
            #         obs.reshape(self.batch_size, -1).to(self.device),
            #         r_obs.reshape(self.batch_size, -1).to(self.device),
            #     ),
            #     act.squeeze().to(self.device),
            #     other_output.reshape(self.batch_size, -1).to(self.device),
            # )
            concat = torch.concat([obs.reshape(self.batch_size, -1).to(self.device), r_obs.reshape(self.batch_size, -1).to(self.device), act.squeeze().to(self.device),other_output.reshape(self.batch_size, -1).to(self.device),], dim=1)
            loss_auxiliary = self.model.meta_critic(concat)
            # loss_auxiliary = self.model.meta_critic(
            #     (
            #         obs.reshape(self.batch_size, -1).to(self.device),
            #         r_obs.reshape(self.batch_size, -1).to(self.device),
            #     ),
            #     act.squeeze().to(self.device),
            #     other_output.reshape(self.batch_size, -1).to(self.device),
            # )
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
        total_actor_loss.backward(retain_graph=True)
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
