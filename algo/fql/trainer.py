import copy

import torch
import torch.nn.functional as F
from tqdm import tqdm


class FQL:
    def __init__(
        self,
        model,
        replay_buffer,
        actor_optimizer,
        critic_optimizer,
        bc_flow_optimizer,
        batch_size,
        polyak=0.995,
        gamma=0.9,
        alpha=1.0,
        beta=0.3,
    ):
        self.alg_name = "FQL"
        self.model = model
        self.target = copy.deepcopy(model)
        self.replay_buffer = replay_buffer
        self.actor_optimizer = actor_optimizer
        self.critic_optimizer = critic_optimizer
        self.bc_flow_optimizer = bc_flow_optimizer
        self.batch_size = batch_size
        self.polyak = polyak
        self.gamma = torch.as_tensor([gamma])
        self.alpha = torch.as_tensor([alpha])
        self.beta = torch.as_tensor([beta])
        self.distill_only = False
        self.device = model.device

    def safe_exp(self, x):
        return torch.where(x < 50, torch.exp(x), x)

    def update(self, update_actor=False, data_for_logging=None):
        sample = self.replay_buffer.sample(self.batch_size)
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

        x_0 = torch.randn((act.shape[0], act.shape[1]), device=self.device)
        x_1 = act.to(self.device)
        t = torch.rand((act.shape[0], 1), device=self.device)
        x_t = (1 - t) * x_0 + t * x_1
        vel = x_1 - x_0
        
        # with torch.no_grad():
        next_obs_actions = self.model.actor.sample_one_step_action((obs.to(self.device),r_obs.reshape(-1, 1, self.r_obs_dim).to(self.device),),noise=z)
        q1, q2 = self.model.critic((obs.to(self.device), r_obs.to(self.device)),act=next_obs_actions.detach(),)
        q_min = torch.min(torch.cat((q1, q2), 1), dim=1)[0].reshape((-1, 1))
        #lmbda = self.alpha / q_values.abs().mean().detach()
        pred = self.model.bc_flow((obs.to(self.device), r_obs.to(self.device)), t, x_t)
        bc_flow_loss = F.mse_loss(pred, vel)
        self.bc_flow_optimizer.zero_grad()
        bc_flow_loss.backward()
        self.bc_flow_optimizer.step()

        z = torch.randn((act.shape[0], act.shape[1]), device=self.device)
        target_flow_actions = self.model.actor.sample_flow_step_action((obs.to(self.device),r_obs.reshape(-1, 1, self.r_obs_dim).to(self.device),),bc_flow=self.bc_flow, noise=z,flow_steps=self.flow_steps)
        actor_actions = self.model.actor.sample_one_step_action((obs.to(self.device),r_obs.reshape(-1, 1, self.r_obs_dim).to(self.device),),noise=z)
        distill_loss = F.mse_loss(actor_actions, target_flow_actions)
        ld = distill_loss.data.item()
        if self.distill_only:
            loss_act = (self.alpha * distill_loss).mean()
        else:
            loss_act = (- q_min + self.alpha * distill_loss).mean()
        self.actor_optimizer.zero_grad()
        loss_act.backward()
        self.actor_optimizer.step()
        la = loss_act.data.item()

        if data_for_logging is not None:
            data_for_logging[0].log(
                {
                    "loss/actor": la,
                    "loss/distill": ld,
                },
                step=data_for_logging[1],
            )

    def update_target(self):
        for param, target_param in zip(
            self.model.parameters(), self.target.parameters()
        ):
            target_param.data.mul_(self.polyak)
            target_param.data.add_((1 - self.polyak) * param.data)
