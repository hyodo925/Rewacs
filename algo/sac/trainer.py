import copy

import torch
import torch.nn.functional as F
from tqdm import tqdm


class SAC:
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
        self.alg_name = "SAC"
        self.model = model
        self.target = copy.deepcopy(model)
        self.replay_buffer = replay_buffer
        self.actor_optimizer = actor_optimizer
        self.critic_optimizer = critic_optimizer
        self.alpha_optimizer = alpha_optimizer
        self.batch_size = batch_size
        self.polyak = polyak
        self.gamma = torch.as_tensor([gamma])
        self.beta = torch.as_tensor([beta])
        self.alpha = torch.as_tensor([alpha])
        self.auto_entropy_tuning = False
        target_entropy=None
        self.device = model.device

        if self.auto_entropy_tuning:
            if target_entropy is None:
                self.target_entropy = -torch.prod(torch.Tensor([self.act_dim]).to(self.device)).item()
            else:
                self.target_entropy = target_entropy
            self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
            self.alpha_optimizer = alpha_optimizer
            self.alpha = self.log_alpha.exp()
        else:
            self.alpha = torch.tensor(self.alpha).to(self.device)

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

        action_gen, log_prob, _ = self.model.actor.sample(
            (
                obs.to(self.device),
                r_obs.reshape(self.batch_size, 1, -1).to(self.device),
            )
        )

        Q1, Q2 = self.critic(
            (obs.to(self.device), r_obs.to(self.device)), action_gen
        )
        Q_min = torch.min(torch.cat((Q1, Q2), 1), dim=1)[
            0
        ].unsqueeze(-1)

        loss_act = ((self.alpha * log_prob) - Q_min).mean()
        self.actor_optimizer.zero_grad()
        loss_act.backward()
        self.actor_optimizer.step()
        la = loss_act.data.item()

        if self.auto_entropy_tuning:
            alpha_loss = -(self.log_alpha * (log_prob + self.target_entropy).detach()).mean()
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()
            self.alpha = self.log_alpha.exp()

            if data_for_logging is not None:
                data_for_logging[0].log(
                    {
                        "loss/actor": la,
                    },
                    step=data_for_logging[1],
                )

    def update_target(self):
        for param, target_param in zip(
            self.model.parameters(), self.target.parameters()
        ):
            target_param.data.mul_(self.polyak)
            target_param.data.add_((1 - self.polyak) * param.data)
