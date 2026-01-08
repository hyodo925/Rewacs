import copy

import torch
import torch.nn.functional as F
from tqdm import tqdm


class TD3BC:
    def __init__(
        self,
        model,
        replay_buffer,
        actor_optimizer,
        critic_optimizer,
        batch_size,
        polyak=0.995,
        gamma=0.9,
        alpha=1.0,
        beta=0.3,
    ):
        self.alg_name = "TD3_BC"
        self.model = model
        self.target = copy.deepcopy(model)
        self.replay_buffer = replay_buffer
        self.actor_optimizer = actor_optimizer
        self.critic_optimizer = critic_optimizer
        self.batch_size = batch_size
        self.polyak = polyak
        self.gamma = torch.as_tensor([gamma])
        self.alpha = torch.as_tensor([alpha])
        self.beta = torch.as_tensor([beta])

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

        if update_actor:
            action_gen, _, _ = self.model.actor.sample(
                (
                    obs.to(self.device),
                    r_obs.reshape(self.batch_size, 1, -1).to(self.device),
                ),
            )
            q1, q2 = self.model.critic((obs.to(self.device), r_obs.to(self.device)),act=act.squeeze().to(self.device),)
            q_values = torch.min(torch.cat((q1, q2), 1), dim=1)[0].reshape((-1, 1))
            q_values = q_values[0] 
            lmbda = self.alpha / q_values.abs().mean().detach()
            loss_act = -lmbda * q_values.mean() + 0.5 * torch.mean((action_gen - act.to(self.device)) ** 2)
            self.actor_optimizer.zero_grad()
            loss_act.backward()
            self.actor_optimizer.step()
            la = loss_act.data.item()


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
