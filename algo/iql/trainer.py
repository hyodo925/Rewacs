import copy

import torch
import torch.nn.functional as F
from tqdm import tqdm


class IQL:
    def __init__(
        self,
        model,
        replay_buffer,
        actor_optimizer,
        critic_optimizer,
        value_optimizer,
        batch_size,
        polyak=0.995,
        expectile = 0.8,
        gamma=0.9,
        beta=0.3,
    ):
        self.alg_name = "IQL"
        self.model = model
        self.target = copy.deepcopy(model)
        self.replay_buffer = replay_buffer
        self.actor_optimizer = actor_optimizer
        self.critic_optimizer = critic_optimizer
        self.value_optimizer = value_optimizer
        self.batch_size = batch_size
        self.polyak = polyak
        self.expectile = torch.as_tensor([expectile])
        self.gamma = torch.as_tensor([gamma])
        self.beta = torch.as_tensor([beta])

        self.device = model.device

    def safe_exp(self, x):
        return torch.where(x < 50, torch.exp(x), x)

    def update(self, update_actor=False, data_for_logging=None):
        sample = self.replay_buffer.sample(self.batch_size)
        obs, next_obs, r_obs, next_r_obs, act, rwd, done = list(sample.values())
        # with torch.no_grad():
        Q1, Q2 = self.model.critic(
            (obs.to(self.device), r_obs.to(self.device)),
            act.squeeze().to(self.device),
        )
        q_min = torch.min(torch.cat((Q1, Q2), 1), dim=1)[0].reshape((-1, 1))
        value = self.model.value(
            (
                obs.to(self.device),
                r_obs.reshape(self.batch_size, 1, -1).to(self.device),
            )
        )
        diff = q_min - value
        expectile_weight = torch.where(diff > 0, self.expectile, 1 - self.expectile)
        loss_value= (expectile_weight * (diff)**2).mean()
        self.value_optimizer.zero_grad()
        loss_value.backward()
        self.value_optimizer.step()
        lv = loss_value.data.item()

        with torch.no_grad():
            next_v = self.model.value(
                (
                    next_obs.to(self.device),
                    next_r_obs.reshape(self.batch_size, 1, -1).to(self.device),
                )
            )

            Q_target = rwd.to(self.device) + (self.gamma * next_v) * done.to(
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
                    "loss/value": lv
                },
                step=data_for_logging[1],
            )

        if update_actor:
            with torch.no_grad():
                v_ref = self.model.value(
                     (
                        obs.to(self.device),
                        r_obs.reshape(self.batch_size, 1, -1).to(self.device),
                    )
                )

                v_act1, v_act2 = self.model.critic(
                    (obs.to(self.device), r_obs.to(self.device)),
                    act.squeeze().to(self.device),
                )

                qw_ref = torch.min(torch.cat((v_act1, v_act2), 1), dim=1)[0].reshape(
                    (-1, 1)
                )

                adv = torch.max(torch.zeros_like(qw_ref), qw_ref - v_ref)
                # adv = qw_ref - qw_gen
                # weights = F.softmax(adv / beta, dim=0)
                weights = self.safe_exp(adv / self.beta)

            loss_act = -(
                self.model.actor.get_log_prob(
                    (
                        obs.to(self.device),
                        r_obs.reshape(self.batch_size, 1, -1).to(self.device),
                    ),
                    act.squeeze().to(self.device),
                )
                * weights
            ).mean()

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
