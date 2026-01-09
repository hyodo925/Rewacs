import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

class Scalar(nn.Module):
    def __init__(self, init_value: float):
        super().__init__()
        self.constant = nn.Parameter(torch.tensor(init_value, dtype=torch.float32))

    def forward(self) -> nn.Parameter:
        return self.constant
class CQL:
    def __init__(
        self,
        model,
        replay_buffer,
        actor_optimizer,
        critic_optimizer,
        alpha_optimizer,
        batch_size,
        action_dim,
        policy_lr,
        qf_lr,
        polyak=0.995,
        gamma=0.9,
        beta=0.3,
    ):
        self.alg_name = "CQL"
        self.model = model
        self.action_dim = action_dim
        self.policy_lr = policy_lr
        self.qf_lr = qf_lr
        self.target = copy.deepcopy(model)
        self.replay_buffer = replay_buffer
        self.actor_optimizer = actor_optimizer
        self.critic_optimizer = critic_optimizer
        self.alpha_optimizer = alpha_optimizer
        self.batch_size = batch_size
        self.polyak = polyak
        self.gamma = torch.as_tensor([gamma])
        self.beta = torch.as_tensor([beta])
        
        self.device = model.device

        self.cql_alpha = 10.0
        self.cql_temp = 1.0
        self.cql_lagrange = True
        self.use_automatic_entropy_tuning = True
        self.cql_importance_sample = True
        self.cql_target_action_gap = -1.0
        self.target_entropy = -np.prod(self.action_dim).item()
        self.cql_n_actions = 10
        self.alpha_multiplier = 1.0
        self.cql_clip_diff_min = float("inf")
        self.cql_clip_diff_max = float("-inf")

        if self.use_automatic_entropy_tuning:
            self.log_alpha = Scalar(0.0)
            self.alpha_optimizer = torch.optim.Adam(
                self.log_alpha.parameters(),
                lr=self.policy_lr,
            )
        else:
            self.log_alpha = None

        self.log_alpha_prime = Scalar(1.0)
        self.alpha_prime_optimizer = torch.optim.Adam(
            self.log_alpha_prime.parameters(),
            lr=self.qf_lr,
        )

    def safe_exp(self, x):
        return torch.where(x < 50, torch.exp(x), x)

    def update(self, current_it, total_it,  data_for_logging=None):
        sample = self.replay_buffer.sample(self.batch_size)
        obs, next_obs, r_obs, next_r_obs, act, rwd, done = list(sample.values())
        act_target, log_prob, _ = self.model.actor.sample(
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
            next_act_target, _, _ = self.model.actor.sample(
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

        cql_current_actions, cql_current_log_pis, _ = self.model.actor.sample(
            (
                obs.to(self.device),
                r_obs.reshape(self.batch_size, 1, -1).to(self.device),
            ), 
            repeat=self.cql_n_actions
        )
        cql_next_actions, cql_next_log_pis, _ = self.model.actor.sample(
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

        if self.use_automatic_entropy_tuning:
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()
            lalpha = loss_critic.data.item()

        self.actor_optimizer.zero_grad()
        loss_act.backward()
        self.actor_optimizer.step()
        la = loss_critic.data.item()

        self.critic_optimizer.zero_grad()
        loss_critic.backward()
        self.critic_optimizer.step()
        lc = loss_critic.data.item()

        if data_for_logging is not None:
            data_for_logging[0].log(
                {
                    "loss/actor": la,
                    "loss/alpha": lalpha,
                    "loss/critic": lc,
                },
                step=data_for_logging[1],
            )

    def update_target(self):
        for param, target_param in zip(
            self.model.parameters(), self.target.parameters()
        ):
            target_param.data.mul_(self.polyak)
            target_param.data.add_((1 - self.polyak) * param.data)
