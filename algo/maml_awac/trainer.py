import copy
import higher
import torch
import torch.nn.functional as F
import random
from tqdm import tqdm


class MAMLAWAC:
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
        self.alg_name = "MAMLAWAC"
        self.model = model
        self.target = copy.deepcopy(model)
        self.replay_buffer = replay_buffer
        self.actor_optimizer = actor_optimizer
        self.critic_optimizer = critic_optimizer
        self.batch_size = batch_size
        self.polyak = polyak
        self.gamma = torch.as_tensor([gamma])
        self.beta = torch.as_tensor([beta])

        self.device = model.device

    def safe_exp(self, x):
        return torch.where(x < 50, torch.exp(x), x)

    def soft_update(self, source, target):
        for param_source, param_target in zip(source.named_parameters(), target.named_parameters()):
            assert param_source[0] == param_target[0]
            param_target[1].data = self.target_vf_alpha * param_target[1].data + (1 - self.target_vf_alpha) * param_source[1].data


    def advantage_loss(self, policy, q_function, obs, r_obs, act):
        Q1, Q2 = q_function(
            (obs.to(self.device), r_obs.to(self.device)),
            act=act.to(self.device),
        )

        action_gen, _, _ ,_= policy.sample(
            (
                obs.to(self.device),
                r_obs.reshape(-1, 1, self.r_obs_dim).to(self.device),
            )
        )
        qw_ref = torch.min(torch.cat((Q1, Q2), 1), dim=1)[0].reshape(
            (-1, 1)
        )

        v_act1, v_act2 = q_function(
            (obs.to(self.device), r_obs.to(self.device)),
            act=action_gen.detach(),
        )

        qw_gen = torch.min(torch.cat((v_act1, v_act2), 1), dim=1)[
            0
        ].reshape((-1, 1))

        adv = torch.max(torch.zeros_like(qw_ref), qw_ref - qw_gen).detach()
        weights = self.thexp(adv / self.beta)
        
        loss_act = -(
                policy.get_log_prob(
                    (
                        obs.to(self.device),
                        r_obs.reshape(-1, 1, self.r_obs_dim).to(
                            self.device
                        ),
                    ),
                    act.squeeze().to(self.device),
                )
                * weights
            ).mean()
        return loss_act
    def value_loss(self,q_function, target, prev_obs, obs, prev_r_obs, r_obs, act, rwd, done):
        gamma_bar = pow(self.gamma, self.time_step * self.v_pref)
        next_act_target, next_log_prob, _ ,_= self.policy.sample(
            (
                obs.to(self.device),
                r_obs.reshape(-1, 1, self.r_obs_dim).to(self.device),
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
            gamma_bar * Q_target_min
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
        qf_target = copy.deepcopy(qf)
        opt = torch.optim.SGD([{'params': p, 'lr': None} for p in qf.parameters()])
        with higher.innerloop_ctx(qf, opt, override={'lr': [F.softplus(l)*0.001 for l in self.q_lrs]}, copy_initial_weights=False) as (f_q_function, diff_q_opt):  
            for step in range(self.itr_inner_loop):
                inner_value_loss = self.value_loss(f_q_function, qf_target, obs.to(self.device), next_obs.to(self.device), r_obs.to(self.device), next_r_obs.to(self.device), act.to(self.device), rwd.to(self.device), done.to(self.device))
                diff_q_opt.step(inner_value_loss)
                self.soft_update(f_q_function, qf_target)

            meta_value_loss = self.value_loss(f_q_function, qf_target, obs_val.to(self.device), next_obs_val.to(self.device), r_obs_val.to(self.device), next_r_obs_val.to(self.device), act_val.to(self.device), rwd_val.to(self.device), done_val.to(self.device))
        
        adapted_value_function = f_q_function
        opt = torch.optim.SGD([{'params': p, 'lr': None} for p in self.model.actor.parameters()])
        with higher.innerloop_ctx(self.model.actor, opt, override={'lr': [F.softplus(l)*0.001 for l in self.policy_lrs]}, copy_initial_weights=False) as (f_policy, diff_policy_opt):
            for step in range(self.itr_inner_loop):
                inner_policy_loss = self.advantage_loss(f_policy, adapted_value_function, obs.to(self.device), r_obs.to(self.device), act.to(self.device))
                diff_policy_opt.step(inner_policy_loss)
            meta_policy_loss = self.advantage_loss(f_policy, adapted_value_function, obs_val.to(self.device), r_obs_val.to(self.device), act_val.to(self.device))
        return  meta_value_loss, meta_policy_loss
    
    def update(self, train_tasks, data_for_logging=None):
        meta_policy_losses = []
        meta_value_losses = []

        sampled_tasks = random.sample(train_tasks, 5) 

        for task in sampled_tasks:
            # for batch in task:
            sample = task.sample(self.batch_size)
            sample_val = task.sample(self.batch_size)
            meta_value_loss, meta_policy_loss=self.step(sample, sample_val)
            meta_policy_losses.append(meta_policy_loss)
            meta_value_losses.append(meta_value_loss)
            # break
        meta_value_losses_mean = torch.mean(torch.stack(meta_value_losses))
        meta_value_losses_mean.backward()

        meta_policy_losses_mean = torch.mean(torch.stack(meta_policy_losses))
        meta_policy_losses_mean.backward()
        if data_for_logging is not None:
            data_for_logging[0].log(
                {
                    "meta_value_losses_mean": meta_policy_losses_mean,
                    "meta_policy_losses_mean": meta_policy_losses_mean,
                },
                step=data_for_logging[1],
            )

    def update_target(self):
        for param, target_param in zip(
            self.model.parameters(), self.target.parameters()
        ):
            target_param.data.mul_(self.polyak)
            target_param.data.add_((1 - self.polyak) * param.data)
