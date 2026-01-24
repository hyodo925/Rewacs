import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from .virtual_updater import VirtualActorUpdater
import higher

def get_grad_norms(model, prefix):
    metrics = {}
    total_norm = 0
    for name, param in model.named_parameters():
        if param.grad is not None:
            param_norm = param.grad.data.norm(2).item()
            metrics[f"grads/{prefix}_{name}"] = param_norm
            total_norm += param_norm ** 2
    metrics[f"grads/{prefix}_total_norm"] = total_norm ** 0.5
    return metrics

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

        self.lr =lr
        self.device = model.device
        self.gamma = torch.as_tensor([gamma]).to(self.device)
        self.beta = torch.as_tensor([beta]).to(self.device)

        self.updater = VirtualActorUpdater()

    def safe_exp(self, x):
        return torch.where(x < 50, torch.exp(x), x)

    def update(self, data_for_logging=None):
        grad_metrics = {}
        sample = self.replay_buffer.sample(self.batch_size)
        obs, next_obs, r_obs, next_r_obs, act, rwd, done = list(sample.values())

        sample_val = self.replay_buffer_val.sample(self.batch_size)
        # sample_val = self.replay_buffer.sample(self.batch_size)
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
            grad_metrics.update(get_grad_norms(self.model.critic, "critic"))
            data_for_logging[0].log(
                {
                    "loss/critic": lc,
                },
                step=data_for_logging[1],
            )
        ###############################
        ### compute actor loss      ###
        ###############################
        action_gen, log_prob, _, other_output = self.model.actor.sample((obs.to(self.device),r_obs.reshape(self.batch_size, 1, -1).to(self.device)))
        with torch.no_grad():
            qw_ref = torch.min(torch.cat((Q1, Q2), 1), dim=1)[0].reshape((-1, 1))
            v_act1, v_act2 = self.model.critic((obs.to(self.device), r_obs.to(self.device)),act=action_gen.detach(),)
            qw_gen = torch.min(torch.cat((v_act1, v_act2), 1), dim=1)[0].reshape((-1, 1))
            adv = torch.max(torch.zeros_like(qw_ref), qw_ref - qw_gen)
            weights = self.safe_exp(adv / self.beta)

        loss_act = -(self.model.actor.get_log_prob((obs.to(self.device),r_obs.reshape(self.batch_size, 1, -1).to(self.device),),act.squeeze().to(self.device),)* weights).mean()

        ###############################
        ### compute loss aux        ###
        ############################### 
        loss_auxiliary = self.model.meta_critic(
            (
                obs.reshape(self.batch_size, -1).to(self.device),
                r_obs.reshape(self.batch_size, -1).to(self.device),
            ),
            act.squeeze().to(self.device),
            other_output.reshape(self.batch_size, -1).to(self.device),
        )

        ################################################
        ### first pseudo update with loss act        ###
        ################################################

        grads_critic = torch.autograd.grad(loss_act, self.model.actor.parameters(), create_graph=True, allow_unused=True)
        self.updater.step(self.model.actor, grads_critic, "phi_old", self.lr)
        old_param = self.updater.get("phi_old")
        qw_ref = torch.min(torch.cat((Q1, Q2), 1), dim=1)[0].reshape((-1, 1))
        pi_val, log_pi_val, *_ = self.model.actor.sample_with_params((obs_val.to(self.device),r_obs_val.reshape(self.batch_size, 1, -1).to(self.device),),params=old_param,)
        with torch.no_grad():
            v_act1, v_act2 = self.model.critic((obs_val.to(self.device), r_obs_val.to(self.device)),act=pi_val)
            qw_gen = torch.min(torch.cat((v_act1, v_act2), 1), dim=1)[0].reshape((-1, 1))
            adv = torch.max(torch.zeros_like(qw_ref), qw_ref - qw_gen)
            weights = self.safe_exp(adv / self.beta)

        policy_loss_val = -(self.model.actor.get_log_prob_with_params((obs_val.to(self.device),r_obs_val.reshape(self.batch_size, 1, -1).to(self.device),),pi_val.squeeze().to(self.device),params=old_param)* weights).mean()

        # abs_theta = 0.0
        # for p in self.param_optim_theta:
        #     if p.grad is not None:
        #         penalty = l1_penalty(p.grad)
        #         if torch.isnan(penalty).any():
        #             print("Warning: NaN in L1 penalty")
        #         abs_theta += penalty.item()

        ################################################
        ### second pseudo update with loss aux       ###
        ################################################
        # print(f"DEBUG: loss_auxiliary grad_fn: {loss_auxiliary.grad_fn}") # ここがNoneならMeta-Critic自体が不正
        grads_mcritic = torch.autograd.grad(loss_auxiliary, self.model.actor.parameters(), create_graph=True, allow_unused=True)
        # print(f"DEBUG: grads_mcritic[0] grad_fn: {grads_mcritic[0].grad_fn}") # ここがNoneならcreate_graphが効いていない
        self.updater.step(self.model.actor, grads_mcritic, "phi_new", self.lr, from_params=old_param)
        new_param = self.updater.get("phi_new")
        first_key = list(new_param.keys())[0]
        first_p = new_param[first_key]

        # print(f"DEBUG: first_key: {first_key}")
        # print(f"DEBUG: new_param[{first_key}] grad_fn: {first_p.grad_fn}")

        # # もし grad_fn が None なら、ここで鎖が切れています
        # if first_p.grad_fn is None:
        #     print("!!! 警告: updater の中で計算グラフが切断されています !!!")
        
        pi_val_new, log_pi_val_new, *_ = self.model.actor.sample_with_params((obs_val.to(self.device),r_obs_val.reshape(self.batch_size, 1, -1).to(self.device),),params=new_param)
        # print(f"DEBUG: log_pi_val_new grad_fn: {log_pi_val_new.grad_fn}")
        with torch.no_grad(): 
            v_act1, v_act2 = self.model.critic((obs_val.to(self.device), r_obs_val.to(self.device)),act=pi_val_new)

            qw_gen = torch.min(torch.cat((v_act1, v_act2), 1), dim=1)[0].reshape((-1, 1))
            adv = torch.max(torch.zeros_like(qw_ref), qw_ref - qw_gen)
            weights = self.safe_exp(adv / self.beta)

        policy_loss_val_new = -(self.model.actor.get_log_prob_with_params((obs_val.to(self.device),r_obs_val.reshape(self.batch_size, 1, -1).to(self.device)),pi_val_new.squeeze().to(self.device),params=new_param)* weights).mean()
        # print(f"DEBUG: policy_loss_val_new grad_fn: {policy_loss_val_new.grad_fn}")
        ###############################
        ### compute loss meta       ###
        ###############################
        utility = policy_loss_val - policy_loss_val_new
        utility = torch.tanh(utility)
        # utility = utility / (1 + torch.abs(utility)) 
        loss_meta = -utility
        self.meta_critic_optimizer.zero_grad()
        loss_meta.backward(retain_graph=True)
        if data_for_logging is not None:
            grad_metrics.update(get_grad_norms(self.model.meta_critic, "meta_critic"))
        # for name, param in self.model.meta_critic.named_parameters():
        #     if param.grad is not None:
        #         # 勾配の平均絶対値やノルムを表示
        #         print(f"{name} | grad mean: {param.grad.abs().mean().item():.6f} | max: {param.grad.max().item():.6f}")
        #     else:
        #         print(f"{name} | grad is None (勾配が届いていません！)")
        
   
        # action_gen, log_prob, _, other_output = self.model.actor.sample((obs.to(self.device),r_obs.reshapeself.batch_size, 1, -1).to(self.device)))
        # with torch.no_grad():
        #     qw_ref = torch.min(torch.cat((Q1, Q2), 1), dim=1)[0].reshape((-1, 1))
        #     v_act1, v_act2 = self.model.critic((obs.to(self.device), r_obs.to(self.device)),act=action_gen.detach(),)
        #     qw_gen = torch.min(torch.cat((v_act1, v_act2), 1), dim=1)[0].reshape((-1, 1))
        #     adv = torch.max(torch.zeros_like(qw_ref), qw_ref - qw_gen)
        #     weights = self.safe_exp(adv / self.beta)

        # loss_act = -(self.model.actor.get_log_prob((obs.to(self.device),r_obs.reshape(self.batch_size, 1, -1).to(self.device),),act.squeeze().to(self.device),)* weights).mean()

        # loss_auxiliary = self.model.meta_critic((obs.reshape(self.batch_size, -1).to(self.device),r_obs.reshape(self.batch_size, -1).to(self.device),),act.squeeze().to(self.device),other_output.reshape(self.batch_size, -1).to(self.device),)(
        # loss_act = loss_act + loss_auxiliary
        self.actor_optimizer.zero_grad()
        loss_auxiliary.backward(retain_graph=True)
        loss_act.backward()
        if data_for_logging is not None:
            grad_metrics.update(get_grad_norms(self.model.actor, "actor"))
        # for name, param in self.model.actor.named_parameters():
        #     if param.grad is not None:
        #         # 勾配の平均絶対値やノルムを表示
        #         print(f"{name} | grad mean: {param.grad.abs().mean().item():.6f} | max: {param.grad.max().item():.6f}")
        #     else:
        #         print(f"{name} | grad is None (勾配が届いていません！)")
        self.meta_critic_optimizer.step()
        lm = loss_meta.data.item()
        self.actor_optimizer.step()
        la = loss_act.data.item()

        if data_for_logging is not None:
            log_data = {
                "loss/critic": lc,
                "loss/actor": la,
                "loss/meta": lm,
                "loss/auxiliary": loss_auxiliary.data.item(),
            }
            log_data.update(grad_metrics) # 勾配情報を追加
            data_for_logging[0].log(log_data, step=data_for_logging[1])
    

    def update_target(self, ):
        for param, target_param in zip(
            self.model.parameters(), self.target.parameters()
        ):
            target_param.data.mul_(self.polyak)
            target_param.data.add_((1 - self.polyak) * param.data)

    def finetune(self, data_for_logging=None):
        grad_metrics = {}
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

        action_gen, log_prob, _, other_output = self.model.actor.sample((obs.to(self.device),r_obs.reshape(self.batch_size, 1, -1).to(self.device)))
        with torch.no_grad():
            qw_ref = torch.min(torch.cat((Q1, Q2), 1), dim=1)[0].reshape((-1, 1))
            v_act1, v_act2 = self.model.critic((obs.to(self.device), r_obs.to(self.device)),act=action_gen.detach(),)
            qw_gen = torch.min(torch.cat((v_act1, v_act2), 1), dim=1)[0].reshape((-1, 1))
            adv = torch.max(torch.zeros_like(qw_ref), qw_ref - qw_gen)
            weights = self.safe_exp(adv / self.beta)

        loss_act = -(self.model.actor.get_log_prob((obs.to(self.device),r_obs.reshape(self.batch_size, 1, -1).to(self.device),),act.squeeze().to(self.device),)* weights).mean()

        loss_auxiliary = self.model.meta_critic(
            (
                obs.reshape(self.batch_size, -1).to(self.device),
                r_obs.reshape(self.batch_size, -1).to(self.device),
            ),
            act.squeeze().to(self.device),
            other_output.reshape(self.batch_size, -1).to(self.device),
        )

        grads_critic = torch.autograd.grad(loss_act, self.model.actor.parameters(), create_graph=True, allow_unused=True)
        self.updater.step(self.model.actor, grads_critic, "phi_old", self.lr)
        old_param = self.updater.get("phi_old")
        qw_ref = torch.min(torch.cat((Q1, Q2), 1), dim=1)[0].reshape((-1, 1))
        pi_val, log_pi_val, *_ = self.model.actor.sample_with_params((obs_val.to(self.device),r_obs_val.reshape(self.batch_size, 1, -1).to(self.device),),params=old_param,)
        with torch.no_grad():
            v_act1, v_act2 = self.model.critic((obs_val.to(self.device), r_obs_val.to(self.device)),act=pi_val)
            qw_gen = torch.min(torch.cat((v_act1, v_act2), 1), dim=1)[0].reshape((-1, 1))
            adv = torch.max(torch.zeros_like(qw_ref), qw_ref - qw_gen)
            weights = self.safe_exp(adv / self.beta)

        policy_loss_val = -(self.model.actor.get_log_prob_with_params((obs_val.to(self.device),r_obs_val.reshape(self.batch_size, 1, -1).to(self.device),),pi_val.squeeze().to(self.device),params=old_param)* weights).mean()


        grads_mcritic = torch.autograd.grad(loss_auxiliary, self.model.actor.parameters(), create_graph=True, allow_unused=True)
        self.updater.step(self.model.actor, grads_mcritic, "phi_new", self.lr, from_params=old_param)
        new_param = self.updater.get("phi_new")
        first_key = list(new_param.keys())[0]
        first_p = new_param[first_key]
        
        pi_val_new, log_pi_val_new, *_ = self.model.actor.sample_with_params((obs_val.to(self.device),r_obs_val.reshape(self.batch_size, 1, -1).to(self.device),),params=new_param)
        with torch.no_grad(): 
            v_act1, v_act2 = self.model.critic((obs_val.to(self.device), r_obs_val.to(self.device)),act=pi_val_new)

            qw_gen = torch.min(torch.cat((v_act1, v_act2), 1), dim=1)[0].reshape((-1, 1))
            adv = torch.max(torch.zeros_like(qw_ref), qw_ref - qw_gen)
            weights = self.safe_exp(adv / self.beta)

        policy_loss_val_new = -(self.model.actor.get_log_prob_with_params((obs_val.to(self.device),r_obs_val.reshape(self.batch_size, 1, -1).to(self.device)),pi_val_new.squeeze().to(self.device),params=new_param)* weights).mean()

        utility = policy_loss_val - policy_loss_val_new
        # utility = (utility - utility.mean()) / (utility.std() + 1e-8)
        utility = torch.tanh(utility)
        
        # utility = utility / (1 + torch.abs(utility)) 
        loss_meta = -utility
        self.meta_critic_optimizer.zero_grad()
        loss_meta.backward(retain_graph=True)

        self.actor_optimizer.zero_grad()
        loss_auxiliary.backward(retain_graph=True)
        loss_act.backward()

        self.meta_critic_optimizer.step()
        lm = loss_meta.data.item()
        self.actor_optimizer.step()
        la = loss_act.data.item()


        if data_for_logging is not None:
            grad_metrics.update(get_grad_norms(self.model.critic, "critic"))
            grad_metrics.update(get_grad_norms(self.model.actor, "actor"))
            grad_metrics.update(get_grad_norms(self.model.meta_critic, "meta_critic"))
            log_data = {
                "loss/critic": lc,
                "loss/actor": la,
                "loss/meta": lm,
                "loss/auxiliary": loss_auxiliary.data.item(),
            }
            log_data.update(grad_metrics) # 勾配情報を追加
            data_for_logging[0].log(log_data, step=data_for_logging[1])


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

        self.lr =lr
        self.device = model.device
        self.gamma = torch.as_tensor([gamma]).to(self.device)
        self.beta = torch.as_tensor([beta]).to(self.device)

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
        grad_metrics = {}
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

        loss_auxiliary = self.model.meta_critic(
            (
                obs.reshape(self.batch_size, -1).to(self.device),
                r_obs.reshape(self.batch_size, -1).to(self.device),
            ),
            act.squeeze().to(self.device),
            other_output.reshape(self.batch_size, -1).to(self.device),
        )

        grads_critic = torch.autograd.grad(loss_act, self.model.actor.parameters(), create_graph=True, allow_unused=True)
        self.updater.step(self.model.actor, grads_critic, "phi_old", self.lr)
        old_param = self.updater.get("phi_old")
        pi_val, log_pi_val_old, *_ = self.model.actor.sample_with_params((obs_val.to(self.device),r_obs_val.reshape(self.batch_size, 1, -1).to(self.device),),params=old_param,)
        get_log_prob_old = self.model.actor.get_log_prob_with_params((obs_val.to(self.device),r_obs_val.reshape(self.batch_size, 1, -1).to(self.device),),act_val.squeeze().to(self.device),params=old_param,)
        policy_loss_val = (alpha * log_pi_val_old - get_log_prob_old).mean()


        grads_mcritic = torch.autograd.grad(loss_auxiliary, self.model.actor.parameters(), create_graph=True, allow_unused=True)
        self.updater.step(self.model.actor, grads_mcritic, "phi_new", self.lr, from_params=old_param)
        new_param = self.updater.get("phi_new")
        pi_val_new, log_pi_val_new, *_ = self.model.actor.sample_with_params((obs_val.to(self.device),r_obs_val.reshape(self.batch_size, 1, -1).to(self.device),),params=new_param)
        get_log_prob_new = self.model.actor.get_log_prob_with_params((obs_val.to(self.device),r_obs_val.reshape(self.batch_size, 1, -1).to(self.device),),act_val.squeeze().to(self.device),params=new_param,)
        policy_loss_val_new = (alpha * log_pi_val_new - get_log_prob_new).mean()
        
        
        utility = policy_loss_val - policy_loss_val_new
        # utility = (utility - utility.mean()) / (utility.std() + 1e-8)
        utility = torch.tanh(utility)
        # utility = utility / (1 + torch.abs(utility)) 
        loss_meta = -utility
        self.meta_critic_optimizer.zero_grad()
        loss_meta.backward(retain_graph=True)
        self.actor_optimizer.zero_grad()
        loss_auxiliary.backward(retain_graph=True)
        loss_act.backward()

        self.meta_critic_optimizer.step()
        lm = loss_meta.data.item()
        self.actor_optimizer.step()
        la = loss_act.data.item()


        if data_for_logging is not None:
            grad_metrics.update(get_grad_norms(self.model.critic, "critic"))
            grad_metrics.update(get_grad_norms(self.model.actor, "actor"))
            grad_metrics.update(get_grad_norms(self.model.meta_critic, "meta_critic"))
            log_data = {
                "loss/critic": lc,
                "loss/actor": la,
                "loss/meta": lm,
                "loss/auxiliary": loss_auxiliary.data.item(),
            }
            log_data.update(grad_metrics) # 勾配情報を追加
            data_for_logging[0].log(log_data, step=data_for_logging[1])


    def finetune(self, current_it, total_it, data_for_logging=None):
        grad_metrics = {}
        sample = self.replay_buffer.sample(self.batch_size)
        obs, next_obs, r_obs, next_r_obs, act, rwd, mc_returns, done = list(sample.values())

        sample_val = self.replay_buffer.sample(self.batch_size)
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
        ).uniform_(-1, 1).to(self.device)

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
        ).to(self.device)

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

        loss_auxiliary = self.model.meta_critic(
            (
                obs.reshape(self.batch_size, -1).to(self.device),
                r_obs.reshape(self.batch_size, -1).to(self.device),
            ),
            act.squeeze().to(self.device),
            other_output.reshape(self.batch_size, -1).to(self.device),
        )

        grads_critic = torch.autograd.grad(loss_act, self.model.actor.parameters(), create_graph=True, allow_unused=True)
        self.updater.step(self.model.actor, grads_critic, "phi_old", self.lr)
        old_param = self.updater.get("phi_old")
        pi_val, log_pi_val_old, *_ = self.model.actor.sample_with_params((obs_val.to(self.device),r_obs_val.reshape(self.batch_size, 1, -1).to(self.device),),params=old_param,)
        get_log_prob_old = self.model.actor.get_log_prob_with_params((obs_val.to(self.device),r_obs_val.reshape(self.batch_size, 1, -1).to(self.device),),act_val.squeeze().to(self.device),params=old_param,)
        policy_loss_val = (alpha * log_pi_val_old - get_log_prob_old).mean()


        grads_mcritic = torch.autograd.grad(loss_auxiliary, self.model.actor.parameters(), create_graph=True, allow_unused=True)
        self.updater.step(self.model.actor, grads_mcritic, "phi_new", self.lr, from_params=old_param)
        new_param = self.updater.get("phi_new")
        pi_val_new, log_pi_val_new, *_ = self.model.actor.sample_with_params((obs_val.to(self.device),r_obs_val.reshape(self.batch_size, 1, -1).to(self.device),),params=new_param)
        get_log_prob_new = self.model.actor.get_log_prob_with_params((obs_val.to(self.device),r_obs_val.reshape(self.batch_size, 1, -1).to(self.device),),act_val.squeeze().to(self.device),params=new_param,)
        policy_loss_val_new = (alpha * log_pi_val_new - get_log_prob_new).mean()
        
        
        utility = policy_loss_val - policy_loss_val_new
        # utility = (utility - utility.mean()) / (utility.std() + 1e-8)
        utility = torch.tanh(utility)
        # utility = utility / (1 + torch.abs(utility)) 
        loss_meta = -utility
        self.meta_critic_optimizer.zero_grad()
        loss_meta.backward(retain_graph=True)
        self.actor_optimizer.zero_grad()
        loss_auxiliary.backward(retain_graph=True)
        loss_act.backward()

        self.meta_critic_optimizer.step()
        lm = loss_meta.data.item()
        self.actor_optimizer.step()
        la = loss_act.data.item()


        if data_for_logging is not None:
            grad_metrics.update(get_grad_norms(self.model.critic, "critic"))
            grad_metrics.update(get_grad_norms(self.model.actor, "actor"))
            grad_metrics.update(get_grad_norms(self.model.meta_critic, "meta_critic"))
            log_data = {
                "loss/critic": lc,
                "loss/actor": la,
                "loss/meta": lm,
                "loss/auxiliary": loss_auxiliary.data.item(),
            }
            log_data.update(grad_metrics) # 勾配情報を追加
            data_for_logging[0].log(log_data, step=data_for_logging[1])


    def update_target(self):
        for param, target_param in zip(
            self.model.parameters(), self.target.parameters()
        ):
            target_param.data.mul_(self.polyak)
            target_param.data.add_((1 - self.polyak) * param.data)

