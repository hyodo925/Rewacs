import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
from tqdm import tqdm
import higher
from torchviz import make_dot
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

def check_gradients(model, name):
    has_grad = any(p.grad is not None for p in model.parameters())
    print(f"[{name}] Gradient exists: {has_grad}")
    if has_grad:
        # 勾配の絶対値の平均を見て、ゼロでないか確認
        avg_grad = torch.cat([p.grad.view(-1) for p in model.parameters() if p.grad is not None]).abs().mean().item()
        print(f"[{name}] Average Gradient Magnitude: {avg_grad:.8f}")

def trace_leaf_parameters(loss, model_dict):
    seen = set()
    reachable_params = set()

    def traverse(grad_fn):
        if grad_fn is None or grad_fn in seen:
            return
        seen.add(grad_fn)
        
        # grad_fnがテンソル（AccumulateGrad）を保持している場合
        if hasattr(grad_fn, 'variable'):
            var = grad_fn.variable
            reachable_params.add(id(var))
        
        # 次のノードへ再帰
        if hasattr(grad_fn, 'next_functions'):
            for next_fn, _ in grad_fn.next_functions:
                traverse(next_fn)

    # 追跡開始
    traverse(loss.grad_fn)

    # どのモデルのパラメータが含まれているか判定
    results = {}
    for model_name, model in model_dict.items():
        found_params = []
        for p_name, p in model.named_parameters():
            if id(p) in reachable_params:
                found_params.append(p_name)
        results[model_name] = found_params
    
    return results

#psudo update module fro meta critic network
class Hot_Plug(object):
    def __init__(self, model):
        self.model = model
        self.params = OrderedDict(self.model.named_parameters())
    def update(self, lr=0.01):
        for param_name in self.params.keys():
            path = param_name.split('.')# example layers.0.weight
            cursor = self.model
            for module_name in path[:-1]:#path is the list of keys in dict
                cursor = cursor._modules[module_name]
            if lr > 0:
                #psudo update
                cursor._parameters[path[-1]] = self.params[param_name] - lr*self.params[param_name].grad
            else:
                cursor._parameters[path[-1]] = self.params[param_name]
    def restore(self):
        self.update(lr=0)

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
        flow=None,
        lr=3e-4,
        polyak=0.995,
        gamma=0.9,
        beta=0.3,
    ):
        self.alg_name = "MetaCriticAWAC" 
        self.model = model
        self.flow = flow
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


        feature_net = nn.Sequential(*list(self.model.actor.children())[:-2])
        self.hotplug = Hot_Plug(feature_net)

    def safe_exp(self, x):
        return torch.where(x < 50, torch.exp(x), x)
    
    def update(self, update_actor, data_for_logging=None):
        #batch sample (train data)
        sample = self.replay_buffer.sample(self.batch_size)
        obs, next_obs, r_obs, next_r_obs, act, rwd, done = list(sample.values())
        #batch sample (validation data for meta test)
        sample_val = self.replay_buffer.sample(self.batch_size)
        obs_val, next_obs_val, r_obs_val, next_r_obs_val, act_val, rwd_val, done_val = list(sample_val.values())
        
        with torch.no_grad():
            old_mean, old_log_std, _ = self.model.actor((obs.to(self.device), r_obs.reshape(self.batch_size, 1, -1).to(self.device)))
            c_feats = self.model.critic.integrator(obs.to(self.device), r_obs.reshape(self.batch_size, 1, -1).to(self.device))
            if not self.model.critic.single:
                old_c_feats = torch.cat([c_feats, act.squeeze().to(self.device)], -1)
        #Calculate Critic loss
        with torch.no_grad():#not to flow gradients to actor and target network
            next_act_target, next_log_prob, *_ = self.model.actor.sample((next_obs.to(self.device),next_r_obs.reshape(self.batch_size, 1, -1).to(self.device),))
            Q_target_1, Q_target_2 = self.target.critic((next_obs.to(self.device), next_r_obs.to(self.device)), next_act_target)
            Q_target_min = torch.min(torch.cat((Q_target_1, Q_target_2), 1), dim=1)[0].unsqueeze(-1)#min double Q approach
            Q_target = rwd.to(self.device) + (self.gamma * Q_target_min) * done.to(self.device)#target
        Q1, Q2 = self.model.critic((obs.to(self.device), r_obs.to(self.device)),act.squeeze().to(self.device),)#Calculate current Q
        loss_critic = F.mse_loss(Q_target, Q1) + F.mse_loss(Q_target, Q2)#TD loss function

        #Calculate actor loss
        action_gen, log_prob, _, other_output = self.model.actor.sample((obs.to(self.device),r_obs.reshape(self.batch_size, 1, -1).to(self.device)))
        with torch.no_grad():#not to update critic network (advantage function is originated from critic network)
            qw_ref = torch.min(torch.cat((Q1, Q2), 1), dim=1)[0].reshape((-1, 1))
            v_act1, v_act2 = self.model.critic((obs.to(self.device), r_obs.to(self.device)),act=action_gen.detach(),)
            qw_gen = torch.min(torch.cat((v_act1, v_act2), 1), dim=1)[0].reshape((-1, 1))
            adv = torch.max(torch.zeros_like(qw_ref), qw_ref - qw_gen)
            weights = self.safe_exp(adv / self.beta)# not to explode by exp based loss function
        get_log_prob = self.model.actor.get_log_prob((obs.to(self.device),r_obs.reshape(self.batch_size, 1, -1).to(self.device),),act.squeeze().to(self.device),)#AWAC calculates loss function based on batch sample data not action_gen
        loss_act = -(get_log_prob* weights).mean()

        self.critic_optimizer.zero_grad()#clear grad on critic network
        loss_critic.backward()#flow gradients to model.critic.weight.grad
        grad_diversity_critic = get_grad_direction_stats(self.model.critic)
        critic_grad_norm = get_grad_norm(self.model.critic)
        critic_weight_norm = get_weight_norm(self.model.critic)
        self.critic_optimizer.step()#update critic network by calculating θ' = θ - γ∇loss_critic
        lc = loss_critic.data.item()

        #Calculate loss from meta-critic network
        loss_auxiliary = self.model.meta_critic(
            (obs.reshape(self.batch_size, -1).to(self.device),r_obs.reshape(self.batch_size, -1).to(self.device),),
            act.squeeze().to(self.device),
            other_output.reshape(self.batch_size, -1).to(self.device),#to deriviate by actor parameter
        )

        self.actor_optimizer.zero_grad()
        #It is also for meta optimization
        loss_act.backward(retain_graph=True)#retain graph is for calculating loss.auxiliary.backward
        # check_gradients(self.model.actor, "Actor after loss_act.backward")
        grad_diversity_actor = get_grad_direction_stats(self.model.actor)
        actor_grad_norm = get_grad_norm(self.model.actor)
        actor_weight_norm = get_weight_norm(self.model.actor)
        #first psudo update for actor network
        self.hotplug.update(self.lr)

        #Calculate loss_actor again from validation data
        pi_val, log_pi_val, *_ = self.model.actor.sample((obs_val.to(self.device),r_obs_val.reshape(self.batch_size, 1, -1).to(self.device),))
        with torch.no_grad():#not to update critic network (advantage function is originated from critic network)
            v_act1, v_act2 = self.model.critic((obs_val.to(self.device), r_obs_val.to(self.device)),act=pi_val.detach(),)
            qw_gen = torch.min(torch.cat((v_act1, v_act2), 1), dim=1)[0].reshape((-1, 1))
            adv = torch.max(torch.zeros_like(qw_ref), qw_ref - qw_gen)
            weights = self.safe_exp(adv / self.beta)
        get_log_prob_old = self.model.actor.get_log_prob((obs_val.to(self.device),r_obs_val.reshape(self.batch_size, 1, -1).to(self.device),),act_val.squeeze().to(self.device))
        policy_loss_val = -(get_log_prob_old* weights).mean()

        #It is also for meta optimization
        loss_auxiliary.backward(create_graph=True)#create_graph is for computational graph from policy_loss_val_new to meta-critic model parameter 
        laux = loss_auxiliary.data.item()
        # check_gradients(self.model.actor, "Actor after loss_auxiliary.backward")
        #first psudo update for actor network
        self.hotplug.update(self.lr)

        pi_val_new, log_pi_val_new, *_ = self.model.actor.sample((obs_val.to(self.device),r_obs_val.reshape(self.batch_size, 1, -1).to(self.device),))
        with torch.no_grad(): 
            v_act1, v_act2 = self.model.critic((obs_val.to(self.device), r_obs_val.to(self.device)),act=pi_val_new.detach(),)
            qw_gen = torch.min(torch.cat((v_act1, v_act2), 1), dim=1)[0].reshape((-1, 1))
            adv = torch.max(torch.zeros_like(qw_ref), qw_ref - qw_gen)
            weights = self.safe_exp(adv / self.beta)
        get_log_prob_new = self.model.actor.get_log_prob((obs_val.to(self.device),r_obs_val.reshape(self.batch_size, 1, -1).to(self.device),),act_val.squeeze().to(self.device))
        policy_loss_val_new = -(get_log_prob_new* weights).mean()
        
        #Calculate loss function for updating meta critic
        utility = policy_loss_val - policy_loss_val_new
        utility = torch.tanh(utility)
        loss_meta = -utility


        self.meta_critic_optimizer.zero_grad()
        grad_omega = torch.autograd.grad(loss_meta, self.model.meta_critic.parameters())
        # print(f"[Meta-Critic] Grad_omega calculated: {grad_omega[0] is not None}")
        for gradient, variable in zip(grad_omega, self.model.meta_critic.parameters()):
            variable.grad = gradient
        #update meta critic as a meta optimization
        grad_diversity_meta_critic = get_grad_direction_stats(self.model.meta_critic)
        meta_critic_grad_norm = get_grad_norm(self.model.meta_critic)
        meta_critic_weight_norm = get_weight_norm(self.model.meta_critic)
        self.meta_critic_optimizer.step()
        lm = loss_meta.data.item()
        #update actor as a meta optimization
        if update_actor:
            self.actor_optimizer.step()
        la = loss_act.data.item()
        #reset psudo update
        self.hotplug.restore()  

        with torch.no_grad():
                new_mean, new_log_std, _ = self.model.actor((obs.to(self.device), r_obs.reshape(self.batch_size, 1, -1).to(self.device)))
                a_feats = self.model.actor.integrator(obs.to(self.device), r_obs.reshape(self.batch_size, 1, -1).to(self.device))
                c_feats = self.model.critic.integrator(obs.to(self.device), r_obs.reshape(self.batch_size, 1, -1).to(self.device))
                if not self.model.critic.single:
                    new_c_feats = torch.cat([c_feats, act.squeeze().to(self.device)], -1)
            
                kl_div = get_kl_divergence((old_mean, old_log_std), (new_mean, new_log_std))
                wass_dist = get_wasserstein_dist(old_c_feats, new_c_feats)
                mmd_val = get_mmd(old_c_feats, new_c_feats)
        #log
        with torch.no_grad(): 
            if data_for_logging is not None:
                data_for_logging[0].log(
                    {
                        "loss/actor": la,
                        "loss/critic": lc,
                        "loss/auxiliary": laux,
                        "loss/meta": lm,
                        "diag/Q": ((Q1 + Q2) /2).mean().item(),
                        "diag/log_prob": log_prob.mean().data.item(),
                        "diag/log_prob_old": get_log_prob_old.mean().data.item(),
                        "diag/log_prob_new": get_log_prob_old.mean().data.item(),
                        "diag/actor_grad_norm":actor_grad_norm,
                        "diag/actor_weight_norm":actor_weight_norm,
                        "diag/critic_grad_norm":critic_grad_norm,
                        "diag/critic_weight_norm":critic_weight_norm,
                        "diag/meta_critic_grad_norm":meta_critic_grad_norm,
                        "diag/meta_critic_weight_norm":meta_critic_weight_norm,
                        "diag/actor_kl_divergence": kl_div,
                        "diag/critic_wasserstein_dist": wass_dist,
                        "diag/critic_mmd": mmd_val,
                        "diag/grad_diversity_actor": grad_diversity_actor,
                        "diag/grad_diversity_critic": grad_diversity_critic,
                        "diag/grad_diversity_meta_critic": grad_diversity_meta_critic,
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

    def update_target(self, ):
        for param, target_param in zip(self.model.parameters(), self.target.parameters()):
            #calculate polyac average
            target_param.data.mul_(self.polyak)
            target_param.data.add_((1 - self.polyak) * param.data)


class MetaCriticSAC:
    def __init__(
        self,
        model,
        replay_buffer,
        replay_buffer_val,
        actor_optimizer,
        critic_optimizer,
        meta_critic_optimizer,
        batch_size,
        flow=None,
        lr=3e-4,
        polyak=0.995,
        gamma=0.9,
        beta=0.3,
        alpha=1.0,
        init_temperature=0.1,
    ):
        self.alg_name = "MetaCriticSAC" 
        self.model = model
        self.flow = flow
        self.act_dim = 2
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
        self.alpha = torch.as_tensor([alpha]).to(self.device)

        self.auto_entropy_tuning = True
        target_entropy=None

        if self.auto_entropy_tuning:
            if target_entropy is None:
                self.target_entropy = -torch.prod(torch.Tensor([self.act_dim]).to(self.device)).item()
            else:
                self.target_entropy = target_entropy
            self.log_alpha = torch.tensor(np.log(init_temperature)).to(model.device)
            self.log_alpha.requires_grad = True
            self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=self.lr, betas=(0.5, 0.999),)
            self.alpha = self.log_alpha.exp()
        else:
            self.alpha = torch.tensor(self.alpha).to(self.device)

        feature_net = nn.Sequential(*list(self.model.actor.children())[:-2])
        self.hotplug = Hot_Plug(feature_net)

    def safe_exp(self, x):
        return torch.where(x < 50, torch.exp(x), x)
    
    def update(self, data_for_logging=None):
        #batch sample (train data)
        sample = self.replay_buffer.sample(self.batch_size)
        obs, next_obs, r_obs, next_r_obs, act, rwd, done = list(sample.values())
        #batch sample (validation data for meta test)
        sample_val = self.replay_buffer.sample(self.batch_size)
        obs_val, next_obs_val, r_obs_val, next_r_obs_val, act_val, rwd_val, done_val = list(sample_val.values())

        #Calculate Critic loss
        with torch.no_grad():#not to flow gradients to actor and target network
            next_act_target, next_log_prob, *_ = self.model.actor.sample((next_obs.to(self.device),next_r_obs.reshape(self.batch_size, 1, -1).to(self.device),))
            Q_target_1, Q_target_2 = self.target.critic((next_obs.to(self.device), next_r_obs.to(self.device)), next_act_target)
            Q_target_min = torch.min(torch.cat((Q_target_1, Q_target_2), 1), dim=1)[0].unsqueeze(-1)#min double Q approach
            Q_target = rwd.to(self.device) + (self.gamma * Q_target_min) * done.to(self.device)#target
        Q1, Q2 = self.model.critic((obs.to(self.device), r_obs.to(self.device)),act.squeeze().to(self.device),)#Calculate current Q
        loss_critic = F.mse_loss(Q_target, Q1) + F.mse_loss(Q_target, Q2)#TD loss function

        self.critic_optimizer.zero_grad()#clear grad on critic network
        loss_critic.backward()#flow gradients to model.critic.weight.grad
        self.critic_optimizer.step()#update critic network by calculating θ' = θ - γ∇loss_critic
        lc = loss_critic.data.item()

        #Calculate actor loss
        action_gen, log_prob, _, other_output= self.model.actor.sample((obs.to(self.device),r_obs.reshape(self.batch_size, 1, -1).to(self.device),))
        Q1, Q2 = self.model.critic((obs.to(self.device), r_obs.to(self.device)), action_gen)
        Q_min = torch.min(torch.cat((Q1, Q2), 1), dim=1)[0].unsqueeze(-1)

        loss_act = ((self.alpha.detach() * log_prob) - Q_min).mean()

        #Calculate loss from meta-critic network
        # loss_auxiliary = self.model.meta_critic(
        #     (obs.reshape(self.batch_size, -1).to(self.device),r_obs.reshape(self.batch_size, -1).to(self.device),),
        #     act.squeeze().to(self.device),
        #     other_output.reshape(self.batch_size, -1).to(self.device),#to deriviate by actor parameter
        # )

        self.actor_optimizer.zero_grad()
        #It is also for meta optimization
        loss_act.backward(retain_graph=True)#retain graph is for calculating loss.auxiliary.backward
        #first psudo update for actor network
        # self.hotplug.update(self.lr)

        # #Calculate loss_actor again from validation data
        # action_gen_val, log_prob_val, _, _= self.model.actor.sample((obs_val.to(self.device),r_obs_val.reshape(self.batch_size, 1, -1).to(self.device),))
        # Q1_val, Q2_val = self.model.critic((obs_val.to(self.device), r_obs_val.to(self.device)), action_gen_val)
        # Q_min_val = torch.min(torch.cat((Q1_val, Q2_val), 1), dim=1)[0].unsqueeze(-1)
        # policy_loss_val = ((self.alpha.detach() * log_prob_val) - Q_min_val).mean()

        # #It is also for meta optimization
        # loss_auxiliary.backward(create_graph=True)#create_graph is for computational graph from policy_loss_val_new to meta-critic model parameter 
        # #first psudo update for actor network
        # self.hotplug.update(self.lr)
        
        # action_gen_val_new, log_prob_val_new, _, _= self.model.actor.sample((obs_val.to(self.device),r_obs_val.reshape(self.batch_size, 1, -1).to(self.device),))
        # Q1_val_new, Q2_val_new = self.model.critic((obs_val.to(self.device), r_obs_val.to(self.device)), action_gen_val_new)
        # Q_min_val_new = torch.min(torch.cat((Q1_val_new, Q2_val_new), 1), dim=1)[0].unsqueeze(-1)
        # policy_loss_val_new = ((self.alpha.detach() * log_prob_val_new) - Q_min_val_new).mean()
        # #Calculate loss function for updating meta critic
        # utility = policy_loss_val - policy_loss_val_new
        # utility = torch.tanh(utility)
        # loss_meta = -utility
        # self.meta_critic_optimizer.zero_grad()
        # grad_omega = torch.autograd.grad(loss_meta, self.model.meta_critic.parameters())
        # for gradient, variable in zip(grad_omega, self.model.meta_critic.parameters()):
        #     variable.grad = gradient
        # #update meta critic as a meta optimization
        # self.meta_critic_optimizer.step()
        # lm = loss_meta.data.item()
        #update actor as a meta optimization
        self.actor_optimizer.step()
        la = loss_act.data.item()
        #reset psudo update
        # self.hotplug.restore()  

        if self.auto_entropy_tuning:
            loss_alpha = -(self.log_alpha * (log_prob + self.target_entropy).detach()).mean()
            self.alpha_optimizer.zero_grad()
            loss_alpha.backward()
            self.alpha_optimizer.step()
            self.alpha = self.log_alpha.exp()
            lalpha = loss_alpha.data.item()

        #log
        with torch.no_grad(): 
            if data_for_logging is not None:
                log_data = {
                    "loss/critic": lc,
                    "loss/actor": la,
                    # "loss/meta": lm,
                    "loss/alpha": lalpha,
                    # "loss/auxiliary": loss_auxiliary.data.item(),
                    "log_prob/actor": log_prob.mean().data.item(),
                    # "log_prob/actor_old": log_prob_val.mean().data.item(),
                    # "log_prob/actor_new": log_prob_val_new.mean().data.item(),

                }
                data_for_logging[0].log(log_data, step=data_for_logging[1])

    def update_target(self, ):
        for param, target_param in zip(self.model.parameters(), self.target.parameters()):
            #calculate polyac average
            target_param.data.mul_(self.polyak)
            target_param.data.add_((1 - self.polyak) * param.data)


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
        # replay_buffer_val,
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
        # self.replay_buffer_val = replay_buffer_val
        self.actor_optimizer = actor_optimizer
        self.critic_optimizer = critic_optimizer
        self.meta_critic_optimizer = meta_critic_optimizer
        self.batch_size = batch_size
        self.polyak = polyak

        self.lr =lr
        self.device = model.device
        self.gamma = torch.as_tensor([gamma]).to(self.device)
        self.beta = torch.as_tensor([beta]).to(self.device)


        self.cql_alpha = 10.0
        self.cql_temp = 1.0
        self.cql_lagrange = True
        self.use_automatic_entropy_tuning = True
        self.cql_importance_sample = True
        self.cql_target_action_gap = 5.0
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

        feature_net = nn.Sequential(*list(self.model.actor.children())[:-2])
        self.hotplug = Hot_Plug(feature_net)

    def safe_exp(self, x):
        return torch.where(x < 50, torch.exp(x), x)

    def update(self, current_it, total_it, data_for_logging=None):
        grad_metrics = {}
        sample = self.replay_buffer.sample(self.batch_size)
        obs, next_obs, r_obs, next_r_obs, act, rwd, mc_returns, done = list(sample.values())

        # sample_val = self.replay_buffer_val.sample(self.batch_size)
        sample_val = self.replay_buffer.sample(self.batch_size)
        obs_val, next_obs_val, r_obs_val, next_r_obs_val, act_val, rwd_val, mc_returns_val, done_val = list(sample_val.values())

        act_target, log_prob, _, other_output= self.model.actor.sample((obs.to(self.device),r_obs.reshape(self.batch_size, 1, -1).to(self.device),))
        alpha_loss = -(self.log_alpha() * (log_prob + self.target_entropy).detach()).mean()
        alpha = self.log_alpha().exp() * self.alpha_multiplier

        with torch.no_grad():
            next_act_target, _, _, _= self.model.actor.sample((obs.to(self.device),r_obs.reshape(self.batch_size, 1, -1).to(self.device),))
            Q_target_1, Q_target_2 = self.target.critic((next_obs.to(self.device), next_r_obs.to(self.device)), next_act_target)
            Q_target_min = torch.min(torch.cat((Q_target_1, Q_target_2), 1), dim=1)[0].unsqueeze(-1)
            Q_target = rwd.to(self.device) + (self.gamma * Q_target_min) * done.to(self.device)
        Q1, Q2 = self.model.critic((obs.to(self.device), r_obs.to(self.device)),act.squeeze().to(self.device),)
        td_loss = F.mse_loss(Q_target, Q1) + F.mse_loss(Q_target, Q2)
        batch_size, action_dim = act.shape[0], act.shape[-1]
        cql_random_actions = act.squeeze().new_empty((batch_size, self.cql_n_actions, action_dim), requires_grad=False).uniform_(-1, 1)

        cql_current_actions, cql_current_log_pis, _, _ = self.model.actor.sample((obs.to(self.device),r_obs.reshape(self.batch_size, 1, -1).to(self.device),), repeat=self.cql_n_actions)
        cql_next_actions, cql_next_log_pis, _, _ = self.model.actor.sample((next_obs.to(self.device),next_r_obs.reshape(self.batch_size, 1, -1).to(self.device),), repeat=self.cql_n_actions)
        cql_current_actions, cql_current_log_pis = (cql_current_actions.detach(),cql_current_log_pis.detach(),)
        cql_next_actions, cql_next_log_pis = (cql_next_actions.detach(),cql_next_log_pis.detach(),)
        cql_q1_rand, cql_q2_rand = self.model.critic((obs.to(self.device),r_obs.reshape(self.batch_size, 1, -1).to(self.device),), cql_random_actions)
        cql_q1_current_actions, cql_q2_current_actions = self.model.critic((obs.to(self.device),r_obs.reshape(self.batch_size, 1, -1).to(self.device),), cql_current_actions)
        cql_q1_next_actions, cql_q2_next_actions = self.model.critic((obs.to(self.device),r_obs.reshape(self.batch_size, 1, -1).to(self.device),), cql_next_actions)

        # Calibration
        lower_bounds = mc_returns.reshape(-1, 1).repeat(1, cql_q1_current_actions.shape[1])

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

        cql_cat_q1 = torch.cat([cql_q1_rand,torch.unsqueeze(Q1, 1),cql_q1_next_actions,cql_q1_current_actions,],dim=1,)
        cql_cat_q2 = torch.cat([cql_q2_rand,torch.unsqueeze(Q2, 1),cql_q2_next_actions,cql_q2_current_actions,],dim=1,)
        cql_std_q1 = torch.std(cql_cat_q1, dim=1)
        cql_std_q2 = torch.std(cql_cat_q2, dim=1)

        if self.cql_importance_sample:
            random_density = np.log(0.5**action_dim)
            cql_cat_q1 = torch.cat([cql_q1_rand - random_density,cql_q1_next_actions - cql_next_log_pis.detach(),cql_q1_current_actions - cql_current_log_pis.detach(),],dim=1,)
            cql_cat_q2 = torch.cat([cql_q2_rand - random_density,cql_q2_next_actions - cql_next_log_pis.detach(),cql_q2_current_actions - cql_current_log_pis.detach(),],dim=1,)
        cql_qf1_ood = torch.logsumexp(cql_cat_q1 / self.cql_temp, dim=1) * self.cql_temp
        cql_qf2_ood = torch.logsumexp(cql_cat_q2 / self.cql_temp, dim=1) * self.cql_temp

        """Subtract the log likelihood of data"""
        cql_qf1_diff = torch.clamp(cql_qf1_ood - Q1, self.cql_clip_diff_min, self.cql_clip_diff_max,).mean()
        cql_qf2_diff = torch.clamp(cql_qf2_ood - Q2, self.cql_clip_diff_min, self.cql_clip_diff_max,).mean()

        if self.cql_lagrange:
            alpha_prime = torch.clamp(torch.exp(self.log_alpha_prime()), min=0.0, max=100.0)
            cql_min_qf1_loss = (alpha_prime* self.cql_alpha* (cql_qf1_diff - self.cql_target_action_gap))
            cql_min_qf2_loss = (alpha_prime* self.cql_alpha* (cql_qf2_diff - self.cql_target_action_gap))
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

        get_log_prob = self.model.actor.get_log_prob((obs.to(self.device),r_obs.reshape(self.batch_size, 1, -1).to(self.device),),act.squeeze().to(self.device),)
        loss_act = (alpha.detach() * log_prob - get_log_prob).mean()
        self.actor_optimizer.zero_grad()
        # #It is also for meta optimization
        loss_act.backward(retain_graph=True)#retain graph is for calculating loss.auxiliary.backward
        # #first psudo update for actor network
        self.hotplug.update(self.lr)

        act_target_val, log_prob_val, _, _= self.model.actor.sample((obs_val.to(self.device),r_obs_val.reshape(self.batch_size, 1, -1).to(self.device),))
        get_log_prob_val = self.model.actor.get_log_prob((obs_val.to(self.device),r_obs_val.reshape(self.batch_size, 1, -1).to(self.device),),act_val.squeeze().to(self.device),)
        policy_loss_val = (alpha.detach() * log_prob_val - get_log_prob_val).mean()
        #It is also for meta optimization
        loss_auxiliary.backward(create_graph=True)#create_graph is for computational graph from policy_loss_val_new to meta-critic model parameter 
        #first psudo update for actor network
        self.hotplug.update(self.lr)

        act_target_val_new, log_prob_val_new, _, _= self.model.actor.sample((obs_val.to(self.device),r_obs_val.reshape(self.batch_size, 1, -1).to(self.device),))
        get_log_prob_val_new = self.model.actor.get_log_prob((obs_val.to(self.device),r_obs_val.reshape(self.batch_size, 1, -1).to(self.device),),act_val.squeeze().to(self.device),)
        policy_loss_val_new = (alpha.detach() * log_prob_val_new - get_log_prob_val_new).mean()
        
        utility = policy_loss_val - policy_loss_val_new
        utility = torch.tanh(utility)
        loss_meta = -utility
        
        self.meta_critic_optimizer.zero_grad()
        grad_omega = torch.autograd.grad(loss_meta, self.model.meta_critic.parameters())
        # print(f"[Meta-Critic] Grad_omega calculated: {grad_omega[0] is not None}")
        for gradient, variable in zip(grad_omega, self.model.meta_critic.parameters()):
            variable.grad = gradient
        #update meta critic as a meta optimization
        grad_diversity_meta_critic = get_grad_direction_stats(self.model.meta_critic)
        meta_critic_grad_norm = get_grad_norm(self.model.meta_critic)
        meta_critic_weight_norm = get_weight_norm(self.model.meta_critic)
        self.meta_critic_optimizer.step()
        lm = loss_meta.data.item()
        #update actor as a meta optimization
        self.actor_optimizer.step()
        la = loss_act.data.item()
        #reset psudo update
        self.hotplug.restore()

        with torch.no_grad(): 
            if data_for_logging is not None:
                # grad_metrics.update(get_grad_norms(self.model.critic, "critic"))
                # grad_metrics.update(get_grad_norms(self.model.actor, "actor"))
                # grad_metrics.update(get_grad_norms(self.model.meta_critic, "meta_critic"))
                log_data = {
                    "loss/critic": lc,
                    "loss/actor": la,
                    "loss/meta": lm,
                    "loss/auxiliary": loss_auxiliary.data.item(),
                    "log_prob/now": get_log_prob.mean().data.item(),
                    "log_prob/old": get_log_prob_val.mean().data.item(),
                    "log_prob/new": get_log_prob_val_new.mean().data.item(),
                    "stats/avg_q1": Q1.mean().item(),
                    "stats/max_q1": Q1.max().item(),
                    "stats/alpha_prime": torch.exp(self.log_alpha_prime()).item(),
                    "stats/cql_diff": cql_qf1_diff.item(),
                    "stats/bound_active_rate": (cql_q1_current_actions == lower_bounds.unsqueeze(-1)).float().mean().item(), # 0に近いほど下限が効いていない
                    "stats/target_q_avg": Q_target.mean().item(),
                }
                # log_data.update(grad_metrics) # 勾配情報を追加
                data_for_logging[0].log(log_data, step=data_for_logging[1])


    def update_target(self):
        for param, target_param in zip(self.model.parameters(), self.target.parameters()):
            target_param.data.mul_(self.polyak)
            target_param.data.add_((1 - self.polyak) * param.data)