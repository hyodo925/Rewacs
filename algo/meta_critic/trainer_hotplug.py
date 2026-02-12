import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from .virtual_updater import VirtualActorUpdater, Hot_Plug
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
from .weight_clipping import WeightClippingAdam

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

def analyze_leaf_nodes(loss_tensor, model, name="Loss"):
    """
    計算グラフを遡り、葉ノードがモデルのどのパラメータか特定する
    """
    # モデルの全パラメータのアドレスを辞書化しておく
    param_map = {p.data_ptr(): name for name, p in model.named_parameters()}
    
    leaves = set()
    visited = set()

    def find_leaves(grad_fn):
        if grad_fn is None or grad_fn in visited:
            return
        visited.add(grad_fn)
        
        if hasattr(grad_fn, 'next_functions'):
            for next_f, _ in grad_fn.next_functions:
                if next_f is not None:
                    # 勾配が蓄積されるノード(葉)を確認
                    if "AccumulateGrad" in str(next_f):
                        if hasattr(next_f, 'variable'):
                            leaves.add(next_f.variable)
                    find_leaves(next_f)

    print(f"\n=== Graph Analysis for: {name} ===")
    find_leaves(loss_tensor.grad_fn)
    
    if not leaves:
        print("❌ 葉ノードが一つも見つかりませんでした。")
    else:
        for i, leaf in enumerate(leaves):
            ptr = leaf.data_ptr()
            # モデルのパラメータか、それ以外のテンソル（中間結果など）か
            param_name = param_map.get(ptr, "Unknown (Temporary Tensor or Replaced Param)")
            
            print(f"Leaf {i+1}:")
            print(f"  - Name: {param_name}")
            print(f"  - Size: {list(leaf.size())}")
            print(f"  - Requires_grad: {leaf.requires_grad}")
            
            # ここが重要：履歴(grad_fn)の有無
            if leaf.grad_fn is not None:
                print(f"  - 🔗 履歴あり: grad_fn={type(leaf.grad_fn).__name__}")
                print(f"    (メタ学習が成功していれば、ここに演算名が出ます)")
            else:
                print(f"  - 🍃 履歴なし: 純粋な葉ノードです")
    print("==========================================\n")

def print_grad_graph(fn, indent=0, visited=None):
    if visited is None:
        visited = set()
    
    if fn is None:
        return
    
    # 同じノードを何度も表示しない（グラフが合流するため）
    node_id = id(fn)
    already_visited = node_id in visited
    visited.add(node_id)
    
    # インデントと演算名の表示
    space = "  " * indent
    node_name = str(fn)
    
    # 特徴的なノードに色付け（文字情報）
    marker = ""
    if "AccumulateGrad" in node_name:
        marker = " <--- [Leaf Parameter]"
    elif "Meta" in node_name:
        marker = " <--- [!! Meta-Critic Related !!]"
    
    print(f"{space}|-- {node_name.split(' at ')[0]}{marker}{' (already printed)' if already_visited else ''}")
    
    # 再帰的に親（計算の元となった演算）を辿る
    if not already_visited and hasattr(fn, 'next_functions'):
        for next_f, _ in fn.next_functions:
            print_grad_graph(next_f, indent + 1, visited)

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

        self.updater = VirtualActorUpdater()
        self.anomaly_weight = 1.0
        self.normality_weight = 0.5
        self.handmade_weights = False

        #Weight Clipping
        # self.lr = 3e-4
        # weight_clipping = 0.5
        # clip_last_layer = 1
        # # self.weight_clipping = WeightClippingAdam()
        # self.actor_optimizer = WeightClippingAdam(self.model.actor.parameters(), lr=self.lr, eps=1e-5, zeta=weight_clipping, clip_last_layer=clip_last_layer)
        # self.critic_optimizer = WeightClippingAdam(self.model.critic.parameters(), lr=self.lr, eps=1e-5, zeta=weight_clipping, clip_last_layer=clip_last_layer)


    def safe_exp(self, x):
        return torch.where(x < 50, torch.exp(x), x)

    def update(self, data_for_logging=None):
        # torch.autograd.set_detect_anomaly(True)
        grad_metrics = {}
        sample = self.replay_buffer.sample(self.batch_size)
        obs, next_obs, r_obs, next_r_obs, act, rwd, done = list(sample.values())

        # sample_val = self.replay_buffer_val.sample(self.batch_size)
        sample_val = self.replay_buffer.sample(self.batch_size)
        obs_val, next_obs_val, r_obs_val, next_r_obs_val, act_val, rwd_val, done_val = list(sample_val.values())

        with torch.no_grad():
            old_mean, old_log_std, _ = self.model.actor((obs.to(self.device), r_obs.reshape(self.batch_size, 1, -1).to(self.device)))
            c_feats = self.model.critic.integrator(obs.to(self.device), r_obs.reshape(self.batch_size, 1, -1).to(self.device))
            if not self.model.critic.single:
                old_c_feats = torch.cat([c_feats, act.squeeze().to(self.device)], -1)

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
        grad_diversity_critic = get_grad_direction_stats(self.model.critic)
        critic_grad_norm = get_grad_norm(self.model.critic)
        critic_weight_norm = get_weight_norm(self.model.critic)
        params = dict(self.model.critic.named_parameters())
        # dot = make_dot(loss_critic, params=params)
        # dot.render("meta_critic_awac_critic_graph", format="png")

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
        get_log_prob = self.model.actor.get_log_prob((obs.to(self.device),r_obs.reshape(self.batch_size, 1, -1).to(self.device),),act.squeeze().to(self.device),)
        loss_act = -(get_log_prob* weights).mean()

        ##############################
        ## compute loss aux        ###
        ############################## 
        loss_auxiliary = self.model.meta_critic(
            (
                obs.reshape(self.batch_size, -1).to(self.device),
                r_obs.reshape(self.batch_size, -1).to(self.device),
            ),
            act.squeeze().to(self.device),
            other_output.reshape(self.batch_size, -1).to(self.device),
        )


        # params = dict(self.model.meta_critic.named_parameters())
        # dot = make_dot(loss_meta, params=params)
        # dot.render("meta_critic_awac_meta_critic_graph1", format="png")
        ################################################
        ### first pseudo update with loss act        ###
        ################################################
        all_params = list(self.model.actor.parameters())
        trainable_params = [p for p in self.model.actor.parameters() if p.requires_grad]
        # grads_critic = torch.autograd.grad(loss_act, self.model.actor.parameters(), retain_graph=True, allow_unused=True)

        computed_grads = torch.autograd.grad(loss_act, trainable_params, retain_graph=True, allow_unused=True)
        grads_critic = []
        grad_idx = 0
        for p in all_params:
            if p.requires_grad:
                grads_critic.append(computed_grads[grad_idx])
                grad_idx += 1
            else:
                grads_critic.append(None) 
        # grads_critic = [torch.clamp(g, -1.0, 1.0) if g is not None else None for g in grads_critic]
        self.updater.step(self.model.actor, grads_critic, "phi_old", 1e-3)

        old_param = self.updater.get("phi_old")
        qw_ref = torch.min(torch.cat((Q1, Q2), 1), dim=1)[0].reshape((-1, 1))
        pi_val, log_pi_val, *_ = self.model.actor.sample_with_params((obs_val.to(self.device),r_obs_val.reshape(self.batch_size, 1, -1).to(self.device),),params=old_param,)
        with torch.no_grad():
            v_act1, v_act2 = self.model.critic((obs_val.to(self.device), r_obs_val.to(self.device)),act=pi_val)
            qw_gen = torch.min(torch.cat((v_act1, v_act2), 1), dim=1)[0].reshape((-1, 1))
            adv = torch.max(torch.zeros_like(qw_ref), qw_ref - qw_gen)
            # adv = qw_ref - qw_gen
            # adv = (adv - adv.mean()) / (adv.std() + 1e-8)
            # adv = torch.clamp(adv, min=0.0) # AWACは非負の重みを期待
            weights = self.safe_exp(adv / self.beta)
        get_log_prob_old = self.model.actor.get_log_prob_with_params((obs_val.to(self.device),r_obs_val.reshape(self.batch_size, 1, -1).to(self.device),),act_val.squeeze().to(self.device),params=old_param)
        policy_loss_val = -(get_log_prob_old* weights).mean()

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
        all_params = list(self.model.actor.parameters())
        trainable_params_mcritic = [p for p in self.model.actor.parameters() if p.requires_grad]
        # grads_mcritic = torch.autograd.grad(loss_auxiliary, self.model.actor.parameters(), create_graph=True, allow_unused=True)
        computed_grads = torch.autograd.grad(loss_auxiliary, trainable_params_mcritic, create_graph=True, allow_unused=True)
        grads_mcritic = []
        grad_idx = 0
        for p in all_params:
            if p.requires_grad:
                grads_mcritic.append(computed_grads[grad_idx])
                grad_idx += 1
            else:
                grads_mcritic.append(None) 
        laux = loss_auxiliary.data.item()
        # print(f"DEBUG: grads_mcritic[0] grad_fn: {grads_mcritic[0].grad_fn}") # ここがNoneならcreate_graphが効いていない
        # grads_mcritic = [torch.clamp(g, -1.0, 1.0) if g is not None else None for g in grads_mcritic]
        self.updater.step(self.model.actor, grads_mcritic, "phi_new", 1e-3, from_params=old_param)
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
            # adv = qw_ref - qw_gen
            # adv = (adv - adv.mean()) / (adv.std() + 1e-8)
            # adv = torch.clamp(adv, min=0.0) # AWACは非負の重みを期待
            weights = self.safe_exp(adv / self.beta)
        get_log_prob_new = self.model.actor.get_log_prob_with_params((obs_val.to(self.device),r_obs_val.reshape(self.batch_size, 1, -1).to(self.device),),act_val.squeeze().to(self.device),params=new_param)
        policy_loss_val_new = -(get_log_prob_new* weights).mean()
        # print(f"DEBUG: policy_loss_val_new grad_fn: {policy_loss_val_new.grad_fn}")
        ###############################
        ### compute loss meta       ###
        ###############################
        utility = policy_loss_val - policy_loss_val_new
        utility = torch.tanh(utility)
        # utility = utility / (1 + torch.abs(utility)) 
        loss_meta = -utility
        params = dict(self.model.meta_critic.named_parameters())
        # dot = make_dot(loss_meta, params=params)
        # dot.render("meta_critic_awac_meta_critic_graph2", format="png")
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
        params = dict(self.model.actor.named_parameters())
        # dot = make_dot(loss_act+loss_auxiliary, params=params)
        # dot.render("meta_critic_awac_actor_graph", format="png")
        loss_auxiliary.backward(retain_graph=True)
        loss_act.backward()
        grad_diversity_actor = get_grad_direction_stats(self.model.actor)
        actor_grad_norm = get_grad_norm(self.model.actor)
        actor_weight_norm = get_weight_norm(self.model.actor)
        if data_for_logging is not None:
            grad_metrics.update(get_grad_norms(self.model.actor, "actor"))
        grad_diversity_meta_critic = get_grad_direction_stats(self.model.meta_critic)
        meta_critic_grad_norm = get_grad_norm(self.model.meta_critic)
        meta_critic_weight_norm = get_weight_norm(self.model.meta_critic)
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

        with torch.no_grad():
            new_mean, new_log_std, _ = self.model.actor((obs.to(self.device), r_obs.reshape(self.batch_size, 1, -1).to(self.device)))
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
        for param, target_param in zip(
            self.model.parameters(), self.target.parameters()
        ):
            target_param.data.mul_(self.polyak)
            target_param.data.add_((1 - self.polyak) * param.data)

    def finetune(self, data_for_logging=None):
        # torch.autograd.set_detect_anomaly(True)
        grad_metrics = {}
        sample = self.replay_buffer.sample(self.batch_size)
        obs, next_obs, r_obs, next_r_obs, act, rwd, done = list(sample.values())

        # sample_val = self.replay_buffer_val.sample(self.batch_size)
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
            grad_metrics.update(get_grad_norms(self.model.critic, "critic"))
            data_for_logging[0].log(
                {
                    "loss/critic": lc,
                },
                step=data_for_logging[1],
            )

        action_gen, log_prob, _, other_output = self.model.actor.sample((obs.to(self.device),r_obs.reshape(self.batch_size, 1, -1).to(self.device)))
        with torch.no_grad():
            qw_ref = torch.min(torch.cat((Q1, Q2), 1), dim=1)[0].reshape((-1, 1))
            v_act1, v_act2 = self.model.critic((obs.to(self.device), r_obs.to(self.device)),act=action_gen.detach(),)
            qw_gen = torch.min(torch.cat((v_act1, v_act2), 1), dim=1)[0].reshape((-1, 1))
            adv = torch.max(torch.zeros_like(qw_ref), qw_ref - qw_gen)
            weights = self.safe_exp(adv / self.beta)
        get_log_prob = self.model.actor.get_log_prob((obs.to(self.device),r_obs.reshape(self.batch_size, 1, -1).to(self.device),),act.squeeze().to(self.device),)
        
        # sample_weights = torch.where(switches.to(self.device) > 0.5, 1.0, 0.5)

        # if self.flow is not None:
        #     switching_score = self.flow.get_switching_score(obs.to(self.flow.device))
        #     # print(switching_score)
        #     scale = self.flow.theta * 0.1 
        #     sample_weights = 0.5 + 0.5 * torch.sigmoid((switching_score - self.flow.theta) / scale)
        # elif self.handmade_weights:
        #     sample_weights = torch.where(switches.to(self.device) > 0.5, 1.0, 0.5)
        # else:
        #     sample_weights = 1
        # Advantageベースの重み(weights)と、状況ベースの重み(sample_weights)を結合
        # combined_weights = weights * sample_weights.flatten()
        # loss_act = -(get_log_prob * combined_weights).mean()
        
        loss_act = -(get_log_prob* weights).mean()

        loss_auxiliary = self.model.meta_critic(
            (
                obs_val.reshape(self.batch_size, -1).to(self.device),
                r_obs_val.reshape(self.batch_size, -1).to(self.device),
            ),
            act_val.squeeze().to(self.device),
            other_output.reshape(self.batch_size, -1).to(self.device),
        )

        grads_critic = torch.autograd.grad(loss_act, self.model.actor.parameters(), create_graph=True, allow_unused=True)
        # grads_critic = [torch.clamp(g, -1.0, 1.0) if g is not None else None for g in grads_critic]
        self.updater.step(self.model.actor, grads_critic, "phi_old", 1e-3)
        old_param = self.updater.get("phi_old")
        qw_ref = torch.min(torch.cat((Q1, Q2), 1), dim=1)[0].reshape((-1, 1))
        pi_val, log_pi_val, *_ = self.model.actor.sample_with_params((obs_val.to(self.device),r_obs_val.reshape(self.batch_size, 1, -1).to(self.device),),params=old_param,)
        with torch.no_grad():
            v_act1, v_act2 = self.model.critic((obs_val.to(self.device), r_obs_val.to(self.device)),act=pi_val)
            qw_gen = torch.min(torch.cat((v_act1, v_act2), 1), dim=1)[0].reshape((-1, 1))
            adv = torch.max(torch.zeros_like(qw_ref), qw_ref - qw_gen)
            weights = self.safe_exp(adv / self.beta)
        get_log_prob_old = self.model.actor.get_log_prob_with_params((obs_val.to(self.device),r_obs_val.reshape(self.batch_size, 1, -1).to(self.device),),act_val.squeeze().to(self.device),params=old_param)
        # combined_weights = weights * sample_weights.flatten()
        policy_loss_val = -(get_log_prob_old* weights).mean()
        # policy_loss_val = -(get_log_prob_old* combined_weights).mean()

        grads_mcritic = torch.autograd.grad(loss_auxiliary, self.model.actor.parameters(), create_graph=True, allow_unused=True)
        # print(f"DEBUG: grads_mcritic[0] grad_fn: {grads_mcritic[0].grad_fn}") # ここがNoneならcreate_graphが効いていない
        # grads_mcritic = [torch.clamp(g, -1.0, 1.0) if g is not None else None for g in grads_mcritic]
        self.updater.step(self.model.actor, grads_mcritic, "phi_new", 1e-3, from_params=old_param)
        new_param = self.updater.get("phi_new")
        first_key = list(new_param.keys())[0]
        first_p = new_param[first_key]
        
        pi_val_new, log_pi_val_new, *_ = self.model.actor.sample_with_params((obs_val.to(self.device),r_obs_val.reshape(self.batch_size, 1, -1).to(self.device),),params=new_param)
        # print(f"DEBUG: log_pi_val_new grad_fn: {log_pi_val_new.grad_fn}")
        with torch.no_grad(): 
            v_act1, v_act2 = self.model.critic((obs_val.to(self.device), r_obs_val.to(self.device)),act=pi_val_new)
            qw_gen = torch.min(torch.cat((v_act1, v_act2), 1), dim=1)[0].reshape((-1, 1))
            adv = torch.max(torch.zeros_like(qw_ref), qw_ref - qw_gen)
            weights = self.safe_exp(adv / self.beta)
        get_log_prob_new = self.model.actor.get_log_prob_with_params((obs_val.to(self.device),r_obs_val.reshape(self.batch_size, 1, -1).to(self.device),),act_val.squeeze().to(self.device),params=new_param)
        # combined_weights = weights * sample_weights.flatten()
        policy_loss_val_new = -(get_log_prob_new* weights).mean()
        # policy_loss_val_new = -(get_log_prob_new* combined_weights).mean()

        utility = policy_loss_val - policy_loss_val_new
        utility = torch.tanh(utility)
        # utility = utility / (1 + torch.abs(utility)) 
        loss_meta = -utility
        self.meta_critic_optimizer.zero_grad()

        loss_meta.backward(retain_graph=True)
        if data_for_logging is not None:
            grad_metrics.update(get_grad_norms(self.model.meta_critic, "meta_critic"))

        self.actor_optimizer.zero_grad()
        # loss_auxiliary.backward(retain_graph=True)
        loss_act.backward()
        if data_for_logging is not None:
            grad_metrics.update(get_grad_norms(self.model.actor, "actor"))
        self.meta_critic_optimizer.step()
        lm = loss_meta.data.item()
        self.actor_optimizer.step()
        la = loss_act.data.item()
        with torch.no_grad(): 
            if data_for_logging is not None:
                log_data = {
                    "loss/critic": lc,
                    "loss/actor": la,
                    "loss/meta": lm,
                    "loss/auxiliary": loss_auxiliary.data.item(),
                    "log_prob/actor": get_log_prob.mean().data.item(),
                    "log_prob/actor_old": get_log_prob_old.mean().data.item(),
                    "log_prob/actor_new": get_log_prob_new.mean().data.item(),

                }
                log_data.update(grad_metrics) # 勾配情報を追加
                data_for_logging[0].log(log_data, step=data_for_logging[1])
       

    def visualize_computational_graph(self):
        grad_metrics = {}
        # sample = self.replay_buffer.sample(self.batch_size)
        obs, next_obs, r_obs, next_r_obs, act, rwd, done = list(self.replay_buffer.sample(1).values())

        # sample_val = self.replay_buffer_val.sample(self.batch_size)
        # sample_val = self.replay_buffer.sample(self.batch_size)
        obs_val, next_obs_val, r_obs_val, next_r_obs_val, act_val, rwd_val, done_val = list(self.replay_buffer.sample(1).values())

        with torch.no_grad():
            next_act_target, next_log_prob, *_ = self.model.actor.sample(
                (
                    next_obs.to(self.device),
                    next_r_obs.reshape(1, 1, -1).to(self.device),
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
            act.view(1,-1).to(self.device),
        )

        loss_critic = F.mse_loss(Q_target, Q1) + F.mse_loss(Q_target, Q2)
        # self.critic_optimizer.zero_grad()
        # loss_critic.backward()
        # self.critic_optimizer.step()

        action_gen, log_prob, _, other_output = self.model.actor.sample((obs.to(self.device),r_obs.reshape(1, 1, -1).to(self.device)))
        with torch.no_grad():
            qw_ref = torch.min(torch.cat((Q1, Q2), 1), dim=1)[0].reshape((-1, 1))
            v_act1, v_act2 = self.model.critic((obs.to(self.device), r_obs.to(self.device)),act=action_gen.detach(),)
            qw_gen = torch.min(torch.cat((v_act1, v_act2), 1), dim=1)[0].reshape((-1, 1))
            adv = torch.max(torch.zeros_like(qw_ref), qw_ref - qw_gen)
            weights = self.safe_exp(adv / self.beta)
        get_log_prob = self.model.actor.get_log_prob((obs.to(self.device),r_obs.reshape(1, 1, -1).to(self.device),),act.squeeze().to(self.device),)
        loss_act = -(get_log_prob* weights).mean()

        loss_auxiliary = self.model.meta_critic(
            (
                obs.reshape(1, -1).to(self.device),
                r_obs.reshape(1, -1).to(self.device),
            ),
            act.view(1,-1).to(self.device),
            other_output.reshape(1, -1).to(self.device),
        )

        grads_critic = torch.autograd.grad(loss_act, self.model.actor.parameters(), create_graph=True, allow_unused=True)
        # grads_critic = [torch.clamp(g, -1.0, 1.0) if g is not None else None for g in grads_critic]
        self.updater.step(self.model.actor, grads_critic, "phi_old", 1e-3)
        old_param = self.updater.get("phi_old")
        qw_ref = torch.min(torch.cat((Q1, Q2), 1), dim=1)[0].reshape((-1, 1))
        pi_val, log_pi_val, *_ = self.model.actor.sample_with_params((obs_val.to(self.device),r_obs_val.reshape(1, 1, -1).to(self.device),),params=old_param,)
        with torch.no_grad():
            v_act1, v_act2 = self.model.critic((obs_val.to(self.device), r_obs_val.to(self.device)),act=pi_val)
            qw_gen = torch.min(torch.cat((v_act1, v_act2), 1), dim=1)[0].reshape((-1, 1))
            adv = torch.max(torch.zeros_like(qw_ref), qw_ref - qw_gen)
            weights = self.safe_exp(adv / self.beta)
        get_log_prob_old = self.model.actor.get_log_prob_with_params((obs_val.to(self.device),r_obs_val.reshape(1, 1, -1).to(self.device),),act_val.view(1,-1).to(self.device),params=old_param)
        policy_loss_val = -(get_log_prob_old* weights).mean()

        grads_mcritic = torch.autograd.grad(loss_auxiliary, self.model.actor.parameters(), create_graph=True, allow_unused=True)
        self.updater.step(self.model.actor, grads_mcritic, "phi_new", 1e-3, from_params=old_param)
        new_param = self.updater.get("phi_new")
        
        pi_val_new, log_pi_val_new, *_ = self.model.actor.sample_with_params((obs_val.to(self.device),r_obs_val.reshape(1, 1, -1).to(self.device),),params=new_param)
        with torch.no_grad(): 
            v_act1, v_act2 = self.model.critic((obs_val.to(self.device), r_obs_val.to(self.device)),act=pi_val_new)
            qw_gen = torch.min(torch.cat((v_act1, v_act2), 1), dim=1)[0].reshape((-1, 1))
            adv = torch.max(torch.zeros_like(qw_ref), qw_ref - qw_gen)
            weights = self.safe_exp(adv / self.beta)
        get_log_prob_new = self.model.actor.get_log_prob_with_params((obs_val.to(self.device),r_obs_val.reshape(1, 1, -1).to(self.device),),act_val.view(1,-1).to(self.device),params=new_param)
        policy_loss_val_new = -(get_log_prob_new* weights).mean()
        utility = policy_loss_val - policy_loss_val_new
        utility = torch.tanh(utility)
        loss_meta = -utility
        self.meta_critic_optimizer.zero_grad()

        # loss_meta.backward(retain_graph=True)
        # self.actor_optimizer.zero_grad()
        # loss_auxiliary.backward(retain_graph=True)
        # loss_act.backward()

        # self.meta_critic_optimizer.step()
        # self.actor_optimizer.step()
        # 可視化
        params = dict(self.model.actor.named_parameters())
        dot = make_dot(loss_act+loss_auxiliary, params=params)
        dot.render("meta_critic_awac_actor_graph", format="png")

        params = dict(self.model.critic.named_parameters())
        dot = make_dot(loss_critic, params=params)
        dot.render("meta_critic_awac_critic_graph", format="png")

        params = dict(self.model.meta_critic.named_parameters())
        dot = make_dot(loss_meta, params=params)
        dot.render("meta_critic_awac_meta_critic_graph", format="png")

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

        # sample_val = self.replay_buffer_val.sample(self.batch_size)
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


        with torch.no_grad(): 
            if data_for_logging is not None:
                grad_metrics.update(get_grad_norms(self.model.critic, "critic"))
                grad_metrics.update(get_grad_norms(self.model.actor, "actor"))
                grad_metrics.update(get_grad_norms(self.model.meta_critic, "meta_critic"))
                log_data = {
                    "loss/critic": lc,
                    "loss/actor": la,
                    "loss/meta": lm,
                    "loss/auxiliary": loss_auxiliary.data.item(),
                    "log_prob/now": get_log_prob.mean().data.item(),
                    "log_prob/old": log_pi_val_old.mean().data.item(),
                    "log_prob/new": log_pi_val_new.mean().data.item(),

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


        with torch.no_grad(): 
            if data_for_logging is not None:
                grad_metrics.update(get_grad_norms(self.model.critic, "critic"))
                grad_metrics.update(get_grad_norms(self.model.actor, "actor"))
                grad_metrics.update(get_grad_norms(self.model.meta_critic, "meta_critic"))
                log_data = {
                    "loss/critic": lc,
                    "loss/actor": la,
                    "loss/meta": lm,
                    "loss/auxiliary": loss_auxiliary.data.item(),
                    "log_prob/now": get_log_prob.mean().data.item(),
                    "log_prob/old": log_pi_val_old.mean().data.item(),
                    "log_prob/new": log_pi_val_new.mean().data.item(),

                }
                log_data.update(grad_metrics) # 勾配情報を追加
                data_for_logging[0].log(log_data, step=data_for_logging[1])


    def update_target(self):
        for param, target_param in zip(
            self.model.parameters(), self.target.parameters()
        ):
            target_param.data.mul_(self.polyak)
            target_param.data.add_((1 - self.polyak) * param.data)



class MetaCriticSAC:
    def __init__(
        self,
        model,
        replay_buffer,
        # replay_buffer_val,
        actor_optimizer,
        critic_optimizer,
        meta_critic_optimizer,
        alpha_optimizer,
        batch_size,
        act_dim=2,
        lr=3e-4,
        polyak=0.995,
        gamma=0.9,
        beta=0.3,
        alpha=1.0
    ):
        self.alg_name = "MetaCriticSAC"
        self.model = model
        self.act_dim = act_dim
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
        self.alpha = torch.as_tensor([alpha]).to(self.device)
        self.updater = VirtualActorUpdater()
        self.auto_entropy_tuning = False
        target_entropy=None
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

    def update(self, data_for_logging=None):
        # torch.autograd.set_detect_anomaly(True)
        grad_metrics = {}
        sample = self.replay_buffer.sample(self.batch_size)
        obs, next_obs, r_obs, next_r_obs, act, rwd, done = list(sample.values())

        # sample_val = self.replay_buffer_val.sample(self.batch_size)
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
        grads_critic = [torch.clamp(g, -1.0, 1.0) if g is not None else None for g in grads_critic]
        self.updater.step(self.model.actor, grads_critic, "phi_old", self.lr)
        old_param = self.updater.get("phi_old")
        qw_ref = torch.min(torch.cat((Q1, Q2), 1), dim=1)[0].reshape((-1, 1))
        pi_val, log_pi_val, *_ = self.model.actor.sample_with_params((obs_val.to(self.device),r_obs_val.reshape(self.batch_size, 1, -1).to(self.device),),params=old_param,)

        Q1, Q2 = self.model.critic((obs_val.to(self.device), r_obs_val.to(self.device)), pi_val)
        Q_min = torch.min(torch.cat((Q1, Q2), 1), dim=1)[0].unsqueeze(-1)
        policy_loss_val = ((self.alpha * log_pi_val) - Q_min).mean()

        grads_mcritic = torch.autograd.grad(loss_auxiliary, self.model.actor.parameters(), create_graph=True, allow_unused=True)
        # print(f"DEBUG: grads_mcritic[0] grad_fn: {grads_mcritic[0].grad_fn}") # ここがNoneならcreate_graphが効いていない
        grads_mcritic = [torch.clamp(g, -1.0, 1.0) if g is not None else None for g in grads_mcritic]
        self.updater.step(self.model.actor, grads_mcritic, "phi_new", self.lr, from_params=old_param)
        new_param = self.updater.get("phi_new")


        pi_val_new, log_pi_val_new, *_ = self.model.actor.sample_with_params((obs_val.to(self.device),r_obs_val.reshape(self.batch_size, 1, -1).to(self.device),),params=new_param)
        Q1, Q2 = self.model.critic((obs_val.to(self.device), r_obs_val.to(self.device)), pi_val_new)
        Q_min_new = torch.min(torch.cat((Q1, Q2), 1), dim=1)[0].unsqueeze(-1)
        policy_loss_val_new = ((self.alpha * log_pi_val_new) - Q_min_new).mean()

        utility = policy_loss_val - policy_loss_val_new
        utility = torch.tanh(utility)
        loss_meta = -utility
        self.meta_critic_optimizer.zero_grad()

        loss_meta.backward(retain_graph=True)
        if data_for_logging is not None:
            grad_metrics.update(get_grad_norms(self.model.meta_critic, "meta_critic"))

        self.actor_optimizer.zero_grad()
        loss_auxiliary.backward(retain_graph=True)
        loss_act.backward()
        if data_for_logging is not None:
            grad_metrics.update(get_grad_norms(self.model.actor, "actor"))

        self.meta_critic_optimizer.step()
        lm = loss_meta.data.item()
        self.actor_optimizer.step()
        la = loss_act.data.item()

        if self.auto_entropy_tuning:
            alpha_loss = -(self.log_alpha * (log_prob + self.target_entropy).detach()).mean()
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()
            self.alpha = self.log_alpha.exp()

        with torch.no_grad(): 
            if data_for_logging is not None:
                log_data = {
                    "loss/critic": lc,
                    "loss/actor": la,
                    "loss/meta": lm,
                    "loss/auxiliary": loss_auxiliary.data.item(),
                    "log_prob": self.model.actor.get_log_prob((obs.to(self.device),r_obs.reshape(self.batch_size, 1, -1).to(self.device)),action_gen.squeeze().to(self.device),).mean().data.item()

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

        with torch.no_grad(): 
            if data_for_logging is not None:
                grad_metrics.update(get_grad_norms(self.model.critic, "critic"))
                grad_metrics.update(get_grad_norms(self.model.actor, "actor"))
                grad_metrics.update(get_grad_norms(self.model.meta_critic, "meta_critic"))
                log_data = {
                    "loss/critic": lc,
                    "loss/actor": la,
                    "loss/meta": lm,
                    "loss/auxiliary": loss_auxiliary.data.item(),
                    "log_prob": self.model.actor.get_log_prob((obs.to(self.device),r_obs.reshape(self.batch_size, 1, -1).to(self.device)),action_gen.squeeze().to(self.device),)

                }
                log_data.update(grad_metrics) # 勾配情報を追加
                data_for_logging[0].log(log_data, step=data_for_logging[1])


class MetaCriticNFMaxEnt:
    def __init__(
        self,
        model,
        replay_buffer,
        actor_optimizer,
        critic_optimizer,
        meta_critic_optimizer,
        batch_size,
        polyak=0.995,
        gamma=0.9,
        beta=0.3,
    ):
        self.alg_name = "MetaCriticNFMaxEnt"
        self.model = model
        self.target = copy.deepcopy(model)
        self.replay_buffer = replay_buffer
        self.actor_optimizer = actor_optimizer
        self.critic_optimizer = critic_optimizer
        self.meta_critic_optimizer = meta_critic_optimizer
        self.batch_size = batch_size
        self.polyak = polyak
        self.gamma = torch.as_tensor([gamma]).to(model.device)
        self.beta = torch.as_tensor([beta]).to(model.device)

        self.device = model.device
        self.updater = VirtualActorUpdater()


    def safe_exp(self, x):
        return torch.where(x < 50, torch.exp(x), x)

    def update(self, update_actor=False, data_for_logging=None):
        grad_metrics = {}
        sample = self.replay_buffer.sample(self.batch_size)
        obs, next_obs, r_obs, next_r_obs, act, rwd, done = list(sample.values())
        sample_val = self.replay_buffer.sample(self.batch_size)
        obs_val, next_obs_val, r_obs_val, next_r_obs_val, act_val, rwd_val, done_val = list(sample.values())
        bs = obs.shape[0]
        with torch.no_grad():
            prior_sample_target = self.model.actor.prior.sample((bs,))
            next_act_target = self.model.actor.reverse(
                prior_sample_target,
                (
                    next_obs.to(self.device),
                    next_r_obs.reshape(self.batch_size, 1, -1).to(self.device),
                ),
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




        prior_sample = self.model.actor.prior.sample((bs,))
        act_pi = self.model.actor.reverse(
            prior_sample,
            (
                obs.to(self.device),
                r_obs.reshape(self.batch_size, 1, -1).to(self.device),
            ),
        )

        # act_pi = (act + torch.randn_like(act)).to(self.device)

        Q1_pi, _ = self.model.critic(
            (obs.to(self.device), r_obs.to(self.device)),
            act_pi.squeeze(),
        )

        z, _, logdets = self.model.actor(
            act_pi.squeeze().detach(),
            (
                obs.to(self.device),
                r_obs.reshape(self.batch_size, 1, -1).to(self.device),
            ),
        )
        logprob = self.model.actor.prior.log_prob(z) + logdets

        z_bc, _, logdets_bc = self.model.actor(
            act.squeeze().to(self.device),
            (
                obs.to(self.device),
                r_obs.reshape(self.batch_size, 1, -1).to(self.device),
            ),
        )
        logprob_bc = self.model.actor.prior.log_prob(z_bc) + logdets_bc

        loss_act = (-Q1_pi + logprob).mean() + (-logprob_bc).mean()
        loss_auxiliary = self.model.meta_critic(
            (
                obs.reshape(self.batch_size, -1).to(self.device),
                r_obs.reshape(self.batch_size, -1).to(self.device),
            ),
            act.squeeze().to(self.device),
            z.reshape(self.batch_size, -1).to(self.device),
        )
        # for param in self.model.actor.parameters():
        #     print(param.requires_grad)
        actor_params = [p for p in self.model.actor.parameters() if p.requires_grad]
        grads_critic_subset = torch.autograd.grad(loss_act, actor_params, create_graph=True, allow_unused=True)
        grads_critic = []
        subset_idx = 0
        for p in self.model.actor.parameters():
            if p.requires_grad:
                grads_critic.append(grads_critic_subset[subset_idx])
                subset_idx += 1
            else:
                grads_critic.append(None)
        self.updater.step(self.model.actor, grads_critic, "phi_old", 1e-3)
        old_param = self.updater.get("phi_old")

        prior_sample = self.model.actor.prior.sample((bs,))
        act_pi = self.model.actor.reverse_with_params(prior_sample,(obs_val.to(self.device),r_obs_val.reshape(self.batch_size, 1, -1).to(self.device),),params=old_param)
        Q1_pi_val, _ = self.model.critic((obs_val.to(self.device), r_obs_val.to(self.device)),act_pi.squeeze(),)
        z, _, logdets = self.model.actor.forward_with_params(act_pi.squeeze().detach(),(obs_val.to(self.device),r_obs_val.reshape(self.batch_size, 1, -1).to(self.device),),params=old_param)
        logprob = self.model.actor.prior.log_prob(z) + logdets
        z_bc, _, logdets_bc = self.model.actor.forward_with_params(act_val.squeeze().to(self.device),(obs_val.to(self.device),r_obs_val.reshape(self.batch_size, 1, -1).to(self.device),),params=old_param)
        logprob_bc = self.model.actor.prior.log_prob(z_bc) + logdets_bc
        policy_loss_val = (-Q1_pi_val + logprob).mean() + (-logprob_bc).mean()

        grads_mcritic_subset = torch.autograd.grad(loss_auxiliary, actor_params, create_graph=True, allow_unused=True)
        grads_mcritic = []
        subset_idx = 0
        for p in self.model.actor.parameters():
            if p.requires_grad:
                grads_mcritic.append(grads_mcritic_subset[subset_idx])
                subset_idx += 1
            else:
                grads_mcritic.append(None)
        self.updater.step(self.model.actor, grads_mcritic, "phi_new", 1e-3, from_params=old_param)
        new_param = self.updater.get("phi_new")
        
        prior_sample = self.model.actor.prior.sample((bs,))
        act_pi_new = self.model.actor.reverse_with_params(prior_sample,(obs_val.to(self.device),r_obs_val.reshape(self.batch_size, 1, -1).to(self.device),),params=new_param)
        Q1_pi_new, _ = self.model.critic((obs_val.to(self.device), r_obs_val.to(self.device)),act_pi_new.squeeze(),)
        z_new, _, logdets_new = self.model.actor.forward_with_params(act_pi_new.squeeze().detach(),(obs_val.to(self.device),r_obs_val.reshape(self.batch_size, 1, -1).to(self.device),),params=new_param)
        logprob_new = self.model.actor.prior.log_prob(z_new) + logdets_new
        z_bc_new, _, logdets_bc_new = self.model.actor.forward_with_params(act.squeeze().to(self.device),(obs_val.to(self.device),r_obs_val.reshape(self.batch_size, 1, -1).to(self.device),),params=new_param)
        logprob_bc_new = self.model.actor.prior.log_prob(z_bc_new) + logdets_bc_new
        policy_loss_val_new = (-Q1_pi_new + logprob_new).mean() + (-logprob_bc_new).mean()
        
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

        self.actor_optimizer.zero_grad()
        loss_auxiliary.backward(retain_graph=True)
        loss_act.backward()
        if data_for_logging is not None:
            grad_metrics.update(get_grad_norms(self.model.actor, "actor"))

        self.meta_critic_optimizer.step()
        lm = loss_meta.data.item()
        self.actor_optimizer.step()
        la = loss_act.data.item()
        with torch.no_grad(): 
            if data_for_logging is not None:
                log_data = {
                    "loss/critic": lc,
                    "loss/actor": la,
                    "loss/meta": lm,
                    "loss/auxiliary": loss_auxiliary.data.item(),

                }
                log_data.update(grad_metrics) # 勾配情報を追加
                data_for_logging[0].log(log_data, step=data_for_logging[1])


    def finetune(self, update_actor=False, data_for_logging=None):
        grad_metrics = {}
        sample = self.replay_buffer.sample(self.batch_size)
        obs, next_obs, r_obs, next_r_obs, act, rwd, done = list(sample.values())
        sample_val = self.replay_buffer.sample(self.batch_size)
        obs_val, next_obs_val, r_obs_val, next_r_obs_val, act_val, rwd_val, done_val = list(sample.values())
        bs = obs.shape[0]
        with torch.no_grad():
            prior_sample_target = self.model.actor.prior.sample((bs,))
            next_act_target = self.model.actor.reverse(
                prior_sample_target,
                (
                    next_obs.to(self.device),
                    next_r_obs.reshape(self.batch_size, 1, -1).to(self.device),
                ),
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




        prior_sample = self.model.actor.prior.sample((bs,))
        act_pi = self.model.actor.reverse(
            prior_sample,
            (
                obs.to(self.device),
                r_obs.reshape(self.batch_size, 1, -1).to(self.device),
            ),
        )

        # act_pi = (act + torch.randn_like(act)).to(self.device)

        Q1_pi, _ = self.model.critic(
            (obs.to(self.device), r_obs.to(self.device)),
            act_pi.squeeze(),
        )

        z, _, logdets = self.model.actor(
            act_pi.squeeze().detach(),
            (
                obs.to(self.device),
                r_obs.reshape(self.batch_size, 1, -1).to(self.device),
            ),
        )
        logprob = self.model.actor.prior.log_prob(z) + logdets

        z_bc, _, logdets_bc = self.model.actor(
            act.squeeze().to(self.device),
            (
                obs.to(self.device),
                r_obs.reshape(self.batch_size, 1, -1).to(self.device),
            ),
        )
        logprob_bc = self.model.actor.prior.log_prob(z_bc) + logdets_bc

        loss_act = (-Q1_pi + logprob).mean() + (-logprob_bc).mean()
        # loss_act = (-Q1_pi + logprob).mean()
        # loss_act = (-Q1_pi).mean()
        loss_auxiliary = self.model.meta_critic(
            (
                obs.reshape(self.batch_size, -1).to(self.device),
                r_obs.reshape(self.batch_size, -1).to(self.device),
            ),
            act.squeeze().to(self.device),
            z.reshape(self.batch_size, -1).to(self.device),
        )
        # for param in self.model.actor.parameters():
        #     print(param.requires_grad)
        actor_params = [p for p in self.model.actor.parameters() if p.requires_grad]
        grads_critic_subset = torch.autograd.grad(loss_act, actor_params, create_graph=True, allow_unused=True)
        grads_critic = []
        subset_idx = 0
        for p in self.model.actor.parameters():
            if p.requires_grad:
                grads_critic.append(grads_critic_subset[subset_idx])
                subset_idx += 1
            else:
                grads_critic.append(None)
        self.updater.step(self.model.actor, grads_critic, "phi_old", 1e-3)
        old_param = self.updater.get("phi_old")

        prior_sample = self.model.actor.prior.sample((bs,))
        act_pi = self.model.actor.reverse_with_params(prior_sample,(obs_val.to(self.device),r_obs_val.reshape(self.batch_size, 1, -1).to(self.device),),params=old_param)
        Q1_pi_val, _ = self.model.critic((obs_val.to(self.device), r_obs_val.to(self.device)),act_pi.squeeze(),)
        z, _, logdets = self.model.actor.forward_with_params(act_pi.squeeze().detach(),(obs_val.to(self.device),r_obs_val.reshape(self.batch_size, 1, -1).to(self.device),),params=old_param)
        logprob = self.model.actor.prior.log_prob(z) + logdets
        z_bc, _, logdets_bc = self.model.actor.forward_with_params(act_val.squeeze().to(self.device),(obs_val.to(self.device),r_obs_val.reshape(self.batch_size, 1, -1).to(self.device),),params=old_param)
        logprob_bc = self.model.actor.prior.log_prob(z_bc) + logdets_bc
        policy_loss_val = (-Q1_pi_val + logprob).mean() + (-logprob_bc).mean()
        policy_loss_val = (-Q1_pi_val + logprob).mean()
        policy_loss_val = (-Q1_pi_val).mean()

        grads_mcritic_subset = torch.autograd.grad(loss_auxiliary, actor_params, create_graph=True, allow_unused=True)
        grads_mcritic = []
        subset_idx = 0
        for p in self.model.actor.parameters():
            if p.requires_grad:
                grads_mcritic.append(grads_mcritic_subset[subset_idx])
                subset_idx += 1
            else:
                grads_mcritic.append(None)
        self.updater.step(self.model.actor, grads_mcritic, "phi_new", 1e-3, from_params=old_param)
        new_param = self.updater.get("phi_new")
        
        prior_sample = self.model.actor.prior.sample((bs,))
        act_pi_new = self.model.actor.reverse_with_params(prior_sample,(obs_val.to(self.device),r_obs_val.reshape(self.batch_size, 1, -1).to(self.device),),params=new_param)
        Q1_pi_new, _ = self.model.critic((obs_val.to(self.device), r_obs_val.to(self.device)),act_pi_new.squeeze(),)
        z_new, _, logdets_new = self.model.actor.forward_with_params(act_pi_new.squeeze().detach(),(obs_val.to(self.device),r_obs_val.reshape(self.batch_size, 1, -1).to(self.device),),params=new_param)
        logprob_new = self.model.actor.prior.log_prob(z_new) + logdets_new
        z_bc_new, _, logdets_bc_new = self.model.actor.forward_with_params(act.squeeze().to(self.device),(obs_val.to(self.device),r_obs_val.reshape(self.batch_size, 1, -1).to(self.device),),params=new_param)
        logprob_bc_new = self.model.actor.prior.log_prob(z_bc_new) + logdets_bc_new
        policy_loss_val_new = (-Q1_pi_new + logprob_new).mean() + (-logprob_bc_new).mean()
        policy_loss_val_new = (-Q1_pi_new + logprob_new).mean()
        policy_loss_val_new = (-Q1_pi_new).mean()

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

        self.actor_optimizer.zero_grad()
        loss_auxiliary.backward(retain_graph=True)
        loss_act.backward()
        if data_for_logging is not None:
            grad_metrics.update(get_grad_norms(self.model.actor, "actor"))

        self.meta_critic_optimizer.step()
        lm = loss_meta.data.item()
        self.actor_optimizer.step()
        la = loss_act.data.item()
        with torch.no_grad(): 
            if data_for_logging is not None:
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