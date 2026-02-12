import math
from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn


def l1_penalty(var):
    return torch.abs(var).sum()


class Hot_Plug(object):
    def __init__(self, model):
        self.model = model
        self.params = OrderedDict(self.model.named_parameters())

    def update(self, lr=0.1):
        for param_name in self.params.keys():
            path = param_name.split(".")
            cursor = self.model
            for module_name in path[:-1]:
                cursor = cursor._modules[module_name]
            if lr > 0:
                cursor._parameters[path[-1]] = (
                    self.params[param_name] - lr * self.params[param_name].grad
                )
            else:
                cursor._parameters[path[-1]] = self.params[param_name]

    def restore(self):
        self.update(lr=0)


def get_kl_divergence(old_dist_params, new_dist_params):
    mu1, log_std1 = old_dist_params
    mu2, log_std2 = new_dist_params

    var1 = torch.exp(log_std1)
    var2 = torch.exp(log_std2)

    kl = 0.5 * (torch.log(var2 / var1) + (var1.pow(2) + (mu1 - mu2).pow(2)) / var2.pow(2) - 1.0)
    return kl.mean().item()

def get_wasserstein_dist(feat1, feat2):
    mu1, mu2 = feat1.mean(0), feat2.mean(0)
    var1, var2 = feat1.var(0), feat2.var(0)
    mean_diff = torch.norm(mu1 - mu2, p=2)
    var_diff = torch.norm(var1.sqrt() - var2.sqrt(), p=2)
    
    return (mean_diff + var_diff).item()

def get_mmd(feat1, feat2, kernel='rbf'):
    def guassian_kernel(x, y, sigma=1.0):
        beta = 1.0 / (2.0 * sigma**2)
        dist = torch.cdist(x, y).pow(2)
        return torch.exp(-beta * dist)

    x = feat1.reshape(feat1.size(0), -1)
    y = feat2.reshape(feat2.size(0), -1)
    
    xx = guassian_kernel(x, x).mean()
    yy = guassian_kernel(y, y).mean()
    xy = guassian_kernel(x, y).mean()
    
    return (xx + yy - 2 * xy).sqrt().item()

def get_js_divergence(p_params, q_params):
    mu1, var1 = p_params
    mu2, var2 = q_params
    def kl_with_log_p(log_p, log_q):
        return (log_p.exp() * (log_p - log_q)).sum(-1).mean()
    m_mu = 0.5 * (mu1 + mu2)
    m_var = 0.5 * (var1 + var2)
    
    kl_pm = get_kl_divergence((mu1, var1), (m_mu, m_var))
    kl_qm = get_kl_divergence((mu2, var2), (m_mu, m_var))
    
    return 0.5 * (kl_pm + kl_qm)

def get_hellinger_distance(p_params, q_params):
    mu1, var1 = p_params
    mu2, var2 = q_params
    
    var_sum = (var1 + var2) / 2
    bc = torch.sqrt(torch.sqrt(var1 * var2) / var_sum) * \
         torch.exp(-0.25 * (mu1 - mu2).pow(2) / (var1 + var2))
    
    h_squared = torch.clamp(1 - bc, min=0)
    return torch.sqrt(h_squared).mean().item()

def get_bhattacharyya_distance(p_params, q_params):
    mu1, var1 = p_params
    mu2, var2 = q_params
    
    term1 = 0.25 * (mu1 - mu2).pow(2) / (var1 + var2)
    term2 = 0.5 * torch.log((var1 + var2) / (2 * torch.sqrt(var1 * var2) + 1e-8))
    db = term1 + term2
    return db.mean().item()

def get_mahalanobis_distance(u, v, cov):
    delta = u - v
    inv_cov = torch.inverse(cov + torch.eye(cov.size(0)).to(cov.device) * 1e-6)
    
    m_dist = torch.sqrt(torch.matmul(torch.matmul(delta, inv_cov), delta.t()))
    return m_dist.diag().mean().item()


def get_grad_norm(model_part):
    total_norm = 0
    for p in model_part.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
    return total_norm ** 0.5

def get_weight_norm(model_part):
    total_norm = 0
    for p in model_part.parameters():
        param_norm = p.data.norm(2)
        total_norm += param_norm.item() ** 2
    return total_norm ** 0.5

def get_dormant_units_ratio(features, threshold=1e-6):
    flat_features = features.reshape(features.size(0), -1)
    is_dormant = torch.all(flat_features <= threshold, dim=0)
    return is_dormant.float().mean().item()

def get_effective_rank(features):
    flat_features = features.reshape(features.size(0), -1)
    try:
        sv = torch.linalg.svdvals(flat_features)
        sv_sum = torch.sum(sv)
        if sv_sum < 1e-10: return 0.0
        
        p = sv / sv_sum
        entropy = -torch.sum(p * torch.log(p + 1e-12))
        return torch.exp(entropy).item()
    except Exception:
        return 0.0

def get_approximate_rank(features, prop=0.99):
    flat_features = features.reshape(features.size(0), -1)
    try:
        sv = torch.linalg.svdvals(flat_features)
        sqrd_sv = sv ** 2
        normed_sqrd_sv = torch.sort(sqrd_sv / torch.sum(sqrd_sv), descending=True)[0]
        cum_sum = torch.cumsum(normed_sqrd_sv, dim=0)
        return (cum_sum < prop).sum().item() + 1
    except Exception:
        return 0

def get_abs_approximate_rank(features, prop=0.99):
    flat_features = features.reshape(features.size(0), -1)
    try:
        sv = torch.linalg.svdvals(flat_features)
        normed_sv = torch.sort(sv / torch.sum(sv), descending=True)[0]
        cum_sum = torch.cumsum(normed_sv, dim=0)
        return (cum_sum < prop).sum().item() + 1
    except Exception:
        return 0
    
def get_grad_direction_stats(model_part):
    grad_stats = {}
    
    for name, p in model_part.named_parameters():
        if p.grad is not None and p.grad.ndim >= 2:
            g = p.grad.data.reshape(p.grad.size(0), -1)
            
            try:
                # SVDで特異値を計算
                sv = torch.linalg.svdvals(g)
                sv_sum = torch.sum(sv)
                
                if sv_sum > 1e-10:
                    p_dist = sv / sv_sum
                    entropy = -torch.sum(p_dist * torch.log(p_dist + 1e-12))
                    eff_rank = torch.exp(entropy)
                    max_rank = min(g.shape)
                    relative_rank = (eff_rank / max_rank).item()
                    
                    grad_stats[name] = relative_rank
            except Exception:
                continue
                
    if not grad_stats:
        return 0.0
        
    return sum(grad_stats.values()) / len(grad_stats)

def extend_and_repeat(state, dim: int, repeat: int):
    if isinstance(state, tuple):
        return tuple(
            s.unsqueeze(dim).repeat_interleave(repeat, dim=dim)
            for s in state
        )
    else:
        return state.unsqueeze(dim).repeat_interleave(repeat, dim=dim)