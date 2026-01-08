# Copyright 2022 Twitter, Inc and Zhendong Wang.
# SPDX-License-Identifier: Apache-2.0

import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torchdiffeq import odeint

from dql.helpers import (cosine_beta_schedule,
                            linear_beta_schedule,
                            vp_beta_schedule,
                            extract,
                            Losses)
#from utils.utils import Progress, Silent

def somooth_compute_cost(vector, noisy_actions):
    cost = vector - noisy_actions*0.25
    return cost
# def compute_cost(vector, noisy_actions, resolution=1, inscribed_radius=0.3, weight=10.0, max_cost=200):
#     noisy_actions.requires_grad = True

#     LETHAL_OBSTACLE = torch.tensor(max_cost, requires_grad=False, dtype=torch.float32).to(noisy_actions.device)
#     INSCRIBED_INFLATED_OBSTACLE = torch.tensor(max_cost * 0.95, requires_grad=False, dtype=torch.float32).to(noisy_actions.device)

#     distance = torch.norm(vector + noisy_actions*0.25, p=2)
#     x_distance = vector[0,0] + noisy_actions[0,0]*0.25
#     if -2.7 < x_distance <= 0:
#         cost = LETHAL_OBSTACLE
#     if 0 < x_distance * resolution <= inscribed_radius:
#         cost = LETHAL_OBSTACLE * 0.95
#     else:
#         euclidean_distance = distance * resolution
#         factor = torch.exp(-1.0 * weight * (euclidean_distance - inscribed_radius))
#         cost = INSCRIBED_INFLATED_OBSTACLE * factor
#     return vector/torch.norm(vector)*cost,x_distance
def compute_cost(position_wall, noisy_actions, robot_pos, resolution=1, inscribed_radius=0.3, weight=2.4, max_cost=500):
    LETHAL_OBSTACLE = torch.tensor(max_cost, dtype=torch.float32).to(noisy_actions.device)
    INSCRIBED_INFLATED_OBSTACLE = torch.tensor(max_cost * 0.95, dtype=torch.float32).to(noisy_actions.device)
    # noisy_actions = torch.clamp(noisy_actions, min=-1.0, max=1.0)
    next_robot_pos = robot_pos + noisy_actions * 0.25
    vector = next_robot_pos - position_wall
    guide_vec = robot_pos - position_wall
    distance = torch.norm(vector)
    scale = max(0, (distance - inscribed_radius) / distance)
    distance = distance*scale
    vector = vector*scale
    # print(f"position:{robot_pos,noisy_actions}")
    if vector[0,0] <= 0 and robot_pos[0,0] < 0:
        cost = LETHAL_OBSTACLE
        # print(f"LETHAL_OBSTACLE:{next_robot_pos,robot_pos,noisy_actions}")
    elif 0 < vector[0,0] * resolution <= inscribed_radius:
        cost = INSCRIBED_INFLATED_OBSTACLE
        # print(f"INSCRIBED_INFLATED_OBSTACLE:{next_robot_pos,robot_pos,noisy_actions}")
    else:
        euclidean_distance = distance * resolution
        factor = torch.exp(-1.0 * weight * (euclidean_distance - inscribed_radius))
        cost = INSCRIBED_INFLATED_OBSTACLE * factor
        # print(f"NORMAL:{next_robot_pos,robot_pos,noisy_actions}")
    # print(f"guide:{guide_vec/torch.norm(guide_vec)*cost}")
    return guide_vec/torch.norm(guide_vec)*cost,vector

class Progress:

	def __init__(self, total, name='Progress', ncol=3, max_length=20, indent=0, line_width=100, speed_update_freq=100):
		self.total = total
		self.name = name
		self.ncol = ncol
		self.max_length = max_length
		self.indent = indent
		self.line_width = line_width
		self._speed_update_freq = speed_update_freq

		self._step = 0
		self._prev_line = '\033[F'
		self._clear_line = ' ' * self.line_width

		self._pbar_size = self.ncol * self.max_length
		self._complete_pbar = '#' * self._pbar_size
		self._incomplete_pbar = ' ' * self._pbar_size

		self.lines = ['']
		self.fraction = '{} / {}'.format(0, self.total)

		self.resume()

	def update(self, description, n=1):
		self._step += n
		if self._step % self._speed_update_freq == 0:
			self._time0 = time.time()
			self._step0 = self._step
		self.set_description(description)

	def resume(self):
		self._skip_lines = 1
		print('\n', end='')
		self._time0 = time.time()
		self._step0 = self._step

	def pause(self):
		self._clear()
		self._skip_lines = 1

	def set_description(self, params=[]):

		if type(params) == dict:
			params = sorted([
				(key, val)
				for key, val in params.items()
			])

		############
		# Position #
		############
		self._clear()

		###########
		# Percent #
		###########
		percent, fraction = self._format_percent(self._step, self.total)
		self.fraction = fraction

		#########
		# Speed #
		#########
		speed = self._format_speed(self._step)

		##########
		# Params #
		##########
		num_params = len(params)
		nrow = math.ceil(num_params / self.ncol)
		params_split = self._chunk(params, self.ncol)
		params_string, lines = self._format(params_split)
		self.lines = lines

		description = '{} | {}{}'.format(percent, speed, params_string)
		print(description)
		self._skip_lines = nrow + 1

	def append_description(self, descr):
		self.lines.append(descr)

	def _clear(self):
		position = self._prev_line * self._skip_lines
		empty = '\n'.join([self._clear_line for _ in range(self._skip_lines)])
		print(position, end='')
		print(empty)
		print(position, end='')

	def _format_percent(self, n, total):
		if total:
			percent = n / float(total)

			complete_entries = int(percent * self._pbar_size)
			incomplete_entries = self._pbar_size - complete_entries

			pbar = self._complete_pbar[:complete_entries] + self._incomplete_pbar[:incomplete_entries]
			fraction = '{} / {}'.format(n, total)
			string = '{} [{}] {:3d}%'.format(fraction, pbar, int(percent * 100))
		else:
			fraction = '{}'.format(n)
			string = '{} iterations'.format(n)
		return string, fraction

	def _format_speed(self, n):
		num_steps = n - self._step0
		t = time.time() - self._time0
		speed = num_steps / t
		string = '{:.1f} Hz'.format(speed)
		if num_steps > 0:
			self._speed = string
		return string

	def _chunk(self, l, n):
		return [l[i:i + n] for i in range(0, len(l), n)]

	def _format(self, chunks):
		lines = [self._format_chunk(chunk) for chunk in chunks]
		lines.insert(0, '')
		padding = '\n' + ' ' * self.indent
		string = padding.join(lines)
		return string, lines

	def _format_chunk(self, chunk):
		line = ' | '.join([self._format_param(param) for param in chunk])
		return line

	def _format_param(self, param):
		k, v = param
		return '{} : {}'.format(k, v)[:self.max_length]

	def stamp(self):
		if self.lines != ['']:
			params = ' | '.join(self.lines)
			string = '[ {} ] {}{} | {}'.format(self.name, self.fraction, params, self._speed)
			self._clear()
			print(string, end='\n')
			self._skip_lines = 1
		else:
			self._clear()
			self._skip_lines = 0

	def close(self):
		self.pause()


class Silent:

	def __init__(self, *args, **kwargs):
		pass

	def __getattr__(self, attr):
		return lambda *args: None
class Diffusion(nn.Module):
    def __init__(self, state_dim, action_dim, model, max_action,
                 beta_schedule='linear', n_timesteps=100,
                 loss_type='l2', clip_denoised=True, predict_epsilon=True,random_sample=True):
        super(Diffusion, self).__init__()

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.max_action = max_action
        self.model = model
        self.random_sample = random_sample
        if beta_schedule == 'linear':
            betas = linear_beta_schedule(n_timesteps)
        elif beta_schedule == 'cosine':
            betas = cosine_beta_schedule(n_timesteps)
        elif beta_schedule == 'vp':
            betas = vp_beta_schedule(n_timesteps)

        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, axis=0)
        alphas_cumprod_prev = torch.cat([torch.ones(1), alphas_cumprod[:-1]])

        self.n_timesteps = int(n_timesteps)
        self.clip_denoised = clip_denoised
        self.predict_epsilon = predict_epsilon

        self.register_buffer('betas', betas)
        self.register_buffer('alphas_cumprod', alphas_cumprod)
        self.register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1. - alphas_cumprod))
        self.register_buffer('log_one_minus_alphas_cumprod', torch.log(1. - alphas_cumprod))
        self.register_buffer('sqrt_recip_alphas_cumprod', torch.sqrt(1. / alphas_cumprod))
        self.register_buffer('sqrt_recipm1_alphas_cumprod', torch.sqrt(1. / alphas_cumprod - 1))

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)
        self.register_buffer('posterior_variance', posterior_variance)

        ## log calculation clipped because the posterior variance
        ## is 0 at the beginning of the diffusion chain
        self.register_buffer('posterior_log_variance_clipped',
                             torch.log(torch.clamp(posterior_variance, min=1e-20)))
        self.register_buffer('posterior_mean_coef1',
                             betas * np.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod))
        self.register_buffer('posterior_mean_coef2',
                             (1. - alphas_cumprod_prev) * np.sqrt(alphas) / (1. - alphas_cumprod))

        self.loss_fn = Losses[loss_type]()

    # ------------------------------------------ distillution --------------------------------------#
    def p_sample_loop_t2_t1(self, state, x1,t2, t1 ,shape, verbose=False, return_diffusion=False):
        device = self.betas.device

        batch_size = shape[0]
        if return_diffusion: diffusion = [x]

        progress = Progress(self.n_timesteps) if verbose else Silent()
        for i in reversed(range(int(t1.max().item()) + 1, int(t2.max().item()))):
            timesteps = torch.full((batch_size,), i, device=device, dtype=torch.long)
            x = self.p_sample(x, timesteps, state)

            progress.update({'t': i})

            if return_diffusion: diffusion.append(x)

        progress.close()

        if return_diffusion:
            return x, torch.stack(diffusion, dim=1)
        else:
            return x
    def vp_heun_step(self, x_t, t, t_next):
        """
        Heun's method for DDPM (VP-style) ODE.
        """
        beta_t = extract(self.betas, t, x_t.shape)
        alpha_bar_t = extract(self.alphas_cumprod, t, x_t.shape)
        sqrt_alpha_bar_t = alpha_bar_t.sqrt()

        eps1 = self.model(x_t, t)  # predict ε₁
        drift1 = -0.5 * beta_t * (x_t + eps1 / sqrt_alpha_bar_t)
        x_euler = x_t + (t_next - t) * drift1

        beta_tp = extract(self.betas, t_next, x_t.shape)
        alpha_bar_tp = extract(self.alphas_cumprod, t_next, x_t.shape)
        sqrt_alpha_bar_tp = alpha_bar_tp.sqrt()
        eps2 = self.model(x_euler, t_next)  # predict ε₂
        drift2 = -0.5 * beta_tp * (x_euler + eps2 / sqrt_alpha_bar_tp)

        x_next = x_t + 0.5 * (t_next - t) * (drift1 + drift2)
        return x_next
    def ode_reverse_step_heun_vp_to_ve(self, state, action, x_t, t_next, t_prev, z):
        batch_size, dims = x_t.shape[0], x_t.dim()

        if not torch.is_tensor(t_next):
            t_next = torch.full((batch_size,), t_next, dtype=torch.float32, device=x_t.device)
        if not torch.is_tensor(t_prev):
            t_prev = torch.full((batch_size,), t_prev, dtype=torch.float32, device=x_t.device)

        delta_t = (t_prev - t_next).view(-1, *([1] * (dims - 1)))

        t_next_long = t_next.view(-1).long()
        t_prev_long = t_prev.view(-1).long()

        # beta_t = extract(self.betas, t_next_long, x_t.shape)
        # beta_tp = extract(self.betas, t_prev_long, x_t.shape)

        alpha_bar_t = extract(self.alphas_cumprod, t_next_long, x_t.shape)
        alpha_bar_tp = extract(self.alphas_cumprod, t_prev_long, x_t.shape)

        sqrt_alpha_bar_t = alpha_bar_t.sqrt()
        sqrt_one_minus_alpha_bar_t = (1 - alpha_bar_t).sqrt()

        sqrt_alpha_bar_tp = alpha_bar_tp.sqrt()
        sqrt_one_minus_alpha_bar_tp = (1 - alpha_bar_tp).sqrt()

        # VPスケールの状態生成
        x_vp_prev = sqrt_alpha_bar_tp * action + sqrt_one_minus_alpha_bar_tp * z
        x_vp_next = sqrt_alpha_bar_t * action + sqrt_one_minus_alpha_bar_t * z
        # Step 1: モデル予測 ε₁
        # eps1 = self.model(x_vp, t_next.view(-1), state)
        # drift1 = -0.5 * beta_t * (x_vp + eps1 / sqrt_alpha_bar_t)
        # x_euler_vp = x_vp + delta_t * drift1

        # # Step 2: モデル予測 ε₂ at t_prev
        # eps2 = self.model(x_euler_vp, t_prev.view(-1), state)
        # drift2 = -0.5 * beta_tp * (x_euler_vp + eps2 / sqrt_alpha_bar_tp)

        # Heun補正
        # x_prev_vp
        x_phi = x_t + delta_t * (self.model(x_vp_prev, t_prev.view(-1), state))

        # VP -> VE変換（ノイズ除去）
        # x_prev = (x_prev_vp - sqrt_one_minus_alpha_bar_tp * z) / sqrt_alpha_bar_tp
        # print("x_prev:",x_prev[0])
        # print("x_prev_vp:",x_prev_vp[0])
        # print("t1:",t_next[0])
        # print("t2:",t_prev[0])
        # print("x0:",x_prev[0]-t_next[0]*z[0])
        # print("action:",action[0])
        # print("x_phi",x_phi[0])
        return x_phi

    def score_model(self, x, t, s):
        # return - 0.5*extract(self.sqrt_recipm1_alphas_cumprod, t.long(), x.shape) * self.model(x,t,s)
        sqrt_alpha_bar = self.sqrt_alphas_cumprod.gather(0, t.long()).view(-1, *[1]*(x.dim()-1))  # [B, 1]
        sqrt_one_minus_alpha_bar = self.sqrt_one_minus_alphas_cumprod.gather(0, t.long()).view(-1, *[1]*(x.dim()-1))  # [B, 1]

        # Compute DDIM-like deterministic score: dx/dt
        dx_dt = (x - sqrt_alpha_bar * self.model(x,t,s)) / sqrt_one_minus_alpha_bar  # [B, D]
        return dx_dt
        # return extract(self.sqrt_recip_alphas_cumprod, t.long(), x.shape) * x - 0.5*extract(self.sqrt_recipm1_alphas_cumprod, t.long(), x.shape) * self.model(x,t,s) #dx/dt
    def ode_reverse_step(self, state, x_next, t_next, t_prev):
        batch_size = x_next.shape[0]

        delta_t = (t_prev - t_next)  # Shape: [batch_size]
        t_tensor = t_next.view(-1)   # Ensure t is batch-specific

        score = self.score_model(x_next, t_tensor, state)

        # Reshape delta_t for broadcasting
        delta_t = delta_t.view(-1, *[1]*(x_next.dim()-1))  # e.g., [batch_size, 1] or [batch_size, 1, 1]

        x_prev = x_next + delta_t * score
        # print(x_prev[0],x_next[0],delta_t[0],t_tensor[0])#,self.model(x_next,t_tensor,state)[0]
        return x_prev

    # ------------------------------------------ sampling ------------------------------------------#
    def predict_start_from_noise(self, x_t, t, noise):
        '''
            if self.predict_epsilon, model output is (scaled) noise;
            otherwise, model predicts x0 directly
        '''
        if self.predict_epsilon:
            return (
                    extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
                    extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
            )
        else:
            return noise

    def q_posterior(self, x_start, x_t, t):
        posterior_mean = (
                extract(self.posterior_mean_coef1, t, x_t.shape) * x_start +
                extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract(self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def p_mean_variance(self, x, t, s):
        x_recon = self.predict_start_from_noise(x, t=t, noise=self.model(x, t, s))

        if self.clip_denoised:
            x_recon.clamp_(-self.max_action, self.max_action)
        else:
            assert RuntimeError()

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(x_start=x_recon, x_t=x, t=t)
        return model_mean, posterior_variance, posterior_log_variance

    # @torch.no_grad()
    def p_sample(self, x, t, s):
        b, *_, device = *x.shape, x.device
        model_mean, _, model_log_variance = self.p_mean_variance(x=x, t=t, s=s)
        noise = torch.randn_like(x)
        # no noise when t == 0
        nonzero_mask = (1 - (t == 0).float()).reshape(b, *((1,) * (len(x.shape) - 1)))
        return model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise *0.2*self.random_sample#*0.2 * self.random_sample

    # @torch.no_grad()
    def p_sample_loop(self, state, shape, verbose=False, return_diffusion=False):
        device = self.betas.device

        batch_size = shape[0]
        if self.random_sample:
            x = torch.randn(shape, device=device)
        else:
            x = torch.zeros(shape,device=device)
        if return_diffusion: diffusion = [x]

        progress = Progress(self.n_timesteps) if verbose else Silent()
        for i in reversed(range(0, self.n_timesteps)):
            timesteps = torch.full((batch_size,), i, device=device, dtype=torch.long)
            x = self.p_sample(x, timesteps, state)

            progress.update({'t': i})

            if return_diffusion: diffusion.append(x)

        progress.close()

        if return_diffusion:
            return x, torch.stack(diffusion, dim=1)
        else:
            return x

    # @torch.no_grad()
    def sample(self, state, *args, **kwargs):
        batch_size = state.shape[0]
        shape = (batch_size, self.action_dim)
        action = self.p_sample_loop(state, shape, *args, **kwargs)
        return action.clamp_(-self.max_action, self.max_action)

    # ------------------------------------------ training ------------------------------------------#

    def q_sample(self, x_start, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start)

        sample = (
                extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
                extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )

        return sample

    def p_losses(self, x_start, state, t, x_noisy ,weights=1.0):
        #noise = torch.randn_like(x_start)

        #x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)

        x_recon = self.model(x_noisy, t, state)


        #if self.predict_epsilon:
        #    loss = self.loss_fn(x_recon, noise, weights)
        #else:
        loss = self.loss_fn(x_recon, x_start, weights)

        return loss

    def loss(self, x, state, weights=1.0):
        batch_size = len(x)
        t = torch.randint(0, self.n_timesteps, (batch_size,), device=x.device).long()
        return self.p_losses(x, state, t, weights)

    def forward(self, state, *args, **kwargs):
        return self.sample(state, *args, **kwargs)

