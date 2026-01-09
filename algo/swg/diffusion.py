from collections import namedtuple
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from tqdm import tqdm

from .helpers import (
    cosine_beta_schedule,
    vp_beta_schedule,
    linear_beta_schedule,
    extract,
)

Sample = namedtuple("Sample", "sample chains")


class GaussianDiffusion(nn.Module):
    """
    Base Gaussian diffusion model
    """

    def __init__(self, model, data_dim=3, schedule="vp", n_timesteps=5):
        super().__init__()

        self.model = model
        self.data_dim = data_dim

        schedulers = {
            "vp": vp_beta_schedule,
            "cosine": cosine_beta_schedule,
            "linear": linear_beta_schedule,
        }

        if schedule in schedulers:
            betas = schedulers[schedule](n_timesteps)
        else:
            raise ValueError(
                f"Unknown schedule: {schedule}. Available options are 'vp', 'cosine', or 'linear'."
            )

        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, axis=0)
        alphas_cumprod_prev = torch.cat([torch.ones(1), alphas_cumprod[:-1]])

        self.n_timesteps = int(n_timesteps)

        register_buffer = lambda name, val: self.register_buffer(
            name, val.to(torch.float32)
        )  # helper function to register buffer from float64 to float32

        register_buffer("betas", betas)
        register_buffer("alphas_cumprod", alphas_cumprod)
        register_buffer("alphas_cumprod_prev", alphas_cumprod_prev)

        # calculations for diffusion q(x_t | x_{t-1}) and others
        register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        register_buffer(
            "sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - alphas_cumprod)
        )
        register_buffer("log_one_minus_alphas_cumprod", torch.log(1.0 - alphas_cumprod))
        register_buffer("sqrt_recip_alphas_cumprod", torch.sqrt(1.0 / alphas_cumprod))
        register_buffer(
            "sqrt_recipm1_alphas_cumprod", torch.sqrt(1.0 / alphas_cumprod - 1)
        )

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = (
            betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        )
        register_buffer("posterior_variance", posterior_variance)

        ## log calculation clipped because the posterior variance
        ## is 0 at the beginning of the diffusion chain
        register_buffer(
            "posterior_log_variance_clipped",
            torch.log(torch.clamp(posterior_variance, min=1e-20)),
        )
        register_buffer(
            "posterior_mean_coef1",
            betas * np.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod),
        )
        register_buffer(
            "posterior_mean_coef2",
            (1.0 - alphas_cumprod_prev) * np.sqrt(alphas) / (1.0 - alphas_cumprod),
        )

    # ------------------------------------------ sampling ------------------------------------------#

    def predict_start_from_noise(self, x_t, t, noise):
        """ """
        return (
            extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
            - extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
        )

    def q_posterior(self, x_start, x_t, t):
        posterior_mean = (
            extract(self.posterior_mean_coef1, t, x_t.shape) * x_start
            + extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract(
            self.posterior_log_variance_clipped, t, x_t.shape
        )
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    @torch.inference_mode()
    def p_mean_variance(self, x, state, t):

        epsilon = self.model(x=x, state=state, time=t, training=False)
        x_recon = self.predict_start_from_noise(x, t=t, noise=epsilon)

        if self.clip_denoised:
            x_recon.clamp_(-1.0, 1.0)

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(
            x_start=x_recon, x_t=x, t=t
        )
        return model_mean, posterior_variance, posterior_log_variance

    def p_sample(self, x, state, t):
        b, *_, device = *x.shape, x.device

        batched_time = torch.full((b,), t, device=device, dtype=torch.long)

        model_mean, _, model_log_variance = self.p_mean_variance(
            x=x, state=state, t=batched_time
        )

        if t > 0:
            noise = torch.randn_like(x) * self.temperature
        else:
            noise = 0

        x_pred = model_mean + (0.5 * model_log_variance).exp() * noise

        return x_pred

    def p_sample_loop(self, state, shape):
        """
        Classical DDPM (check this) sampling algorithm

            Parameters:
                state
                shape,

            Returns:
            sample

        """

        device = self.betas.device

        x = torch.randn(shape, device=device) * self.temperature

        chain = [x] if self.return_chain else None

        for t in tqdm(
            reversed(range(0, self.n_timesteps)),
            desc="sampling loop time step",
            total=self.n_timesteps,
            disable=self.disable_progess_bar,
        ):
            x = self.p_sample(x=x, state=state, t=t)
            if self.return_chain:
                chain.append(x)

        if self.return_chain:
            chain = torch.stack(chain, dim=1)

        if self.clip_denoised:
            x.clamp_(-1.0, 1.0)

        return Sample(x, chain)

    def forward(self, state):
        """ """
        batch_size = state.shape[0]

        return self.p_sample_loop(state, shape=(batch_size, self.data_dim))

    def setup_sampling(
        self,
        clip_denoised=True,
        temperature=0.5,
        disable_progess_bar=False,
        return_chain=False,
        **kwargs,
    ):

        self.temperature = temperature
        self.clip_denoised = clip_denoised
        self.disable_progess_bar = disable_progess_bar
        self.return_chain = return_chain

    # ------------------------------------------ training ------------------------------------------#

    def q_sample(self, x_start, t, noise):

        sample = (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
            + extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )

        return sample

    def p_losses(self, x_start, state, t):

        noise = torch.randn_like(x_start)

        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)

        pred_epsilon = self.model(x_noisy, state, t, training=True)

        assert noise.shape == pred_epsilon.shape

        loss = F.mse_loss(pred_epsilon, noise, reduction="none").mean()

        return loss, {"loss": loss}

    def loss(self, x, state):

        batch_size = len(x)
        t = torch.randint(0, self.n_timesteps, (batch_size,), device=x.device).long()

        return self.p_losses(x, state, t)


class SelfWeighted_GaussianDiffusion(GaussianDiffusion):
    """
    Base Gaussian diffusion model
    """

    def __init__(self, model, data_dim=3, schedule="vp", n_timesteps=5):
        super().__init__(
            model=model, data_dim=data_dim, schedule=schedule, n_timesteps=n_timesteps
        )

    @torch.enable_grad()
    def p_mean_variance(self, x, state, t):

        if self.guidance_scale > 0:
            x.requires_grad_(True)

            if x.grad is not None:
                x.grad.zero_()

            epsilon = self.model(x=x, state=state, time=t, training=False)

            swg_term = (
                x[:, -1]
                - extract(self.sqrt_one_minus_alphas_cumprod, t, x[:, -1].shape)
                * epsilon[:, -1]
            )
            swg_term = torch.log(swg_term)
            swg_term_grad = torch.autograd.grad(swg_term.sum(), x, retain_graph=False)[0]
            sqrt_one_minus_alphas_cumprod = extract(
                self.sqrt_one_minus_alphas_cumprod, t, x.shape
            )
            q_guidance_term = -sqrt_one_minus_alphas_cumprod * (swg_term_grad)

            q_guidance_term = q_guidance_term.nan_to_num(nan=0.0)

            epsilon = epsilon + self.guidance_scale * q_guidance_term.detach()

        else:
            epsilon = self.model(x=x, state=state, time=t, training=False)

        with torch.no_grad():
            x_recon = self.predict_start_from_noise(x, t=t, noise=epsilon)

            if self.clip_denoised:
                x_recon[:, :-1].clamp_(-1.0, 1.0)
                x_recon[:, -1].clamp_(self.min_weight_clip, self.max_weight_clip)

            model_mean, posterior_variance, posterior_log_variance = self.q_posterior(
                x_start=x_recon, x_t=x, t=t
            )
        return model_mean, posterior_variance, posterior_log_variance

    def p_sample_loop(self, state, shape):
        """
        DDPM sampling algorithm

            Parameters:
                state
                shape,

            Returns:
            sample

        """

        device = self.betas.device

        x = torch.randn(shape, device=device) * self.temperature

        chain = [x] if self.return_chain else None

        for t in tqdm(
            reversed(range(0, self.n_timesteps)),
            desc="sampling loop time step",
            total=self.n_timesteps,
            disable=self.disable_progess_bar,
        ):
            x = self.p_sample(x=x, state=state, t=t)
            if self.return_chain:
                chain.append(x)

        if self.return_chain:
            chain = torch.stack(chain, dim=1)

        if self.clip_denoised:
            x[:, :-1].clamp_(-1.0, 1.0)
            x[:, -1].clamp_(self.min_weight_clip, self.max_weight_clip)

        return Sample(x, chain)

    def setup_sampling(
        self,
        guidance_scale=1,
        clip_denoised=True,
        temperature=0.5,
        disable_progess_bar=False,
        return_chain=False,
        min_weight_clip=0.0001,
        max_weight_clip=1.0,
    ):

        self.guidance_scale = guidance_scale
        self.temperature = temperature
        self.clip_denoised = clip_denoised
        self.disable_progess_bar = disable_progess_bar
        self.return_chain = return_chain
        self.max_weight_clip = max_weight_clip
        self.min_weight_clip = min_weight_clip