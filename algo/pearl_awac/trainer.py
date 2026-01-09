import copy
import higher
import torch
import torch.nn.functional as F
import random
from tqdm import tqdm


class PEARLAWAC:
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
        self.alg_name = "PEARLAWAC"
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

    def zeros(self, *sizes, torch_device=None, **kwargs):
        if torch_device is None:
            torch_device = self.device
        return torch.zeros(*sizes, **kwargs, device=torch_device)

    def ones(self, *sizes, **kwargs):
        return torch.ones(*sizes, **kwargs).to(self.device)
    
    def to_tensor(self, x, device):
        if isinstance(x, np.ndarray):
            return torch.from_numpy(x).float().to(device)
        elif isinstance(x, torch.Tensor):
            return x.to(device)
        elif isinstance(x, (int, float)):  # ← ★ ここを追加！
            return torch.tensor([x], dtype=torch.float32, device=device)
        else:
            raise TypeError(f"Unsupported type: {type(x)}")

    def soft_update_from_to(self, source, target, tau):
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(
                target_param.data * (1.0 - tau) + param.data * tau
            )

    def thexp(self, x):
        # more stable version of log(1 + exp(x))
        return torch.where(x < 50, torch.exp(x), x)

    def compute_kl_div(self):
        ''' compute KL( q(z|c) || r(z) ) '''
        prior = torch.distributions.Normal(self.zeros(self.latent_dim), self.ones(self.latent_dim))
        posteriors = [torch.distributions.Normal(mu, torch.sqrt(var)) for mu, var in zip(torch.unbind(self.z_means), torch.unbind(self.z_vars))]
        kl_divs = [torch.distributions.kl.kl_divergence(post, prior) for post in posteriors]
        kl_div_sum = torch.sum(torch.stack(kl_divs))
        return kl_div_sum
    
    def _product_of_gaussians(self, mus, sigmas_squared):
        """
        mus: Tensor [K, D]  ← 各タスクで1つだけなら [1, D] になる
        sigmas_squared: Tensor [K, D]
        """
        eps = 1e-6
        sigmas_squared = torch.clamp(sigmas_squared, min=eps)
        precision = 1.0 / sigmas_squared
        mu = torch.sum(mus * precision, dim=0) / torch.sum(precision, dim=0)
        sigma_squared = 1.0 / torch.sum(precision, dim=0)
        return mu, sigma_squared  # shape: [D], [D]

    def sample_z(self):
        if self.use_ib:
            posteriors = [torch.distributions.Normal(m, torch.sqrt(s)) for m, s in zip(torch.unbind(self.z_means), torch.unbind(self.z_vars))]
            z = [d.rsample() for d in posteriors]
            self.z = torch.stack(z)
        else:
            self.z = self.z_means

    def clear_z(self, num_tasks=1):
        '''
        reset q(z|c) to the prior
        sample a new z from the prior
        '''
        #  reset distribution over z to the prior
        mu = self.zeros(num_tasks, self.latent_dim)
        var = self.ones(num_tasks, self.latent_dim)
        self.z_means = mu
        self.z_vars = var

    def infer_posterior(self, context):
        T = self.num_tasks  
        B = context.size(0) // T  

        context = context.view(T, B, -1) 
        params = self.context_encoder(context.to(self.device))  # (5, 100, latent_dim * 2)

        if self.use_ib:
            mu = params[..., :self.latent_dim]           # (5, 100, 5)
            sigma_squared = F.softplus(params[..., self.latent_dim:])  # (5, 100, 5)

            z_params = [
                self._product_of_gaussians(m, s)  # m, s: (100, 5)
                for m, s in zip(mu, sigma_squared)
            ]
            self.z_means = torch.stack([p[0] for p in z_params])  # → (5, 5)
            self.z_vars = torch.stack([p[1] for p in z_params])   # → (5, 5)
        else:
            self.z_means = torch.mean(params, dim=1)  # (5, 5)


    def sample_awac(self, indices):
        batches = [
            self.train_tasks.random_batch(idx, batch_size=self.embedding_batch_size)
            for idx in indices
        ]
        unpacked_list = [self.unpack_batch(batch, sparse_reward=False) for batch in batches]

        grouped = [[x[i] for x in unpacked_list] for i in range(len(unpacked_list[0]))]

        padded_grouped = []
        for i, t_list in enumerate(grouped):
            if len(t_list[0].shape) == 3:  # shape: [batch, agents, feat] の場合
                # max_agents = max(x.shape[1] for x in t_list)
                max_agents = self.max_ped_num
                padded_tasks = []
                for x in t_list:
                    b, a, f = x.shape
                    if a < max_agents:
                        padding = torch.zeros((b, max_agents - a, f), device=x.device)
                        x = torch.cat([x, padding], dim=1)
                    padded_tasks.append(x)
                stacked = torch.cat(padded_tasks, dim=0)  # shape: [tasks * batch, max_agents, feat]
            else:
                stacked = torch.cat(t_list, dim=0)  # 通常の [batch, feat] など
            padded_grouped.append(stacked)

        return padded_grouped

    
    def unpack_batch(self, batch, sparse_reward=False):
        prev_obs = batch['observations']  # shape: [batch, agents, feat]
        act = batch['actions']
        rwd = batch['sparse_rewards'] if sparse_reward and 'sparse_rewards' in batch else batch['rewards']
        obs = batch['next_observations']
        done = batch['terminals']
        prev_r_obs = batch['robot_obs']
        r_obs = batch['next_robot_obs']
        return [prev_obs, obs, prev_r_obs, r_obs, act, rwd, done]
    
    def sample_context(self, indices, context):
        if not hasattr(indices, '__iter__'):
            indices = [indices]

        batches = [
            context.random_batch(idx, batch_size=self.embedding_batch_size)
            for idx in indices
        ]
        context_list = [self.unpack_batch(batch, sparse_reward=self.sparse_rewards) for batch in batches]

        context_grouped = [[x[i] for x in context_list] for i in range(len(context_list[0]))]

        processed = []
        for i, t_list in enumerate(context_grouped):
            if len(t_list[0].shape) == 3:
                # max_agents = max(x.shape[1] for x in t_list)
                max_agents = self.max_ped_num
                padded_tasks = []
                for x in t_list:
                    b, a, f = x.shape
                    if a < max_agents:
                        padding = torch.zeros((b, max_agents - a, f), device=x.device)
                        x = torch.cat([x, padding], dim=1)
                    padded_tasks.append(x)
                stacked = torch.cat(padded_tasks, dim=0)  # [tasks * batch, agents, feat]
                processed.append(stacked.view(stacked.size(0), -1))  # flatten: [tasks * batch, agents * feat]
            else:
                processed.append(torch.cat(t_list, dim=0))

        min_len = min(t.shape[0] for t in processed[:-1])
        if self.use_next_obs_in_context:
            context = torch.cat([t[:min_len] for t in processed[:-1]], dim=1)
        else:
            context = torch.cat([t[:min_len] for t in processed[:-2]], dim=1)

        return context



    def train(self, train_tasks, context):
        # for task, context in zip(tasks, context):
            # indices = np.random.choice(train_tasks, self.meta_batch)
            self.train_tasks = train_tasks
            indices = list(range(self.num_tasks))
            mb_size = self.embedding_mini_batch_size
            self.num_update = self.embedding_batch_size // mb_size

            # sample context batch
            context_batch = self.sample_context(indices, context=context)

            # zero out context and hidden encoder state
            self.clear_z(num_tasks=len(indices))

            # do this in a loop so we can truncate backprop in the recurrent encoder
            for i in range(self.num_update):
                context = context_batch[i * mb_size: i * mb_size + mb_size, :]
                self.step(indices, context)
                # stop backprop
                self.z = self.z.detach()
                if i == self.num_update:
                    break
    
    def run_inference(self, obs, r_obs, context):
        self.infer_posterior(context)
        self.sample_z()

        T_B = obs.shape[0]        # e.g., 500
        T = self.z.size(0)
        B = T_B // T

        # task_z の作成: 各 z を B 回繰り返して T*B 行に
        task_z = [z.repeat(B, 1) for z in self.z]  # z: (latent_dim,)
        task_z = torch.cat(task_z, dim=0)          # → (T * B, latent_dim)

        # r_obs の shape を (T*B, 1, D_r_obs) に調整
        if r_obs.dim() == 2:
            r_obs = r_obs.unsqueeze(1)  # (T*B, D) → (T*B, 1, D)

        # 確認用プリント
        # print(f"obs.shape     = {obs.shape}")     # (T * B, A, D_obs)
        # print(f"r_obs.shape   = {r_obs.shape}")   # (T * B, 1, D_r_obs)
        # print(f"task_z.shape  = {task_z.shape}")  # (T * B, latent_dim)

        policy_outputs = self.policy.sample(
            (obs.to(self.device), r_obs.to(self.device)),
            z=task_z.detach(),
            return_pretanh_value=True
        )

        return policy_outputs, task_z


    
    def step(self, indices, context):
        gamma_bar = pow(self.gamma, self.time_step * self.v_pref)
        prev_obs, obs, prev_r_obs, r_obs, act, rwd, done = self.sample_awac(indices)
        """
        Policy and Alpha Loss
        """
        policy_outputs, task_z = self.run_inference(prev_obs.to(self.device), prev_r_obs.to(self.device), context.to(self.device))
        new_actions, next_log_prob, policy_mean, policy_log_std, pre_tanh_value = policy_outputs

        task_z_detached = task_z.detach()

        self.context_encoder_optimizer.zero_grad()
        if self.use_information_bottleneck:
            kl_div = self.compute_kl_div()
            kl_loss = self.kl_lambda * kl_div
            kl_loss.backward(retain_graph=True)

        with torch.no_grad():
            next_act_target, next_log_prob, _, _= self.policy.sample(
                (
                    obs.to(self.device),
                    r_obs.reshape(-1, 1, self.r_obs_dim).to(self.device),
                ),
                z=task_z_detached
            )

            Q_target_1, Q_target_2 = self.target(
                (obs.to(self.device), r_obs.to(self.device)), 
                act=next_act_target,
                z=task_z_detached
            )
            Q_target_min = torch.min(
                torch.cat((Q_target_1, Q_target_2), 1), dim=1
            )[0].unsqueeze(-1)

            Q_target = rwd.to(self.device) + (
                gamma_bar * Q_target_min
            ) * done.to(self.device)

        Q1, Q2 = self.critic(
            (prev_obs.to(self.device), prev_r_obs.to(self.device)),
            act=act.squeeze().to(self.device),
            z=task_z_detached
        )

        loss_value = F.mse_loss(Q_target, Q1) + F.mse_loss(Q_target, Q2)
        self.critic_optimizer.zero_grad()
        loss_value.backward()
        self.critic_optimizer.step()
        self.context_encoder_optimizer.step()
        
        kl_total_norm = 0
        for p in self.context_encoder.parameters():
            if p.grad is not None:
                kl_total_norm += p.grad.data.norm(2).item()
        critic_total_norm = 0
        for p in self.critic.parameters():
            if p.grad is not None:
                critic_total_norm += p.grad.data.norm(2).item() 

        with torch.no_grad():
            qw_ref = torch.min(torch.cat((Q1, Q2), 1), dim=1)[0].reshape(
                (-1, 1)
            )

            v_act1, v_act2 = self.critic(
                (prev_obs.to(self.device), prev_r_obs.to(self.device)),
                act=new_actions.detach(),
                z=task_z_detached
            )

            qw_gen = torch.min(torch.cat((v_act1, v_act2), 1), dim=1)[
                0
            ].reshape((-1, 1))

            adv = torch.max(torch.zeros_like(qw_ref), qw_ref - qw_gen)
            weights = self.thexp(adv / self.beta)

        loss_act = -(
            self.policy.get_log_prob(
                (
                    prev_obs.to(self.device),
                    prev_r_obs.reshape(-1, 1, self.r_obs_dim).to(
                        self.device
                    ),
                ),
                act.squeeze().to(self.device),
                z=task_z_detached
            )
            * weights
        ).mean()
        self.policy_optimizer.zero_grad()
        loss_act.backward()
        self.policy_optimizer.step()
        actor_total_norm = 0
        for p in self.policy.parameters():
            if p.grad is not None:
                actor_total_norm += p.grad.data.norm(2).item()
        self.soft_update_from_to(
            self.critic, self.target, self.soft_target_tau
        )
        if self.use_wandb:
            wandb.log({
            "loss_act": loss_act.data.item(),
            "loss_value": loss_value.data.item(),
            "loss_kl": kl_loss.data.item(),
            "critic grad norm": critic_total_norm,
            "actor grad norm": actor_total_norm,
            "kl grad norm": kl_total_norm,
        })
        # self._n_train_steps_total += 1
        
    def pack_context(
        self, 
        prev_obs, obs,               # human 観測 (N, D)
        prev_r_obs, r_obs,           # robot 観測 (D,) または (1, D)
        action, reward, done,        # すべてスカラー or ベクトル (D,)
        max_peds=5                   # 最大人数（学習時と一致させる）
    ):
        # human の padding or truncate
        def pad_or_trunc(x, max_len):
            N, D = x.shape
            if N < max_len:
                padding = torch.zeros((max_len - N, D), device=self.device)
                x = torch.cat([x, padding], dim=0)
            else:
                x = x[:max_len]
            return x.flatten()

        # 必ず Tensor に変換
        prev_obs = self.to_tensor(prev_obs, self.device)        # shape: (N, D)
        obs = self.to_tensor(obs, self.device)
        prev_r_obs = self.to_tensor(prev_r_obs, self.device)    # shape: (D,) or (1, D)
        r_obs = self.to_tensor(r_obs, self.device)
        action = self.to_tensor(action, self.device)
        reward = self.to_tensor(reward, self.device)
        done = self.to_tensor(done, self.device)

        # reshape robot obs to (1, D) if needed
        if prev_r_obs.dim() == 1:
            prev_r_obs = prev_r_obs.unsqueeze(0)
        if r_obs.dim() == 1:
            r_obs = r_obs.unsqueeze(0)

        context_vec = torch.cat([
            pad_or_trunc(prev_obs, max_peds),         # [max_peds * D]
            pad_or_trunc(obs, max_peds),              # [max_peds * D]
            prev_r_obs.flatten(),                     # [r_obs_dim]
            r_obs.flatten(),                          # [r_obs_dim]
            action.flatten(),                         # [act_dim]
            reward.flatten(),                         # [1]
            # done.flatten()                            # [1]
        ])

        return context_vec

    def update_target(self):
        for param, target_param in zip(
            self.model.parameters(), self.target.parameters()
        ):
            target_param.data.mul_(self.polyak)
            target_param.data.add_((1 - self.polyak) * param.data)
