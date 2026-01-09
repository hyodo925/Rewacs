import copy
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

ADV_MIN = -6.0
ADV_MAX = 4.0
EXP_MIN = 1e-3
EXP_MAX = 10.0
TEMP = 1

@dataclass
class Action_Weight_Dataset:
    actions_weight: np.ndarray
    observations: np.ndarray
class SWG:
    def __init__(
        self,
        model,
        replay_buffer,
        actor_optimizer,
        q_optimizer,
        value_optimizer,
        scheduler,
        batch_size,
        polyak=0.995,
        gamma=0.9,
        beta=0.3,
    ):
        self.alg_name = "SWG"
        self.model = model
        self.target = copy.deepcopy(model)
        self.replay_buffer = replay_buffer
        self.actor_optimizer = actor_optimizer
        self.q_optimizer = q_optimizer
        self.value_optimizer = value_optimizer
        self.batch_size = batch_size
        self.polyak = polyak
        self.gamma = torch.as_tensor([gamma])
        self.beta = torch.as_tensor([beta])

        self.device = model.device

        self.scheduler = scheduler

    def safe_exp(self, x):
        return torch.where(x < 50, torch.exp(x), x)

    def step_ema(self):
        if self.step < self.step_start_ema:
            return
        self.ema.update_model_average(self.ema_model, self.actor)

    def exp_w_clip(self, x, x0, mode="zero"):
        if mode == "zero":
            return torch.where(x < x0, torch.exp(x), torch.exp(x0))
        elif mode == "first":
            return torch.where(
                x < x0, torch.exp(x), torch.exp(x0) + torch.exp(x0) * (x - x0)
            )
        elif mode == "second":
            return torch.where(
                x < x0,
                torch.exp(x),
                torch.exp(x0)
                + torch.exp(x0) * (x - x0)
                + (torch.exp(x0) / 2) * ((x - x0) ** 2),
            )
        else:
            raise ValueError(f"Invalid mode: {mode}")

    def update_critic(self, data_for_logging=None):
        sample = self.replay_buffer.sample(self.batch_size)
        obs, next_obs, r_obs, next_r_obs, act, rwd, done = list(sample.values())

        with torch.no_grad():
            Q_target_1, Q_target_2 = self.target.q(
                (obs.to(self.device), r_obs.to(self.device)), act.squeeze().to(self.device)
            )
            q_values = torch.min(torch.cat((Q_target_1, Q_target_2), 1), dim=1)[0].unsqueeze(-1)
        values = self.model.value((obs.to(self.device), r_obs.to(self.device)))

        diff = q_values - values
        with torch.no_grad():  # Stop gradient for exp_diff
            exp_diff = self.exp_w_clip(diff * beta, clip, mode)
        loss_value = ((exp_diff - 1) * diff).mean()
        self.value_optimizer.zero_grad(set_to_none=True)
        loss_value.backward()
        torch.nn.utils.clip_grad_norm_(self.model.value.parameters(), max_norm=1.0)
        self.value_optimizer.step()
        lv = loss_value.data.item()

        with torch.no_grad():
            next_values = self.model.value((next_obs.to(self.device), next_r_obs.to(self.device)))
            Q_target = rwd.to(self.device) + (self.gamma * next_values) * done.to(
                self.device
            )

        Q1, Q2 = self.model.q(
            (obs.to(self.device), r_obs.to(self.device)),
            act.squeeze().to(self.device),
        )

        loss_critic = F.mse_loss(Q_target, Q1) + F.mse_loss(Q_target, Q2)
        self.q_optimizer.zero_grad(set_to_none=True)
        loss_critic.backward()
        torch.nn.utils.clip_grad_norm_(self.model.q.parameters(), max_norm=1.0)
        self.q_optimizer.step()
        lq = loss_critic.data.item()

        if data_for_logging is not None:
            data_for_logging[0].log(
                {
                    "loss/q": lq,
                    "loss/value": lv,
                },
                step=data_for_logging[1],
            )


    def update_actor(self, update_actor=False, data_for_logging=None):
        sample = self.replay_buffer.sample(self.batch_size)
        obs, next_obs, r_obs, next_r_obs, act, rwd, done = list(sample.values())

        loss, info = self.model.diffusion_model.loss(self.replay_buffer)
        loss.backward()

        self.actor_optimizer.step()
        self.actor_optimizer.zero_grad(set_to_none=True)

        self.scheduler.step()


    def update_target(self):
        for param, target_param in zip(
            self.model.q.parameters(), self.target.q.parameters()
        ):
            target_param.data.mul_(self.polyak)
            target_param.data.add_((1 - self.polyak) * param.data)

    def build_weights(
        self,  # TODO add more weights...
        q_model,
        value_model,
        critic_hyperparam: float = 0.7,
        weights_function: str = "expectile",
        norm: bool = False,
    ):
        humans_states = self.replay_buffer[:len(self.replay_buffer)]["humans_obs"]
        robot_states = self.replay_buffer[:len(self.replay_buffer)]["robot_obs"]
        actions = self.replay_buffer[:len(self.replay_buffer)]["actions"]

        humans_states_batch = np.array_split(humans_states, states.shape[0] // 256 + 1)
        robot_states_batch = np.array_split(robot_states, states.shape[0] // 256 + 1)
        actions_batch = np.array_split(actions, actions.shape[0] // 256 + 1)

        weights_list = []
        for states, actions in tqdm(zip(states_batch, actions_batch)):

            states = torch.tensor(states, dtype=torch.float32, device=self.device)
            actions = torch.tensor(actions, dtype=torch.float32, device=self.device)

            qs = q_model(action=actions, state=states)  # (B, 1)
            vs = value_model(state=states)
            adv = qs - vs

            if weights_function == "expectile":
                weight = torch.where(
                    adv > 0, critic_hyperparam, 1 - critic_hyperparam
                )  # hyperparam of critic...

            elif weights_function == "quantile":  # TODO
                pass

            elif weights_function == "linex":
                weight = torch.abs(torch.exp(critic_hyperparam * adv) - 1) / torch.abs(
                    adv
                )
                weight = torch.clamp(weight, min=ADV_MIN, max=ADV_MAX)

            elif weights_function == "exponential":
                adv = torch.clamp(adv, min=ADV_MIN, max=ADV_MAX)
                weight = torch.exp(TEMP * adv)  # in this case e**bA
                weight = torch.clamp(weight, min=EXP_MIN, max=EXP_MAX)

            elif weights_function == "advantage":
                weight = adv

            elif weights_function == "dice":
                pi_residual = adv / critic_hyperparam

                if pi_residual.dim() == 1:
                    pi_residual = pi_residual.unsqueeze(1)

                weight = torch.where(
                    pi_residual >= 0, pi_residual / 2 + 1, torch.exp(pi_residual)
                )
                weight = torch.clamp(weight, min=1e-40, max=100)

            else:
                print("Not supported weights function")

            weights_list.append(weight)

        weights_tensor = torch.cat(weights_list, dim=0)
        weights_tensor = torch.nan_to_num(weights_tensor, nan=0.0)

        if norm:
            max = torch.max(weights_tensor)
            min = torch.min(weights_tensor)
            weights_tensor = (weights_tensor - min) / (max - min)

        min_weight = torch.min(weights_tensor)
        max_weight = torch.max(weights_tensor)
        mean_weight = torch.mean(weights_tensor)
        std_weights = torch.std(weights_tensor)

        print(f"min weight: {min_weight} | max_weight: {max_weight} | mean weight: {mean_weight} | std: {std_weights}")

        assert weights_tensor.shape == (len(self.replay_buffer.next_observations), 1)

        self.weight_dataset = Action_Weight_Dataset(
            actions_weight=np.append(
                self.replay_buffer.actions, weights_tensor.cpu().numpy(), axis=-1
            ),
            observations=self.replay_buffer.observations,
        )
        del self.replay_buffer