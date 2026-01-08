# from utils.buffers import OffPolicyTuple
from time import time

import numpy as np
import torch
from numpy.lib.function_base import average
from torch._C import dtype
from tqdm import tqdm

from rewacs.envs.utils.info import *
from rewacs.envs.utils.state import JointState
from utils.env import convert_action
from utils.trajectory import Trajectory


class ExploerCrowdSim:
    def __init__(
        self,
        env,
        total_ep,
        max_traj_length,
        min_traj_length=0,
        obs_dim=0,
        act_dim=0,
        state_dim=0,
        r_obs_dim=0,
        discount=0.9,
        policy=None,
        render=False,
        transfunc=None,
    ):
        super().__init__()

        self.env = env
        self.policy = policy
        self.total_ep = total_ep
        self.max_traj_length = max_traj_length
        self.min_traj_length = min_traj_length
        # self.num_trajs = num_trajs
        # self.num_samples = num_samples
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.state_dim = state_dim
        self.r_obs_dim = r_obs_dim
        self.discount = discount
        self.render = render
        self.transfunc = transfunc

    def get_r_state(self):
        r_state = self.env.robot.get_full_state()
        r_pos = np.array(r_state.position)
        r_theta = np.array(r_state.theta)
        goal = np.array(r_state.goal_position)
        diff = goal - r_pos
        dist = np.linalg.norm(diff)
        rot = np.arctan2(diff[1], diff[0])
        if self.env.robot.kinematics == "unicycle":
            theta = r_theta - rot
        else:
            # set theta to be zero since it's not used
            theta = 0

        return np.array([dist, theta]), np.array([r_pos[0], r_pos[1], r_theta])

    def exploration(self, random_p_num=False, p_range=(1, 5), render=False):
        """
        Generate trajectories of length up max_traj_length each.

        Returns:
        A list of trajectories (See models.Trajectory).

        Additional parameters:
        - model: An object that implements models.FilteringModel interface.
            Used to track the state. If None, an ObservableModel is used,
            which returns the current observation.
        - policy: An object that implements policies.Policy interface.
            Used to provide actions.
        - render: Whether to render generated trajectories in real-time.
            This calls 'render' method which needs ot be implemented.
        - num_trajs: Number of trajectories to return.
        - num_samples: Total number of samples in generated trajectories.

        Must set num_trajs or num_samples (but not both) to a positive number.
        """
        trajs = []

        # if (self.num_samples > 0) == (self.num_trajs > 0):
        #     raise ValueError('Must specify exactly one of num_trajs and num_samples')

        done_all = False
        d_a = self.act_dim
        d_ro = self.r_obs_dim
        i_sample = 0
        tic = time()

        for _ in tqdm(range(self.total_ep), desc="Preliminary Exploration"):
            if random_p_num:
                p_num = np.random.randint(*p_range)
                self.env.set_human_num(p_num)
                d_o = self.obs_dim * p_num
            else:
                d_o = self.obs_dim * self.env.human_num
            obs = torch.empty((self.max_traj_length, d_o), dtype=torch.float32)
            act = torch.empty((self.max_traj_length, d_a), dtype=torch.float32)
            r_obs = torch.empty((self.max_traj_length, d_ro), dtype=torch.float32)
            # next_obs = torch.empty((self.max_traj_length, d_o), dtype=torch.float32)
            # next_r_obs = torch.empty((self.max_traj_length, d_ro), dtype=torch.float32)
            # r_pos = torch.empty((self.max_traj_length, 3), dtype=torch.float32)
            rwd = torch.empty((self.max_traj_length, 1), dtype=torch.float32)
            v = torch.empty((self.max_traj_length, 1), dtype=torch.float32)
            # Make a reset for each trajectory
            if self.policy != None:
                self.policy.reset()
            o_r = self.env.reset("train")
            # o = self.transfunc(o_r)
            joint_obs = JointState(self.env.robot.get_full_state(), o_r)
            robot_obs, human_obs = self.transfunc(joint_obs)
            o0 = torch.clone(human_obs.reshape(-1))
            r_o0 = torch.clone(robot_obs)

            for j in range(self.max_traj_length):
                if render:
                    self.env.render()

                if self.policy != None:
                    a = self.policy(o_r)
                else:
                    a = self.env.robot.act(o_r)
                # obs[j, :] = torch.as_tensor(human_obs.reshape(-1), dtype=torch.float32)
                # r_obs[j, :] = torch.as_tensor(robot_obs, dtype=torch.float32)
                # a = np.clip(np.array(a) + torch.randn_like(torch.as_tensor(a, dtype=torch.float32)).cpu().data.numpy(), -1, 1)
                # a = convert_action(a)
                o_r, r, done, info = self.env.step(a)
                # o = self.transfunc(o_r)
                joint_obs = JointState(self.env.robot.get_full_state(), o_r)
                robot_obs, human_obs = self.transfunc(joint_obs)
                act[j, :] = torch.as_tensor(a, dtype=torch.float32)
                obs[j, :] = torch.as_tensor(human_obs.reshape(-1), dtype=torch.float32)
                r_obs[j, :] = torch.as_tensor(robot_obs, dtype=torch.float32)
                # next_obs[j, :] = torch.as_tensor(human_obs.reshape(-1), dtype=torch.float32)
                # next_r_obs[j, :] = torch.as_tensor(robot_obs, dtype=torch.float32)
                # r_pos[j, :] = torch.as_tensor(rp, dtype=torch.float32)
                rwd[j, :] = torch.as_tensor(r, dtype=torch.float32)

                if done:
                    break

            j += 1
            drop_traj = False
            test = torch.empty((self.max_traj_length, 1), dtype=torch.float32)
            tt = rwd[:j, :]
            for ii in range(j):
                v[ii, :] = sum(
                    [
                        pow(
                            self.discount,
                            max(t - ii, 0)
                            * self.env.robot.time_step
                            * self.env.robot.v_pref,
                        )
                        * reward
                        * (1 if t >= ii else 0)
                        for t, reward in enumerate(rwd[:j, :])
                    ]
                )

                # test[ii, :] = sum([pow(self.discount, (t - ii) * self.env.robot.time_step * self.env.robot.v_pref) * reward *
                #              (1 if t >= ii else 0) for t, reward in enumerate(rwd[:j, :])])

                # v[ii, :] = sum([pow(self.discount, max(t - ii, 0)* self.env.robot.time_step * self.env.robot.v_pref) * reward
                #             * (1 if t >= ii else 0) for t, reward in enumerate(rwd[:j, :])])
            if j >= self.min_traj_length:
                # Check if we need to truncate trajectory to maintain num_samples
                # if self.num_samples > 0 and i_sample + j >= self.num_samples:
                #     j -= (i_sample + j - self.num_samples)
                #     done_all = True
                # # TODO: remove this will never happen because of outer if?
                # if j < self.min_traj_length:
                #     # Last trajectory is too short. Ignore it.
                #     drop_traj = True

                # if not drop_traj:
                # if isinstance(info, ReachGoal) or isinstance(info, Collision):
                if isinstance(info, ReachGoal) or isinstance(info, Collision):
                    i_sample += j
                    new_traj = Trajectory(
                        obs=obs[:j, :],
                        act=act[:j, :],
                        r_obs=r_obs[:j, :],
                        # r_pos=r_pos[:j, :],
                        # next_obs=next_obs,
                        # next_r_obs=next_r_obs,
                        obs0=o0,
                        r_obs0=r_o0,
                        rwd=rwd[:j, :],
                        v=v[:j, :],
                    )

                    trajs.append(new_traj)

                    # if self.num_trajs > 0 and len(trajs) == self.num_trajs:
                    #     done_all = True
        print("Gathering trajectories took:", time() - tic)
        # col_trajs = [(t.obs, t.act, t.r_obs, t.obs0, t.rwd, t.v) for t in trajs]
        # X_obs_rnd = [c[0] for c in col_trajs]
        # X_act_rnd = [c[1] for c in col_trajs]
        # r_obs = [c[2] for c in col_trajs]
        # obs0 = [c[3] for c in col_trajs]
        # rwd = [c[4] for c in col_trajs]
        # v = [c[5] for c in col_trajs]
        # obs_trajs = X_obs_rnd
        # act_trajs = X_act_rnd
        # torch.cuda.empty_cache()
        # return obs_trajs, act_trajs, r_obs, obs0, rwd, v, trajs
        return trajs

    def exploration_k_ep(
        self,
        buffer,
        model=None,
        k=1,
        epsilon=0.5,
        render=False,
        pbar=None,
        buffer_type="Normal",
        random_p_num=False,
        p_range=(1, 5),
    ):
        trajs = []
        d_a = self.act_dim
        sum_rewards = 0
        sum_cdrs = 0
        sum_returns = 0
        ###############################
        success_times = []
        collision_times = []
        timeout_times = []
        success = 0
        collision = 0
        timeout = 0
        too_close = 0
        min_dist = []
        collision_cases = []
        timeout_cases = []
        ###############################
        for _ in range(k):
            if random_p_num:
                p_num = np.random.randint(*p_range)
                self.env.set_human_num(p_num)
                d_o = self.obs_dim * p_num
            else:
                d_o = self.obs_dim * self.env.human_num
            d_ro = self.r_obs_dim
            obs = torch.empty((self.max_traj_length, d_o), dtype=torch.float32)
            act = torch.empty((self.max_traj_length, d_a), dtype=torch.float32)
            r_obs = torch.empty((self.max_traj_length, d_ro), dtype=torch.float32)
            # next_obs = torch.empty((self.max_traj_length, d_o), dtype=torch.float32)
            # next_r_obs = torch.empty((self.max_traj_length, d_ro), dtype=torch.float32)
            # r_pos = torch.empty((self.max_traj_length, 3), dtype=torch.float32)
            rwd = torch.empty((self.max_traj_length, 1), dtype=torch.float32)
            v = torch.empty((self.max_traj_length, 1), dtype=torch.float32)
            # Make a reset for each trajectory
            if self.policy != None:
                self.policy.reset()
            o_r = self.env.reset("train")
            joint_obs = JointState(self.env.robot.get_full_state(), o_r)
            robot_obs, human_obs = self.transfunc(joint_obs)
            o0 = torch.clone(human_obs.reshape(-1))
            r_o0 = torch.clone(robot_obs)

            for j in range(self.max_traj_length):
                ro, rp = self.get_r_state()
                if model != None:
                    # integrated_state = psr.get_integrated_state(state.detach(), torch.as_tensor(o).reshape(1,-1).to(psr.device), torch.as_tensor(ro, dtype=torch.float32).to(psr.device))
                    # integrated_states.append(integrated_state)
                    if model.discrete:
                        e = np.random.rand()
                        if e < epsilon:
                            a = model.generate_random_action()
                            # a = psr.optimistic_exploration(state.squeeze(), torch.as_tensor(o, dtype=torch.float32).to(psr.device), torch.as_tensor(rp, dtype=torch.float32).to(psr.device), epsilon=epsilon).cpu().data.numpy()
                            # a = psr.generate_random_action_binomial(state.squeeze(), torch.as_tensor(o, dtype=torch.float32).to(psr.device), torch.as_tensor(rp, dtype=torch.float32).to(psr.device)).cpu().data.numpy()
                            # a = self.env.robot.act(o_r)
                            # a = np.clip(np.array(a) + torch.randn_like(torch.as_tensor(a, dtype=torch.float32)).cpu().data.numpy(), -1, 1)
                        else:
                            # a = psr.generate_action_sb(state.squeeze(), human_obs.flatten().to(psr.device), robot_obs.flatten().to(psr.device)).cpu().data.numpy()
                            a = (
                                model.generate_action_mcts(
                                    human_obs.flatten().to(model.device),
                                    robot_obs.flatten().to(model.device),
                                )
                                .cpu()
                                .data.numpy()
                            )
                            # a = convert_action(a)
                    else:
                        integrated_state = model.get_integrated_state(
                            human_obs.reshape(1, -1).to(model.device),
                            robot_obs.to(model.device),
                        )
                        # a = psr.generate_action(integrated_state).cpu().data.numpy()[0]
                        a = (
                            model.generate_action(integrated_state)
                            .cpu()
                            .data.numpy()[0]
                            + np.random.normal(0, 1 * 0.2, size=2)
                        ).clip(-1, 1)
                    # a = psr.generate_random_action_binomial(state.squeeze(), torch.as_tensor(o, dtype=torch.float32).to(psr.device), torch.as_tensor(rp, dtype=torch.float32).to(psr.device)).cpu().data.numpy()
                    # a = self.env.robot.act(o_r)
                    # a = psr.optimistic_exploration(state.squeeze(), torch.as_tensor(o, dtype=torch.float32).to(psr.device), torch.as_tensor(rp, dtype=torch.float32).to(psr.device), epsilon=epsilon).cpu().data.numpy()
                    # a = psr.generate_noised_action(state.squeeze(), torch.as_tensor(o, dtype=torch.float32).to(psr.device), torch.as_tensor(rp, dtype=torch.float32).to(psr.device))
                    # a = convert_action(a)
                else:
                    if self.policy != None:
                        a = self.policy(o_r)
                    else:
                        a = self.env.robot.act(o_r)
                # obs[j, :] = torch.as_tensor(human_obs.reshape(-1), dtype=torch.float32)
                # r_obs[j, :] = torch.as_tensor(robot_obs, dtype=torch.float32)
                a = convert_action(a)
                o_r, r, done, info = self.env.step(a)

                joint_obs = JointState(self.env.robot.get_full_state(), o_r)
                robot_obs, human_obs = self.transfunc(joint_obs)

                act[j, :] = torch.as_tensor(a, dtype=torch.float32)
                obs[j, :] = torch.as_tensor(human_obs.reshape(-1), dtype=torch.float32)
                r_obs[j, :] = torch.as_tensor(robot_obs, dtype=torch.float32)
                # next_obs[j, :] = torch.as_tensor(human_obs.reshape(-1), dtype=torch.float32)
                # next_r_obs[j, :] = torch.as_tensor(robot_obs, dtype=torch.float32)
                # r_pos[j, :] = torch.as_tensor(rp, dtype=torch.float32)
                rwd[j, :] = torch.as_tensor(r, dtype=torch.float32)
                if isinstance(info, Discomfort):
                    too_close += 1
                    min_dist.append(info.min_dist)

                if done:
                    if render:
                        self.env.render()
                    sum_rewards += r
                    break

            if isinstance(info, ReachGoal):
                success += 1
                success_times.append(self.env.global_time)
            elif isinstance(info, Collision):
                collision += 1
                collision_cases.append(k)
                collision_times.append(self.env.global_time)
            elif isinstance(info, Timeout):
                timeout += 1
                timeout_cases.append(k)
                timeout_times.append(self.env.time_limit)

            j += 1
            rewards = rwd[:j, :]

            sum_cdrs += sum(
                [
                    pow(
                        self.discount,
                        t * self.env.robot.time_step * self.env.robot.v_pref,
                    )
                    * reward
                    for t, reward in enumerate(rewards)
                ]
            )
            step_returns = []
            for step in range(len(rewards)):
                step_return = sum(
                    [
                        pow(
                            self.discount,
                            t * self.env.robot.time_step * self.env.robot.v_pref,
                        )
                        * reward
                        for t, reward in enumerate(rewards[step:])
                    ]
                )
                step_returns.append(step_return.cpu().data.numpy())
            sum_returns += average(step_returns)
            for ii in range(j):
                v[ii, :] = sum(
                    [
                        pow(
                            self.discount,
                            max(t - ii, 0)
                            * self.env.robot.time_step
                            * self.env.robot.v_pref,
                        )
                        * reward
                        * (1 if t >= ii else 0)
                        for t, reward in enumerate(rewards)
                    ]
                )

            if isinstance(info, ReachGoal) or isinstance(info, Collision):
                new_traj = Trajectory(
                    obs=obs[:j, :],
                    act=act[:j, :],
                    r_obs=r_obs[:j, :],
                    # r_pos=r_pos[:j, :],
                    # next_obs=next_obs,
                    # next_r_obs=next_r_obs,
                    obs0=o0,
                    r_obs0=r_o0,
                    rwd=rwd[:j, :],
                    v=v[:j, :],
                )

                # buffer.push([new_traj])
                buffer.add([new_traj])
        if pbar:
            pbar.set_postfix(Reward=str(sum_rewards / k))

        avg_reward = sum_rewards / k
        avg_cdr = sum_cdrs / k
        avg_return = sum_returns / k
        success_rate = success / k
        collision_rate = collision / k
        timeout_rate = timeout / k
        assert success + collision + timeout == k
        avg_nav_time = (
            sum(success_times) / len(success_times)
            if success_times
            else self.env.time_limit
        )

        return (
            avg_reward,
            avg_cdr,
            avg_return,
            success_rate,
            collision_rate,
            timeout_rate,
            avg_nav_time,
        )

    def exploration_rm_k_ep_orca(
        self, memory, k=1, render=False, random_p_num=False, p_range=(1, 5)
    ):
        trajs = []
        d_a = self.act_dim
        sum_rewards = 0
        sum_cdrs = 0
        sum_returns = 0
        ###############################
        success_times = []
        collision_times = []
        timeout_times = []
        success = 0
        collision = 0
        timeout = 0
        too_close = 0
        min_dist = []
        collision_cases = []
        timeout_cases = []
        ###############################
        for _ in tqdm(range(k), desc="Preliminary Exploration"):
            if random_p_num:
                p_num = np.random.randint(*p_range)
                self.env.set_human_num(p_num)
                d_o = self.obs_dim * p_num
            else:
                d_o = self.obs_dim * self.env.human_num
            d_ro = self.r_obs_dim
            d_s = self.state_dim * self.env.human_num

            obs = torch.empty((self.max_traj_length, d_o), dtype=torch.float32)
            prev_obs = torch.empty((self.max_traj_length, d_o), dtype=torch.float32)
            act = torch.empty((self.max_traj_length, d_a), dtype=torch.float32)
            r_obs = torch.empty((self.max_traj_length, d_ro), dtype=torch.float32)
            prev_r_obs = torch.empty((self.max_traj_length, d_ro), dtype=torch.float32)

            rwd = torch.empty((self.max_traj_length, 1), dtype=torch.float32)
            v = torch.empty((self.max_traj_length, 1), dtype=torch.float32)
            is_done = torch.empty((self.max_traj_length, 1), dtype=torch.float32)
            # Make a reset for each trajectory
            if self.policy != None:
                self.policy.reset()

            o_r = self.env.reset("train")
            joint_obs = JointState(self.env.robot.get_full_state(), o_r)
            robot_obs, human_obs = self.transfunc(joint_obs)
            o0 = torch.clone(human_obs.reshape(-1))
            r_o0 = torch.clone(robot_obs)

            for j in range(self.max_traj_length):
                a = self.env.robot.act(o_r)

                prev_obs[j, :] = torch.as_tensor(
                    human_obs.reshape(-1), dtype=torch.float32
                )
                prev_r_obs[j, :] = torch.as_tensor(robot_obs, dtype=torch.float32)
                o_r, r, done, info = self.env.step(a)

                joint_obs = JointState(self.env.robot.get_full_state(), o_r)
                robot_obs, human_obs = self.transfunc(joint_obs)

                act[j, :] = torch.as_tensor(a, dtype=torch.float32)
                obs[j, :] = torch.as_tensor(human_obs.reshape(-1), dtype=torch.float32)
                r_obs[j, :] = torch.as_tensor(robot_obs, dtype=torch.float32)

                rwd[j, :] = torch.as_tensor(r, dtype=torch.float32)
                is_done[j, :] = torch.as_tensor(1 - int(done), dtype=torch.float32)

                if isinstance(info, Discomfort):
                    too_close += 1
                    min_dist.append(info.min_dist)

                if done:
                    if render:
                        self.env.render()
                    sum_rewards += r
                    break

            if isinstance(info, ReachGoal):
                success += 1
                success_times.append(self.env.global_time)
            elif isinstance(info, Collision):
                collision += 1
                collision_cases.append(k)
                collision_times.append(self.env.global_time)
            elif isinstance(info, Timeout):
                timeout += 1
                timeout_cases.append(k)
                timeout_times.append(self.env.time_limit)

            j += 1
            rewards = rwd[:j, :]
            sum_cdrs += sum(
                [
                    pow(
                        self.discount,
                        t * self.env.robot.time_step * self.env.robot.v_pref,
                    )
                    * reward
                    for t, reward in enumerate(rewards)
                ]
            )
            step_returns = []
            for step in range(len(rewards)):
                step_return = sum(
                    [
                        pow(
                            self.discount,
                            t * self.env.robot.time_step * self.env.robot.v_pref,
                        )
                        * reward
                        for t, reward in enumerate(rewards[step:])
                    ]
                )
                step_returns.append(step_return.cpu().data.numpy())
            sum_returns += average(step_returns)

            # gamma_bar = pow(
            #     self.discount, self.env.robot.time_step * self.env.robot.v_pref
            # )
            # for ii in range(j):
            #     v_ = pow(
            #         gamma_bar,
            #         len(rewards) - ii + 1,
            #     ) * (rewards[-1] / (1 - gamma_bar))

            #     v[ii, :] = (
            #         sum(
            #             [
            #                 pow(
            #                     self.discount,
            #                     max(t - ii, 0)
            #                     * self.env.robot.time_step
            #                     * self.env.robot.v_pref,
            #                 )
            #                 * reward
            #                 * (1 if t >= ii else 0)
            #                 for t, reward in enumerate(rewards)
            #             ]
            #         )
            #         + v_
            #     )

            for ii in range(j):
                v[ii, :] = sum(
                    [
                        pow(
                            self.discount,
                            max(t - ii, 0)
                            * self.env.robot.time_step
                            * self.env.robot.v_pref,
                        )
                        * reward
                        * (1 if t >= ii else 0)
                        for t, reward in enumerate(rewards)
                    ]
                )

            # test = []
            # for ii in range(j - 1):
            #     value = sum(
            #         [
            #             pow(
            #                 self.discount,
            #                 (t - ii) * self.env.robot.time_step * self.env.robot.v_pref,
            #             )
            #             * reward
            #             * (1 if t >= ii else 0)
            #             for t, reward in enumerate(rewards)
            #         ]
            #     )
            #     test.append(value)

            # if isinstance(info, ReachGoal):
            if isinstance(info, ReachGoal) or isinstance(info, Collision):
                for i in range(j):
                    # test = v[i]
                    # yy = r_obs[i]
                    samples = [
                        prev_obs[i].reshape(-1, self.obs_dim),
                        obs[i].reshape(-1, self.obs_dim),
                        prev_r_obs[i],
                        r_obs[i],
                        act[i].reshape(-1, self.act_dim),
                        rwd[i],
                        v[i],
                        is_done[i],
                    ]
                    memory.push(samples)
                # trajs.append(new_traj)
                # print("test")
        # if len(trajs) >= 1:
        #     buffer.add(trajs)

        avg_reward = sum_rewards / k
        avg_cdr = sum_cdrs / k
        avg_return = sum_returns / k
        success_rate = success / k
        collision_rate = collision / k
        timeout_rate = timeout / k
        assert success + collision + timeout == k
        avg_nav_time = (
            sum(success_times) / len(success_times)
            if success_times
            else self.env.time_limit
        )

        return (
            avg_reward,
            avg_cdr,
            avg_return,
            success_rate,
            collision_rate,
            timeout_rate,
            avg_nav_time,
        )

    def exploration_rm_k_ep(
        self,
        memory,
        model=None,
        k=1,
        epsilon=0.5,
        render=False,
        pbar=None,
        random_p_num=False,
        p_range=(1, 5),
    ):
        trajs = []
        d_a = self.act_dim
        sum_rewards = 0
        sum_cdrs = 0
        sum_returns = 0
        ###############################
        success_times = []
        collision_times = []
        timeout_times = []
        success = 0
        collision = 0
        timeout = 0
        too_close = 0
        min_dist = []
        collision_cases = []
        timeout_cases = []
        ###############################
        for _ in range(k):
            if random_p_num:
                p_num = np.random.randint(*p_range)
                self.env.set_human_num(p_num)
                d_o = self.obs_dim * p_num
            else:
                d_o = self.obs_dim * self.env.human_num
            d_ro = self.r_obs_dim
            d_s = self.state_dim * self.env.human_num

            obs = torch.empty((self.max_traj_length, d_o), dtype=torch.float32)
            prev_obs = torch.empty((self.max_traj_length, d_o), dtype=torch.float32)
            act = torch.empty((self.max_traj_length, d_a), dtype=torch.float32)
            r_obs = torch.empty((self.max_traj_length, d_ro), dtype=torch.float32)
            prev_r_obs = torch.empty((self.max_traj_length, d_ro), dtype=torch.float32)

            prev_state = torch.empty((self.max_traj_length, d_s), dtype=torch.float32)
            state = torch.empty((self.max_traj_length, d_s), dtype=torch.float32)
            # next_obs = torch.empty((self.max_traj_length, d_o), dtype=torch.float32)
            # next_r_obs = torch.empty((self.max_traj_length, d_ro), dtype=torch.float32)
            # r_pos = torch.empty((self.max_traj_length, 3), dtype=torch.float32)
            rwd = torch.empty((self.max_traj_length, 1), dtype=torch.float32)
            v = torch.empty((self.max_traj_length, 1), dtype=torch.float32)
            is_done = torch.empty((self.max_traj_length, 1), dtype=torch.float32)
            # Make a reset for each trajectory
            if self.policy != None:
                self.policy.reset()

            o_r = self.env.reset("train")
            joint_obs = JointState(self.env.robot.get_full_state(), o_r)
            robot_obs, human_obs = self.transfunc(joint_obs)
            o0 = torch.clone(human_obs.reshape(-1))
            r_o0 = torch.clone(robot_obs)
            # if model != None:
            #     s = psr.estimate_state(human_obs.reshape(-1).to(psr.device))
            #     # integrated_states = []

            for j in range(self.max_traj_length):
                ro, rp = self.get_r_state()
                if model != None:
                    # integrated_state = psr.get_integrated_state(state.detach(), torch.as_tensor(o).reshape(1,-1).to(psr.device), torch.as_tensor(ro, dtype=torch.float32).to(psr.device))
                    # integrated_states.append(integrated_state)
                    e = np.random.rand()
                    if not model.use_actor:
                        e = np.random.rand()
                        if e < epsilon:
                            a = model.generate_random_action()
                            # a = psr.optimistic_exploration(state.squeeze(), torch.as_tensor(o, dtype=torch.float32).to(psr.device), torch.as_tensor(rp, dtype=torch.float32).to(psr.device), epsilon=epsilon).cpu().data.numpy()
                            # a = psr.generate_random_action_binomial(state.squeeze(), torch.as_tensor(o, dtype=torch.float32).to(psr.device), torch.as_tensor(rp, dtype=torch.float32).to(psr.device)).cpu().data.numpy()
                            # a = self.env.robot.act(o_r)
                            # a = np.clip(np.array(a) + torch.randn_like(torch.as_tensor(a, dtype=torch.float32)).cpu().data.numpy(), -1, 1)
                        else:
                            # a = psr.generate_action_sb(state.squeeze(), human_obs.flatten().to(psr.device), robot_obs.flatten().to(psr.device)).cpu().data.numpy()
                            a = (
                                model.generate_action_mcts(
                                    human_obs.to(model.device),
                                    robot_obs.to(model.device),
                                )
                                .cpu()
                                .data.numpy()
                            )
                            # a = convert_action(a)
                    else:
                        # a = psr.generate_action(integrated_state).cpu().data.numpy()[0]
                        if model.stochastic_actor:
                            a, _, _ = model.actor.sample(
                                (
                                    human_obs.unsqueeze(0).to(model.device),
                                    robot_obs.reshape(1, 1, -1).to(model.device),
                                )
                            )
                            a = (
                                a.clamp(model.actor.act_min, model.actor.act_max)
                                .cpu()
                                .data.numpy()
                                .squeeze()
                            )
                        else:
                            a = (
                                model.generate_action(
                                    (
                                        human_obs.unsqueeze(0).to(model.device),
                                        robot_obs.reshape(1, 1, -1).to(model.device),
                                    )
                                )
                                .cpu()
                                .data.numpy()[0]
                                + np.random.normal(0, 1 * epsilon, size=2)
                            ).clip(-1, 1)
                    # a = psr.generate_random_action_binomial(state.squeeze(), torch.as_tensor(o, dtype=torch.float32).to(psr.device), torch.as_tensor(rp, dtype=torch.float32).to(psr.device)).cpu().data.numpy()
                    # a = self.env.robot.act(o_r)
                    # a = psr.optimistic_exploration(state.squeeze(), torch.as_tensor(o, dtype=torch.float32).to(psr.device), torch.as_tensor(rp, dtype=torch.float32).to(psr.device), epsilon=epsilon).cpu().data.numpy()
                    # a = psr.generate_noised_action(state.squeeze(), torch.as_tensor(o, dtype=torch.float32).to(psr.device), torch.as_tensor(rp, dtype=torch.float32).to(psr.device))
                    a = convert_action(a, kinematics=self.env.robot.kinematics)
                else:
                    if self.policy != None:
                        a = self.policy(o_r)
                    else:
                        a = self.env.robot.act(o_r)
                # obs[j, :] = torch.as_tensor(human_obs.reshape(-1), dtype=torch.float32)
                # r_obs[j, :] = torch.as_tensor(robot_obs, dtype=torch.float32)
                prev_obs[j, :] = torch.as_tensor(
                    human_obs.reshape(-1), dtype=torch.float32
                )
                prev_r_obs[j, :] = torch.as_tensor(robot_obs, dtype=torch.float32)
                o_r, r, done, info = self.env.step(a)

                # prev_state[j, :] = torch.as_tensor(s.reshape(-1), dtype=torch.float32)

                joint_obs = JointState(self.env.robot.get_full_state(), o_r)
                robot_obs, human_obs = self.transfunc(joint_obs)

                act[j, :] = torch.as_tensor(a, dtype=torch.float32)
                obs[j, :] = torch.as_tensor(human_obs.reshape(-1), dtype=torch.float32)
                r_obs[j, :] = torch.as_tensor(robot_obs, dtype=torch.float32)

                # state[j, :] = torch.as_tensor(s.reshape(-1), dtype=torch.float32)
                # next_obs[j, :] = torch.as_tensor(human_obs.reshape(-1), dtype=torch.float32)
                # next_r_obs[j, :] = torch.as_tensor(robot_obs, dtype=torch.float32)
                # r_pos[j, :] = torch.as_tensor(rp, dtype=torch.float32)
                rwd[j, :] = torch.as_tensor(r, dtype=torch.float32)
                is_done[j, :] = torch.as_tensor(1 - int(done), dtype=torch.float32)

                if isinstance(info, Discomfort):
                    too_close += 1
                    min_dist.append(info.min_dist)

                if done:
                    if render:
                        self.env.render()
                    sum_rewards += r
                    break

            if isinstance(info, ReachGoal):
                success += 1
                success_times.append(self.env.global_time)
            elif isinstance(info, Collision):
                collision += 1
                collision_cases.append(k)
                collision_times.append(self.env.global_time)
            elif isinstance(info, Timeout):
                timeout += 1
                timeout_cases.append(k)
                timeout_times.append(self.env.time_limit)

            j += 1
            rewards = rwd[:j, :]

            sum_cdrs += sum(
                [
                    pow(
                        self.discount,
                        t * self.env.robot.time_step * self.env.robot.v_pref,
                    )
                    * reward
                    for t, reward in enumerate(rewards)
                ]
            )
            step_returns = []
            for step in range(len(rewards)):
                step_return = sum(
                    [
                        pow(
                            self.discount,
                            t * self.env.robot.time_step * self.env.robot.v_pref,
                        )
                        * reward
                        for t, reward in enumerate(rewards[step:])
                    ]
                )
                step_returns.append(step_return.cpu().data.numpy())
            sum_returns += average(step_returns)
            for ii in range(j):
                v[ii, :] = sum(
                    [
                        pow(
                            self.discount,
                            max(t - ii, 0)
                            * self.env.robot.time_step
                            * self.env.robot.v_pref,
                        )
                        * reward
                        * (1 if t >= ii else 0)
                        for t, reward in enumerate(rewards)
                    ]
                )

            # if isinstance(info, ReachGoal):
            if isinstance(info, ReachGoal) or isinstance(info, Collision):
                # if True:
                for i in range(j):
                    samples = [
                        prev_obs[i].reshape(-1, self.obs_dim),
                        obs[i].reshape(-1, self.obs_dim),
                        prev_r_obs[i],
                        r_obs[i],
                        act[i].reshape(-1, self.act_dim),
                        # prev_state[i].reshape(-1, self.state_dim),
                        # state[i].reshape(-1, self.state_dim),
                        rwd[i],
                        v[i],
                        is_done[i],
                    ]
                    memory.push(samples)
                # trajs.appendi(new_traj)
            # if len(trajs) >= 1:
            #     buffer.add(trajs)
        if pbar:
            pbar.set_postfix(Reward=str(sum_rewards / k))

        avg_reward = sum_rewards / k
        avg_cdr = sum_cdrs / k
        avg_return = sum_returns / k
        success_rate = success / k
        collision_rate = collision / k
        timeout_rate = timeout / k
        assert success + collision + timeout == k
        avg_nav_time = (
            sum(success_times) / len(success_times)
            if success_times
            else self.env.time_limit
        )

        return (
            avg_reward,
            avg_cdr,
            avg_return,
            success_rate,
            collision_rate,
            timeout_rate,
            avg_nav_time,
        )

    def get_elements(self, trajs):
        col_trajs = [(t.obs, t.act, t.r_obs, t.obs0, t.rwd, t.v) for t in trajs]
        X_obs_rnd = [c[0] for c in col_trajs]
        X_act_rnd = [c[1] for c in col_trajs]
        r_obs = [c[2] for c in col_trajs]
        obs0 = [c[3] for c in col_trajs]
        rwd = [c[4] for c in col_trajs]
        v = [c[5] for c in col_trajs]
        obs_trajs = X_obs_rnd
        act_trajs = X_act_rnd
        torch.cuda.empty_cache()
        return obs_trajs, act_trajs, r_obs, obs0, rwd, v

    def exploration_rm_k_ep_wr(
        self,
        memory,
        model=None,
        k=1,
        epsilon=0.5,
        render=False,
        pbar=None,
        random_p_num=False,
        p_range=(1, 5),
    ):
        trajs = []
        d_a = self.act_dim
        sum_rewards = 0
        sum_cdrs = 0
        sum_returns = 0
        ###############################
        success_times = []
        collision_times = []
        timeout_times = []
        success = 0
        collision = 0
        timeout = 0
        too_close = 0
        min_dist = []
        collision_cases = []
        timeout_cases = []
        ###############################
        for _ in range(k):
            if random_p_num:
                p_num = np.random.randint(*p_range)
                self.env.set_human_num(p_num)
                d_o = self.obs_dim * p_num
            else:
                d_o = self.obs_dim * self.env.human_num
            d_ro = self.r_obs_dim
            d_s = self.state_dim * self.env.human_num

            obs = torch.empty((self.max_traj_length, d_o), dtype=torch.float32)
            prev_obs = torch.empty((self.max_traj_length, d_o), dtype=torch.float32)
            act = torch.empty((self.max_traj_length, d_a), dtype=torch.float32)
            act_orca = torch.empty((self.max_traj_length, d_a), dtype=torch.float32)
            r_obs = torch.empty((self.max_traj_length, d_ro), dtype=torch.float32)
            prev_r_obs = torch.empty((self.max_traj_length, d_ro), dtype=torch.float32)

            prev_state = torch.empty((self.max_traj_length, d_s), dtype=torch.float32)
            state = torch.empty((self.max_traj_length, d_s), dtype=torch.float32)
            rwd = torch.empty((self.max_traj_length, 1), dtype=torch.float32)
            v = torch.empty((self.max_traj_length, 1), dtype=torch.float32)
            is_done = torch.empty((self.max_traj_length, 1), dtype=torch.float32)
            # Make a reset for each trajectory
            if self.policy != None:
                self.policy.reset()

            o_r = self.env.reset("train")
            joint_obs = JointState(self.env.robot.get_full_state(), o_r)
            robot_obs, human_obs = self.transfunc(joint_obs)
            o0 = torch.clone(human_obs.reshape(-1))
            r_o0 = torch.clone(robot_obs)

            for j in range(self.max_traj_length):
                ro, rp = self.get_r_state()
                if model != None:
                    e = np.random.rand()
                    if not model.use_actor:
                        e = np.random.rand()
                        if e < epsilon:
                            a = model.generate_random_action()
                            # a = psr.optimistic_exploration(state.squeeze(), torch.as_tensor(o, dtype=torch.float32).to(psr.device), torch.as_tensor(rp, dtype=torch.float32).to(psr.device), epsilon=epsilon).cpu().data.numpy()
                            # a = psr.generate_random_action_binomial(state.squeeze(), torch.as_tensor(o, dtype=torch.float32).to(psr.device), torch.as_tensor(rp, dtype=torch.float32).to(psr.device)).cpu().data.numpy()
                            # a = self.env.robot.act(o_r)
                            # a = np.clip(np.array(a) + torch.randn_like(torch.as_tensor(a, dtype=torch.float32)).cpu().data.numpy(), -1, 1)
                        else:
                            # a = psr.generate_action_sb(state.squeeze(), human_obs.flatten().to(psr.device), robot_obs.flatten().to(psr.device)).cpu().data.numpy()
                            a = (
                                model.generate_action_mcts(
                                    human_obs.to(model.device),
                                    robot_obs.to(model.device),
                                )
                                .cpu()
                                .data.numpy()
                            )
                            # a = convert_action(a)
                    else:
                        # a = psr.generate_action(integrated_state).cpu().data.numpy()[0]
                        if model.stochastic_actor:
                            a, _, _ = model.actor.sample(
                                (
                                    human_obs.unsqueeze(0).to(model.device),
                                    robot_obs.reshape(1, 1, -1).to(model.device),
                                )
                            )
                            a = a.cpu().data.numpy().squeeze()
                        else:
                            a = (
                                model.generate_action(
                                    (
                                        human_obs.unsqueeze(0).to(model.device),
                                        robot_obs.reshape(1, 1, -1).to(model.device),
                                    )
                                )
                                .cpu()
                                .data.numpy()[0]
                                + np.random.normal(0, 1 * epsilon, size=2)
                            ).clip(-1, 1)

                    a = convert_action(a)
                    a_orca = self.env.robot.act(o_r)

                prev_obs[j, :] = torch.as_tensor(
                    human_obs.reshape(-1), dtype=torch.float32
                )
                prev_r_obs[j, :] = torch.as_tensor(robot_obs, dtype=torch.float32)
                o_r, r, done, info = self.env.step(a)

                joint_obs = JointState(self.env.robot.get_full_state(), o_r)
                robot_obs, human_obs = self.transfunc(joint_obs)

                act[j, :] = torch.as_tensor(a, dtype=torch.float32)
                act_orca[j, :] = torch.as_tensor(a_orca, dtype=torch.float32)
                obs[j, :] = torch.as_tensor(human_obs.reshape(-1), dtype=torch.float32)
                r_obs[j, :] = torch.as_tensor(robot_obs, dtype=torch.float32)

                rwd[j, :] = torch.as_tensor(r, dtype=torch.float32)
                is_done[j, :] = torch.as_tensor(1 - int(done), dtype=torch.float32)

                if isinstance(info, Discomfort):
                    too_close += 1
                    min_dist.append(info.min_dist)

                if done:
                    if render:
                        self.env.render()
                    sum_rewards += r
                    break

            if isinstance(info, ReachGoal):
                success += 1
                success_times.append(self.env.global_time)
            elif isinstance(info, Collision):
                collision += 1
                collision_cases.append(k)
                collision_times.append(self.env.global_time)
            elif isinstance(info, Timeout):
                timeout += 1
                timeout_cases.append(k)
                timeout_times.append(self.env.time_limit)

            # j += 1
            rewards = rwd[: j + 1, :]

            sum_cdrs += sum(
                [
                    pow(
                        self.discount,
                        t * self.env.robot.time_step * self.env.robot.v_pref,
                    )
                    * reward
                    for t, reward in enumerate(rewards)
                ]
            )
            step_returns = []
            for step in range(len(rewards)):
                step_return = sum(
                    [
                        pow(
                            self.discount,
                            t * self.env.robot.time_step * self.env.robot.v_pref,
                        )
                        * reward
                        for t, reward in enumerate(rewards[step:])
                    ]
                )
                step_returns.append(step_return.cpu().data.numpy())
            sum_returns += average(step_returns)
            for ii in range(j):
                v[ii, :] = sum(
                    [
                        pow(
                            self.discount,
                            max(t - ii, 0)
                            * self.env.robot.time_step
                            * self.env.robot.v_pref,
                        )
                        * reward
                        * (1 if t >= ii else 0)
                        for t, reward in enumerate(rewards)
                    ]
                )

                # if isinstance(info, ReachGoal) or isinstance(info, Collision):
            for i in range(j - 1):
                samples = [
                    prev_obs[i].reshape(-1, self.obs_dim),
                    obs[i].reshape(-1, self.obs_dim),
                    prev_r_obs[i],
                    r_obs[i],
                    act[i].reshape(-1, self.act_dim),
                    # prev_state[i].reshape(-1, self.state_dim),
                    # state[i].reshape(-1, self.state_dim),
                    rwd[i],
                    v[i],
                    is_done[i],
                    act_orca[i].reshape(-1, self.act_dim),
                ]
                memory.push(samples)
                # trajs.append(new_traj)
        # if len(trajs) >= 1:
        #     buffer.add(trajs)
        if pbar:
            pbar.set_postfix(Reward=str(sum_rewards / k))

        avg_reward = sum_rewards / k
        avg_cdr = sum_cdrs / k
        avg_return = sum_returns / k
        success_rate = success / k
        collision_rate = collision / k
        timeout_rate = timeout / k
        assert success + collision + timeout == k
        avg_nav_time = (
            sum(success_times) / len(success_times)
            if success_times
            else self.env.time_limit
        )

        return (
            avg_reward,
            avg_cdr,
            avg_return,
            success_rate,
            collision_rate,
            timeout_rate,
            avg_nav_time,
        )

    def get_elements(self, trajs):
        col_trajs = [(t.obs, t.act, t.r_obs, t.obs0, t.rwd, t.v) for t in trajs]
        X_obs_rnd = [c[0] for c in col_trajs]
        X_act_rnd = [c[1] for c in col_trajs]
        r_obs = [c[2] for c in col_trajs]
        obs0 = [c[3] for c in col_trajs]
        rwd = [c[4] for c in col_trajs]
        v = [c[5] for c in col_trajs]
        obs_trajs = X_obs_rnd
        act_trajs = X_act_rnd
        torch.cuda.empty_cache()
        return obs_trajs, act_trajs, r_obs, obs0, rwd, v


class PSRExploerCrowdSimTest:
    def __init__(
        self,
        env,
        max_traj_length,
        min_traj_length=0,
        num_trajs=0,
        num_samples=0,
        obs_dim=0,
        act_dim=0,
        policy=None,
        render=False,
        transfunc=None,
    ):
        super().__init__()

        self.env = env
        if policy != None:
            self.policy = policy
        self.max_traj_length = max_traj_length
        self.min_traj_length = min_traj_length
        self.num_trajs = num_trajs
        self.num_samples = num_samples
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.render = render
        self.transfunc = transfunc

    def get_r_obs(self):
        r_state = self.env.robot.get_full_state()
        r_pos = np.array(r_state.position)
        r_theta = np.array(r_state.theta)
        goal = np.array(r_state.goal_position)
        diff = goal - r_pos
        dist = np.linalg.norm(diff)
        rot = np.arctan2(diff[1], diff[0])
        if self.env.robot.kinematics == "unicycle":
            theta = r_theta - rot
        else:
            # set theta to be zero since it's not used
            theta = 0

        return np.array([dist, theta])
        # return diff

    def exploration(self):
        """
        Generate trajectories of length up max_traj_length each.

        Returns:
        A list of trajectories (See models.Trajectory).

        Additional parameters:
        - model: An object that implements models.FilteringModel interface.
            Used to track the state. If None, an ObservableModel is used,
            which returns the current observation.
        - policy: An object that implements policies.Policy interface.
            Used to provide actions.
        - render: Whether to render generated trajectories in real-time.
            This calls 'render' method which needs ot be implemented.
        - num_trajs: Number of trajectories to return.
        - num_samples: Total number of samples in generated trajectories.

        Must set num_trajs or num_samples (but not both) to a positive number.
        """
        trajs = []

        if (self.num_samples > 0) == (self.num_trajs > 0):
            raise ValueError("Must specify exactly one of num_trajs and num_samples")

        done_all = False
        d_o = self.obs_dim
        d_a = self.act_dim
        i_sample = 0
        tic = time()

        while not done_all:
            obs = torch.empty((self.max_traj_length, d_o), dtype=torch.float32)
            act = torch.empty((self.max_traj_length, d_a), dtype=torch.float32)
            # Make a reset for each trajectory
            if self.policy != None:
                self.policy.reset()
            o_r = self.env.reset("train")
            o = self.transfunc(o_r)
            o0 = torch.clone(torch.as_tensor(o, dtype=torch.float32))

            for j in range(self.max_traj_length):
                if self.render:
                    self.env.render()
                if self.policy != None:
                    a = self.policy(o_r)
                else:
                    a = self.env.robot.act(o_r)
                o_r, r, done, _ = self.env.step(a)
                o = self.transfunc(o_r)
                act[j, :] = torch.as_tensor(a, dtype=torch.float32)
                obs[j, :] = torch.as_tensor(o, dtype=torch.float32)

                if done:
                    break

            j += 1
            drop_traj = False

            if j >= self.min_traj_length:
                # Check if we need to truncate trajectory to maintain num_samples
                if self.num_samples > 0 and i_sample + j >= self.num_samples:
                    j -= i_sample + j - self.num_samples
                    done_all = True
                # TODO: remove this will never happen because of outer if?
                if j < self.min_traj_length:
                    # Last trajectory is too short. Ignore it.
                    drop_traj = True

                if not drop_traj:
                    i_sample += j

                    new_traj = Trajectory(obs=obs[:j, :], act=act[:j, :], obs0=o0)

                    trajs.append(new_traj)

                    if self.num_trajs > 0 and len(trajs) == self.num_trajs:
                        done_all = True
        print("Gathering trajectories took:", time() - tic)
        col_trajs = [(t.obs, t.act, t.obs0) for t in trajs]
        X_obs_rnd = [c[0] for c in col_trajs]
        X_act_rnd = [c[1] for c in col_trajs]
        obs0 = [c[2] for c in col_trajs]
        obs_trajs = X_obs_rnd
        act_trajs = X_act_rnd
        torch.cuda.empty_cache()
        return obs_trajs, act_trajs, obs0
