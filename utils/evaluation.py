import csv
import datetime

import matplotlib.cm as cm
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import torch
from numpy import average
from tqdm import tqdm

from rewacs.envs.utils.info import Collision, Discomfort, ReachGoal, Timeout
from rewacs.envs.utils.state import JointState


def MSE(Y, YH):
    return np.square(Y - YH).mean()


def RMSE(Y, YH):
    return np.sqrt(np.square(Y - YH).mean())


def eval_policy(
    eval_env,
    model,
    transfunc,
    convert_action,
    discount=0.9,
    render=False,
    render_type="",
    path=None,
    eval_episodes=10,
    scenario="test",
    random_p_num=False,
    p_range=(1, 11),
    ax=None,
    output_name=None,
    print_results=False,
):
    rewards = []
    sum_rewards = 0.0
    sum_cdrs = 0.0
    sum_returns = 0.0

    eval_env.robot.print_info()

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

    for i in tqdm(range(eval_episodes)):
        if random_p_num:
            p_num = np.random.randint(*p_range)
            eval_env.set_human_num(p_num)
        robot_state, human_state = eval_env.reset(scenario)
        done = False
        robot_obs, humans_obs = transfunc(
            robot_state,
            human_state,
        )

        # robot_obs, human_obs = robot_obs/4., human_obs/4.
        # state = psr.state0.expand(5, 20).to(psr.device)
        while not done:
            _, _, action = model.generate_action(
                (
                    humans_obs.unsqueeze(0).to(model.device),
                    robot_obs.reshape(1, 1, -1).to(model.device),
                )
            )
            action = (
                action.clamp(model.actor.act_min, model.actor.act_max)
                .cpu()
                .data.numpy()
                .squeeze()
            )
            action = convert_action(action)
            # action = eval_env.robot.act(obs_r)
            # obs_pred = model.predict_obs(
            #     human_obs.reshape(1, -1).to(model.device), robot_obs.to(model.device)
            # )

            robot_state, human_state, reward, done, info = eval_env.step(action)
            robot_obs, humans_obs = transfunc(
                robot_state,
                human_state,
            )

            # robot_obs, human_obs = robot_obs/4., human_obs/4.
            # if ax != None:
            #     pos = obs_pred[0, :, :2]
            #     ax.clear()
            #     for human in pos:
            #         ax.set_xlim(-4, 4)
            #         ax.set_ylim(-4, 4)
            #         human_circle = plt.Circle(human, 0.3, fill=False, color='b')
            #         ax.add_artist(human_circle)
            #     plt.pause(0.5)

            sum_rewards += reward
            rewards.append(reward)
            if render and done:
                if render_type == "video":
                    eval_env.render("video", path + "/" + str(i) + ".mp4")
                elif render_type == "traj":
                    if path:
                        eval_env.render("traj", path + "/" + str(i) + ".png")
                    else:
                        eval_env.render("traj")
                else:
                    eval_env.render()
            # if isinstance(info, Danger):
            if isinstance(info, Discomfort):
                too_close += 1
                min_dist.append(info.min_dist)
        if isinstance(info, ReachGoal):
            success += 1
            success_times.append(eval_env.global_time)
        elif isinstance(info, Collision):
            collision += 1
            collision_cases.append(i)
            collision_times.append(eval_env.global_time)
        elif isinstance(info, Timeout):
            timeout += 1
            timeout_cases.append(i)
            timeout_times.append(eval_env.time_limit)
        cdr = sum(
            [
                pow(discount, t * eval_env.robot.time_step * eval_env.robot.v_pref) * r
                for t, r in enumerate(rewards)
            ]
        )
        step_returns = []
        for step in range(len(rewards)):
            step_return = sum(
                [
                    pow(discount, t * eval_env.robot.time_step * eval_env.robot.v_pref)
                    * reward
                    for t, reward in enumerate(rewards[step:])
                ]
            )
            step_returns.append(step_return)
        sum_returns += average(step_returns)
        del rewards[:]
        sum_cdrs += cdr

    avg_reward = sum_rewards / eval_episodes
    avg_cdr = sum_cdrs / eval_episodes
    avg_return = sum_returns / eval_episodes
    success_rate = success / eval_episodes
    collision_rate = collision / eval_episodes
    timeout_rate = timeout / eval_episodes
    assert success + collision + timeout == eval_episodes
    avg_nav_time = (
        sum(success_times) / len(success_times)
        if success_times
        else eval_env.time_limit
    )

    if print_results:
        print("\n")
        print("----------------------------")
        print("Scenario : " + str(eval_env.test_scenario) + "-" + str(scenario))
        print("----------------------------")
        print(
            f"Evaluation over {eval_episodes} Average Reward: {avg_reward:.3f} Average Cumulative Discounted Reward: {avg_cdr:.3f}, Average Return: {avg_return:.3f}"
        )
        print(
            f"Success Rate {success_rate} Collision Rate: {collision_rate:.3f} Timeout Rate: {timeout_rate:.3f} Success Time: {avg_nav_time:.3f}"
        )
        print("----------------------------")

        if output_name is not None:
            with open(output_name + ".csv", mode="w") as f:
                writer = csv.writer(f)
                header = [
                    "Average Reward",
                    "Average Cumulative Discounted Reward",
                    "Average Return",
                    "Success Rate",
                    "Collision Rate",
                    "Timeout Rate",
                    "Success Time",
                ]
                results = [
                    avg_reward,
                    avg_cdr,
                    avg_return,
                    success_rate,
                    collision_rate,
                    timeout_rate,
                    avg_nav_time,
                ]
                writer.writerow(header)
                writer.writerow(results)

            with open(output_name + ".txt", mode="w") as f:
                f.write("----------------------------" + "\n")
                f.write(
                    "Scenario : "
                    + str(eval_env.test_scenario)
                    + "-"
                    + str(scenario)
                    + "\n"
                )
                f.write("----------------------------" + "\n")
                f.write(
                    f"Evaluation over {eval_episodes} Average Reward: {avg_reward:.3f} Average Cumulative Discounted Reward: {avg_cdr:.3f}, Average Return: {avg_return:.3f}\n"
                )
                f.write(
                    f"Success Rate {success_rate} Collision Rate: {collision_rate:.3f} Timeout Rate: {timeout_rate:.3f} Success Time: {avg_nav_time:.3f}\n"
                )
                f.write("----------------------------\n")
    return (
        avg_reward,
        avg_cdr,
        avg_return,
        success_rate,
        collision_rate,
        timeout_rate,
        avg_nav_time,
    )



def eval_policy_with_flow(
    eval_env,
    model,
    flow,
    transfunc,
    convert_action,
    discount=0.9,
    render=False,
    render_type="",
    path=None,
    eval_episodes=10,
    scenario="test",
    random_p_num=False,
    p_range=(1, 11),
    ax=None,
    output_name=None,
    print_results=False,
    render_switching=False,
    learning_based_only=False,
):
    rewards = []
    sum_rewards = 0.0
    sum_cdrs = 0.0
    sum_returns = 0.0

    eval_env.robot.print_info()

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
    switching_list = []
    anomaly = 0
    normality = 0
    ###############################

    for i in tqdm(range(eval_episodes)):
        if random_p_num:
            p_num = np.random.randint(*p_range)
            eval_env.set_human_num(p_num)
        robot_state, human_state = eval_env.reset(scenario)
        done = False
        robot_obs, humans_obs = transfunc(
            robot_state,
            human_state,
        )

        # robot_obs, human_obs = robot_obs/4., human_obs/4.
        # state = psr.state0.expand(5, 20).to(psr.device)
        while not done:
            _, _, action = model.generate_action(
                (
                    humans_obs.unsqueeze(0).to(model.device),
                    robot_obs.reshape(1, 1, -1).to(model.device),
                )
            )
            action = (
                action.clamp(model.actor.act_min, model.actor.act_max)
                .cpu()
                .data.numpy()
                .squeeze()
            )
            action_lb = convert_action(action)
            # action = eval_env.robot.act(obs_r)
            # obs_pred = model.predict_obs(
            #     human_obs.reshape(1, -1).to(model.device), robot_obs.to(model.device)
            # )
            action_orca = eval_env.robot.act(human_state)
            switching_necessity = flow.switching_necessity(
                humans_obs.unsqueeze(0).to(flow.device)
            )
            if switching_necessity:
                # if flow.detect_anomaly(
                #     (
                #         human_obs.unsqueeze(0).to(flow.device),
                #         robot_obs.reshape(1, 1, -1).to(flow.device),
                #     ),
                # ):
                if learning_based_only:
                    action = action_lb
                else:
                    action = action_orca
                anomaly += 1
                # anomaly_in_ep += 1
                if render_switching:
                    switching_list.append(True)
            else:
                action = action_lb
                normality += 1
                # normality_in_ep += 1
                if render_switching:
                    switching_list.append(False)

            robot_state, human_state, reward, done, info = eval_env.step(action)
            robot_obs, humans_obs = transfunc(
                robot_state,
                human_state,
            )


            
            
            # robot_obs, human_obs = robot_obs/4., human_obs/4.
            # if ax != None:
            #     pos = obs_pred[0, :, :2]
            #     ax.clear()
            #     for human in pos:
            #         ax.set_xlim(-4, 4)
            #         ax.set_ylim(-4, 4)
            #         human_circle = plt.Circle(human, 0.3, fill=False, color='b')
            #         ax.add_artist(human_circle)
            #     plt.pause(0.5)

            sum_rewards += reward
            rewards.append(reward)
            if render and done:
                if render_type == "video":
                    eval_env.render("video", path + "/" + str(i) + ".mp4")
                elif render_type == "traj":
                    if path:
                        eval_env.render("traj", path + "/" + str(i) + ".png")
                    else:
                        eval_env.render("traj")
                else:
                    eval_env.render()
            # if isinstance(info, Danger):
            if isinstance(info, Discomfort):
                too_close += 1
                min_dist.append(info.min_dist)
        if isinstance(info, ReachGoal):
            success += 1
            success_times.append(eval_env.global_time)
        elif isinstance(info, Collision):
            collision += 1
            collision_cases.append(i)
            collision_times.append(eval_env.global_time)
        elif isinstance(info, Timeout):
            timeout += 1
            timeout_cases.append(i)
            timeout_times.append(eval_env.time_limit)
        cdr = sum(
            [
                pow(discount, t * eval_env.robot.time_step * eval_env.robot.v_pref) * r
                for t, r in enumerate(rewards)
            ]
        )
        step_returns = []
        for step in range(len(rewards)):
            step_return = sum(
                [
                    pow(discount, t * eval_env.robot.time_step * eval_env.robot.v_pref)
                    * reward
                    for t, reward in enumerate(rewards[step:])
                ]
            )
            step_returns.append(step_return)
        sum_returns += average(step_returns)
        del rewards[:]
        sum_cdrs += cdr

    avg_reward = sum_rewards / eval_episodes
    avg_cdr = sum_cdrs / eval_episodes
    avg_return = sum_returns / eval_episodes
    success_rate = success / eval_episodes
    collision_rate = collision / eval_episodes
    timeout_rate = timeout / eval_episodes
    assert success + collision + timeout == eval_episodes
    avg_nav_time = (
        sum(success_times) / len(success_times)
        if success_times
        else eval_env.time_limit
    )
    total_steps = anomaly + normality
    anomaly_rate = float(anomaly) / total_steps * 100
    normality_rate = 100.0 - anomaly_rate

    if print_results:
        print("\n")
        print("----------------------------")
        print("Scenario : " + str(eval_env.test_scenario) + "-" + str(scenario))
        print("----------------------------")
        print("Total Steps : " + str(total_steps))
        print("Anomaly Rate : " + str(anomaly_rate))
        print("Normality Rate : " + str(normality_rate))
        print("----------------------------")
        print(
            f"Evaluation over {eval_episodes} Average Reward: {avg_reward:.3f} Average Cumulative Discounted Reward: {avg_cdr:.3f}, Average Return: {avg_return:.3f}"
        )
        print(
            f"Success Rate {success_rate} Collision Rate: {collision_rate:.3f} Timeout Rate: {timeout_rate:.3f} Success Time: {avg_nav_time:.3f}"
        )
        print("----------------------------")

        if output_name is not None:
            with open(output_name + ".csv", mode="w") as f:
                writer = csv.writer(f)
                header = [
                    "Average Reward",
                    "Average Cumulative Discounted Reward",
                    "Average Return",
                    "Success Rate",
                    "Collision Rate",
                    "Timeout Rate",
                    "Success Time",
                ]
                results = [
                    avg_reward,
                    avg_cdr,
                    avg_return,
                    success_rate,
                    collision_rate,
                    timeout_rate,
                    avg_nav_time,
                ]
                writer.writerow(header)
                writer.writerow(results)

            with open(output_name + ".txt", mode="w") as f:
                f.write("----------------------------" + "\n")
                f.write(
                    "Scenario : "
                    + str(eval_env.test_scenario)
                    + "-"
                    + str(scenario)
                    + "\n"
                )
                f.write("----------------------------" + "\n")
                f.write(
                    f"Evaluation over {eval_episodes} Average Reward: {avg_reward:.3f} Average Cumulative Discounted Reward: {avg_cdr:.3f}, Average Return: {avg_return:.3f}\n"
                )
                f.write(
                    f"Success Rate {success_rate} Collision Rate: {collision_rate:.3f} Timeout Rate: {timeout_rate:.3f} Success Time: {avg_nav_time:.3f}\n"
                )
                f.write("----------------------------\n")
    return (
        avg_reward,
        avg_cdr,
        avg_return,
        success_rate,
        collision_rate,
        timeout_rate,
        avg_nav_time,
    )

