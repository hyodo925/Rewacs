from tqdm import tqdm
from crowd_sim.envs.utils.info import *
import numpy as np
from numpy.lib.function_base import average
import torch
import matplotlib.pyplot as plt
from utils.env import convert_action
from crowd_sim.envs.utils.state import JointState


def eval_flow(
    eval_env,
    model,
    transfunc,
    eval_episodes=10,
    scenario="test",
    random_p_num=False,
    p_range=(1, 11),
):
    anomaly = 0
    normality = 0
    for i in tqdm(range(eval_episodes)):
        if random_p_num:
            p_num = np.random.randint(*p_range)
            eval_env.set_human_num(p_num)
        obs_r, done = eval_env.reset(scenario), False
        # obs = transfunc(obs_r)
        joint_obs = JointState(eval_env.robot.get_full_state(), obs_r)
        robot_obs, human_obs = transfunc(joint_obs)

        while not done:
            # if model.detect_anomaly(human_obs[:, :2].flatten().unsqueeze(0)):
            if model.switching_necessity(human_obs.unsqueeze(0).to(model.device)):
                anomaly += 1
                # print("Anomaly is detected.")

            else:
                normality += 1

            action = eval_env.robot.act(obs_r)
            obs_r, reward, done, info = eval_env.step(action)
            # obs = transfunc(obs_r)
            joint_obs = JointState(eval_env.robot.get_full_state(), obs_r)
            robot_obs, human_obs = transfunc(joint_obs)

    total_steps = anomaly + normality
    anomaly_rate = float(anomaly) / total_steps * 100
    normality_rate = 100.0 - anomaly_rate
    print("----------------------------")
    print("Total Steps : " + str(total_steps))
    print("Anomaly Rate : " + str(anomaly_rate))
    print("Normality Rate : " + str(normality_rate))
    print("----------------------------")
