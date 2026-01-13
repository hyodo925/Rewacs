import argparse
import configparser
import copy
import csv
import datetime
import importlib
import json
import os
import random
import shutil

import matplotlib.pyplot as plt
import numpy as np
import torch
from torchrl.data import LazyTensorStorage, ReplayBuffer, SamplerWithoutReplacement
from tqdm import tqdm, trange

from rewacs.envs import CrowdSim
from rewacs.envs.policy.policy_factory import policy_factory
from rewacs.envs.utils.action import ActionRot, ActionXY, ActionXYW
from rewacs.envs.utils.robot import Robot
from rewacs.envs.utils.transformations import GetRobotFrameObs
from utils.evaluation import eval_policy
from utils.explorer import ExplorerCrowdSim
from flow.model import GraphSituationFlow
from flow.trainer import grevnet_training
try:
    import wandb
except ModuleNotFoundError:
    pass


def seed_all(seed):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(seed)


def define_env(
    config,
    debug=False,
):
    cfg = config.cfg
    env = CrowdSim()
    env.configure(cfg)
    robot = Robot(cfg, "robot")
    robot.time_step = env.time_step
    env.set_robot(robot)

    if robot.visible:
        safety_space = 0
    else:
        safety_space = 0.15

    policy = policy_factory[cfg.robot.policy]()
    policy.safety_space = safety_space

    robot.set_policy(policy)

    if debug:
        print(config.b.to_dict(config.cfg))

    return env, robot


start_time_log = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("Using MPS")
elif torch.cuda.is_available():
    device = torch.device("cuda")
    print("Using CUDA")
else:
    device = torch.device("cpu")
    print("Using CPU")
# start_time_log = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

config_path = "./configs/flow_config.py"
spec = importlib.util.spec_from_file_location("config", config_path)

config = importlib.util.module_from_spec(spec)
spec.loader.exec_module(config)

cfg = config.cfg

if cfg.log.wandb:
    # wandb.tensorboard.patch(root_logdir=f"logs/{start_time_log}")

    run = wandb.init(
        project=cfg.log.wandb_project, 
        save_code=True,
        mode=cfg.log.wandb_mode,
        name=f"{start_time_log}_switching_administrator_training",
        dir=f"wandb/Switching_Administrator_training",
    )
    run.config.update(config.b.to_wandb_dict(cfg))

    results_log_columns = [
        "reward",
        "cdr",
        "return",
        "success_rate",
        "collision_rate",
        "timeout_rate",
        "avg_nav_time",
    ]
    val_log_columns = ["step_num"] + results_log_columns
    val_table = wandb.Table(columns=val_log_columns)

    shutil.copy(config_path, os.path.join(run.dir, "config.py"))

    code_artifact = wandb.Artifact(name="config_code_artifact", type="code")
    code_artifact.add_file(os.path.join(run.dir, "config.py"))
    wandb.log_artifact(code_artifact)

    if cfg.log.save_model:
        trained_models_dir = os.path.join(run.dir, "trained_models")
        os.makedirs(trained_models_dir, exist_ok=True)


seed_all(cfg.train.random_seed)

##################################################################################
# load env
env, robot = define_env(debug=True, config=config)
##################################################################################

transfunc = GetRobotFrameObs(
    with_peds_vel=cfg.transfunc.with_peds_vel,
    peds_vel_as_relative=cfg.transfunc.peds_vel_as_relative,
    use_omega=cfg.transfunc.use_omega,
)


def convert_action(action):
    action = ActionXY(action[0], action[1])

    return action


model = GraphSituationFlow(
    obs_dim=cfg.model.obs_dim,
    h_dim=cfg.model.h_dim,
    n_flow_blocks=cfg.model.n_flow_blocks,
    n_flow_hidden_num=cfg.model.n_flow_hidden_num,
    threshold_type=cfg.model.threshold_type,
)

model.to(device)

expl = ExplorerCrowdSim(
    env=env,
    # num_samples=5000,
    obs_dim=cfg.model.obs_dim,
    act_dim=cfg.model.action_dim,
    r_obs_dim=cfg.model.r_obs_dim,
    transfunc=transfunc,
    convert_action=convert_action,
    render=False,
)

buffer = ReplayBuffer(storage=LazyTensorStorage(cfg.train.buffer_capacity),sampler=SamplerWithoutReplacement())

expl_logs = expl.exploration_k_ep_orca(
    buffer=buffer,
    k=cfg.train.preliminary_exp_n,
    # k=100,
    render=False,
)

max_cdr = float("-inf")

# data_for_set_th = data_for_flow_set_threshold(memory_total)

lr = cfg.train.lr
epoch_num = cfg.train.total_it

flow_optimizer = torch.optim.Adam(model.parameters(), lr=lr)

# obs_data_list = []
# r_obs_data_list = []
# for data in memory.memory:
#     prev_obs, _, prev_r_obs, _, _, _, _, _ = data
#     obs_data_list.append(prev_obs)
#     r_obs_data_list.append(prev_r_obs.unsqueeze(0))
# obs_data_stack = torch.squeeze(torch.stack(obs_data_list), 2)
# r_obs_data_stack = torch.squeeze(torch.stack(r_obs_data_list), 2)

# data_for_set_th = (obs_data_stack, r_obs_data_stack)
# data_for_set_th = obs_data_stack

grevnet_training(
    model=model,
    data_loader=buffer,
    flow_optimizer=flow_optimizer,
    epoch_num=epoch_num,
    model_dir=trained_models_dir ,
    model_save_freq=cfg.eval.eval_interval,
    data_for_logging= run if cfg.log.wandb else None,
    # data_th=data_for_set_th,
)






