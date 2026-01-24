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
from torchrl.data import LazyTensorStorage, ListStorage, ReplayBuffer
from tqdm import tqdm, trange

from rewacs.envs import CrowdSim
from rewacs.envs.policy.policy_factory import policy_factory
from rewacs.envs.utils.action import ActionRot, ActionXY, ActionXYW
from rewacs.envs.utils.robot import Robot
from rewacs.envs.utils.transformations import GetRobotFrameObs
from rl_navigation import RLNavigation
from utils.evaluation import eval_policy_with_flow
from utils.explorer import ExplorerCrowdSim
from utils.models import (
    SocialCritic,
)
from utils.state_integrators import (
    EmbeddedGaussianIntegrator,
)
from algo.awac.trainer import AWAC
from algo.awac.actor import SocialActorAWAC
from flow.model import GraphSituationFlow

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


############### policy model ##################
# Settings
run_dir = "wandb/awac_training/wandb/run-20260113_130327-zgc57h94"

config_path = os.path.join("configs/awac_config.py")

model_path = os.path.join(run_dir, "files/trained_models/model_best.pth")
#################################

############# flow model ####################
# Settings
flow_run_dir = "wandb/Switching_Administrator_training/wandb/run-20260121_142518-hifx4rkp"

flow_config_path = os.path.join(flow_run_dir, "files/config.py")

flow_model_path = os.path.join(flow_run_dir, "files/trained_models/model_500.pth")

render = False
render_type = "video"
################################_
start_time_log = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

spec = importlib.util.spec_from_file_location("config", config_path)

config = importlib.util.module_from_spec(spec)
spec.loader.exec_module(config)

cfg = config.cfg

flow_spec = importlib.util.spec_from_file_location("config", flow_config_path)

flow_config = importlib.util.module_from_spec(flow_spec)
flow_spec.loader.exec_module(flow_config)

flow_cfg = flow_config.cfg

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

######### policy ##########
actor_integrator = EmbeddedGaussianIntegrator(
    cfg.model.obs_dim,
    cfg.model.r_obs_dim,
    projection_dim=cfg.model.projection_dim,
    enc_hdims=cfg.model.actor_integrator_enc_hdims,
)

actor = SocialActorAWAC(
    cfg.model.projection_dim,
    cfg.model.action_dim,
    action_space=cfg.model.action_space,
    h_dims=cfg.model.actor_h_dims,
    integrator=actor_integrator,
)

critic_integrator = EmbeddedGaussianIntegrator(
    cfg.model.obs_dim,
    cfg.model.r_obs_dim,
    projection_dim=cfg.model.projection_dim,
    enc_hdims=cfg.model.critic_integrator_enc_hdims,
)

critic = SocialCritic(
    cfg.model.projection_dim + cfg.model.action_dim,
    1,
    h_dims=cfg.model.critic_h_dims,
    integrator=critic_integrator,
    single=False,
)

model = RLNavigation(actor=actor, critic=critic)

# jsd = float("inf")

use_rule_based = False
# tbar = trange(args.total_it)
buffer = ReplayBuffer()

buffer.extend(range(5000))

critic_optimizer = torch.optim.Adam(model.critic.parameters(), lr=cfg.train.lr)
actor_optimizer = torch.optim.Adam(model.actor.parameters(), lr=cfg.train.lr)

trainer = AWAC(
    model=model,
    replay_buffer=buffer,
    actor_optimizer=actor_optimizer,
    critic_optimizer=critic_optimizer,
    batch_size=cfg.train.batch_size,
)

model.load_model(model_path)
model.to(device)
######### flow ##########
flow = GraphSituationFlow(
    obs_dim=cfg.model.obs_dim,
    h_dim=flow_cfg.model.h_dim,
    n_flow_blocks=flow_cfg.model.n_flow_blocks,
    n_flow_hidden_num=flow_cfg.model.n_flow_hidden_num,
    threshold_type=flow_cfg.model.threshold_type,
)
flow.load_model(flow_model_path)
flow.to(device)

loss_list = []

max_cdr = float("-inf")

# model.to(device)
# fig, ax = plt.subplots(figsize=(7, 7))
# eval_orca_policy(eval_env=env, psr=psr, transfunc=transfunc, eval_episodes=500)

if render and (render_type == "video"):
    path_v = os.path.join(
        run_dir, f"files/videos/{start_time_log}_{trainer.alg_name}_CrowdSim"
    )
    os.makedirs(os.path.join(run_dir, "files/videos"), exist_ok=True)
    os.mkdir(path_v)

else:
    path_v = None

eval_policy_with_flow(
    eval_env=env,
    model=model,
    flow=flow,
    transfunc=transfunc,
    convert_action=convert_action,
    eval_episodes=env.case_size["test"],
    scenario="test",
    render=render,
    render_type=render_type,
    path=path_v,
    print_results=True,
)
