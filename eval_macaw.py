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
from meta_rl_navigation import MetaRLNavigation
from utils.evaluation import eval_policy
from algo.macaw.explorer import ExplorerCrowdSim
from algo.macaw.critic import  SocialCritic
from utils.state_integrators import (
    EmbeddedGaussianIntegrator,
)
from algo.macaw.trainer import MACAW
from algo.macaw.actor import SocialActorMACAW

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


#################################
# Settings
run_dir = "wandb/macaw_training/wandb/run-20260124_230911-cswmubg4"

config_path = os.path.join("configs/macaw_config.py")

model_path = os.path.join(run_dir, "files/trained_models/model_best.pth")

render = False
render_type = "video"
#################################

start_time_log = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

spec = importlib.util.spec_from_file_location("config", config_path)

config = importlib.util.module_from_spec(spec)
spec.loader.exec_module(config)

cfg = config.cfg

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


actor_integrator = EmbeddedGaussianIntegrator(
    cfg.model.obs_dim,
    cfg.model.r_obs_dim,
    projection_dim=cfg.model.projection_dim,
    enc_hdims=cfg.model.actor_integrator_enc_hdims,
)

actor = SocialActorMACAW(
    cfg.model.projection_dim,
    # cfg.model.projection_dim + cfg.train.num_tasks,
    cfg.model.action_dim,
    action_space=cfg.model.action_space,
    h_dims=cfg.model.actor_h_dims,
    integrator=actor_integrator,
    use_adv_head=cfg.model.use_adv_head,
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

value = SocialCritic(
    cfg.model.projection_dim,
    # cfg.model.projection_dim + cfg.train.num_tasks,
    1,
    h_dims=cfg.model.critic_h_dims,
    integrator=critic_integrator,
    single=True,
)

model = MetaRLNavigation(actor=actor, critic=critic, value=value)
model.load_model(model_path)
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

# jsd = float("inf")

use_rule_based = False
# tbar = trange(args.total_it)
buffer = ReplayBuffer(storage=LazyTensorStorage(cfg.train.buffer_capacity))

value_optimizer = torch.optim.Adam(model.value.parameters(), lr=cfg.train.lr)
critic_optimizer = torch.optim.Adam(model.critic.parameters(), lr=cfg.train.lr)
actor_optimizer = torch.optim.Adam(model.actor.parameters(), lr=cfg.train.lr)

tasks = []
expl_logs = expl.exploration_k_ep_orca(
    buffer=buffer,
    k=10, #cfg.train.preliminary_exp_n,
    scenario=cfg.sim.test_scenario,
    human_num=cfg.sim.human_num,
    policy=cfg.humans.policy,
    # k=100,
    render=False
)
tasks.append(buffer)

trainer = MACAW(
    model=model,
    tasks=tasks,
    actor_optimizer=actor_optimizer,
    critic_optimizer=critic_optimizer,
    value_optimizer=value_optimizer,
    batch_size=cfg.train.batch_size,
)

loss_list = []

max_cdr = float("-inf")



# model.to(device)
# fig, ax = plt.subplots(figsize=(7, 7))
# eval_orca_policy(eval_env=env, psr=psr, transfunc=transfunc, eval_episodes=500)
render = cfg.eval.val_render
render_type = cfg.eval.render_type
if render and (render_type == "video"):
    path_v = os.path.join(
        run_dir, f"files/videos/{start_time_log}_{trainer.alg_name}_CrowdSim"
    )
    os.makedirs(os.path.join(run_dir, "files/videos"), exist_ok=True)
    os.mkdir(path_v)
elif render and (render_type == "traj"):
    path_v = os.path.join(
       f"trajs/eval_macaw/{cfg.sim.val_scenario}/{cfg.humans.test_policy}/{cfg.sim.human_num}/{cfg.train.random_seed}"
    )
    os.makedirs(path_v, exist_ok=True)

else:
    path_v = None

for i in range(1):
    model = trainer.step(
        # update_actor=((cfg.train.total_it % cfg.train.actor_update_interval) == 0),
        # data_for_logging=None #(run, i + 1) if cfg.log.wandb else None,
    )
output_path = f"results/eval_macaw/{cfg.sim.val_scenario}_{cfg.humans.test_policy}_{cfg.sim.human_num}_{cfg.train.random_seed}"

eval_policy(
    eval_env=env,
    model=model,
    transfunc=transfunc,
    scenario=cfg.sim.val_scenario,
    human_num=cfg.sim.human_num,
    policy=cfg.humans.policy,
    convert_action=convert_action,
    eval_episodes=env.case_size["test"],
    phase="test",
    render=cfg.eval.val_render,
    render_type=cfg.eval.render_type,
    path=path_v,
    print_results=True,
    output_name=output_path
)
