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
from utils.evaluation import eval_policy
from utils.explorer import ExplorerCrowdSim
from utils.models import (
    SocialCritic,
)
from utils.state_integrators import (
    EmbeddedGaussianIntegrator,
)
from algo.td3bc.trainer import TD3BC
from algo.awac.actor import SocialActorAWAC

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

config_path = "./configs/td3bc_config.py"
spec = importlib.util.spec_from_file_location("config", config_path)

config = importlib.util.module_from_spec(spec)
spec.loader.exec_module(config)

cfg = config.cfg

if cfg.log.wandb:
    # wandb.tensorboard.patch(root_logdir=f"logs/{start_time_log}")

    run = wandb.init(
        project=cfg.log.wandb_project, save_code=True, mode=cfg.log.wandb_mode
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

critic_optimizer = torch.optim.Adam(model.critic.parameters(), lr=cfg.train.lr)
actor_optimizer = torch.optim.Adam(model.actor.parameters(), lr=cfg.train.lr)

trainer = TD3BC(
    model=model,
    replay_buffer=buffer,
    actor_optimizer=actor_optimizer,
    critic_optimizer=critic_optimizer,
    batch_size=cfg.train.batch_size,
)

expl_logs = expl.exploration_k_ep_orca(
    buffer=buffer,
    k=cfg.train.preliminary_exp_n,
    # k=100,
    render=False,
)


loss_list = []

# val_logs = eval_policy(
#     eval_env=env,
#     model=model,
#     transfunc=transfunc,
#     eval_episodes=env.case_size["test"],
#     scenario="test",
#     render=False,
#     print_results=True
# )

max_cdr = float("-inf")
with tqdm(range(cfg.train.total_it), desc=trainer.alg_name + " Training") as pbar:
    for i, ch in enumerate(pbar):
        with torch.no_grad():
            # if i < 1000:

            if not cfg.train.offline_learning:
                expl_logs = expl.exploration_k_ep(
                    buffer=buffer,
                    model=model,
                    pbar=pbar,
                    render=False,
                )

                if cfg.log.wandb:
                    run.log(
                        {
                            "expl/reward": expl_logs[0],
                            "expl/cdr": expl_logs[1],
                            "expl/return": expl_logs[2],
                            "expl/success_rate": expl_logs[3],
                            "expl/collision_rate": expl_logs[4],
                            "expl/timeout_rate": expl_logs[5],
                            "expl/avg_nav_time": expl_logs[6],
                        },
                        step=i + 1,
                    )

        trainer.update(
            update_actor=((cfg.train.total_it % cfg.train.actor_update_interval) == 0),
            data_for_logging=(run, i + 1) if cfg.log.wandb else None,
        )

        # total_it += 1
        if ((i + 1) % cfg.train.target_update_interval) == 0:
            trainer.update_target()

        if (i + 1) % cfg.eval.eval_interval == 0:
            val_logs = eval_policy(
                eval_env=env,
                model=model,
                transfunc=transfunc,
                convert_action=convert_action,
                eval_episodes=env.case_size["val"],
                scenario="val",
                render=cfg.eval.val_render,
                print_results=True,
            )
            if cfg.log.wandb:
                wandb.log(
                    {
                        "val/reward": val_logs[0],
                        "val/cdr": val_logs[1],
                        "val/return": val_logs[2],
                        "val/success_rate": val_logs[3],
                        "val/collision_rate": val_logs[4],
                        "val/timeout_rate": val_logs[5],
                        "val/avg_nav_time": val_logs[6],
                    },
                    step=i + 1,
                )

                val_log_data = [i + 1] + list(val_logs)
                val_table.add_data(*val_log_data)
                if i + 1 == cfg.train.total_it:
                    run.log({"Validation Table": val_table})

            update_best = val_logs[1] > max_cdr
            if update_best:
                best_model = copy.deepcopy(model.state_dict())
                best_step_num = i + 1
                max_cdr = val_logs[1]

            if cfg.log.save_model:
                model.save_model(trained_models_dir + "/model_{}.pth".format(i + 1))
                if update_best:
                    model.save_model(trained_models_dir + "/model_best.pth")

# model.to(device)
# fig, ax = plt.subplots(figsize=(7, 7))
# eval_orca_policy(eval_env=env, psr=psr, transfunc=transfunc, eval_episodes=500)
render = cfg.eval.render
render_type = cfg.eval.render_type
if render and (render_type == "video"):
    if cfg.log.wandb:
        path_v = os.path.join(run.dir, "videos/training_results")
        os.makedirs(path_v, exist_ok=True)
    else:
        path_v = "videos/{}_{}_{}".format(start_time_log, trainer.alg_name, "CrowdSim")
        os.mkdir(path_v)
else:
    path_v = None

model.load_state_dict(best_model)
print(f"The best model number is {best_step_num}")

test_logs = eval_policy(
    eval_env=env,
    model=model,
    transfunc=transfunc,
    convert_action=convert_action,
    eval_episodes=env.case_size["test"],
    scenario="test",
    render=render,
    render_type=render_type,
    path=path_v,
    print_results=True,
)


if cfg.log.wandb:
    # run.log({"Validation Table": val_table})

    test_log_columns = ["bset_step_num"] + results_log_columns
    test_log_data = [best_step_num] + list(test_logs)
    test_table = wandb.Table(columns=test_log_columns)
    test_table.add_data(*test_log_data)

    run.log({"Test Table": test_table})
    wandb.finish()