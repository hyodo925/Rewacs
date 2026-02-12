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
from torchrl.data.replay_buffers import TensorDictReplayBuffer
from tqdm import tqdm, trange

from rewacs.envs import CrowdSim
from rewacs.envs.policy.policy_factory import policy_factory
from rewacs.envs.utils.action import ActionRot, ActionXY, ActionXYW
from rewacs.envs.utils.robot import Robot
from rewacs.envs.utils.transformations import GetRobotFrameObs
from rl_navigation import RLNavigation
from utils.utils import calculate_odpr_weights
from utils.buffers import ODPRSampler
from utils.evaluation import eval_policy
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

############### policy model ##################
# Settings
# run_dir = "wandb/awac_training/wandb/run-20260113_130327-zgc57h94"square_size=10
# run_dir = "wandb/awac_training/wandb/run-20260205_203727-5n02xam3"#square_size=20
# run_dir = "wandb/awac_training/wandb/run-20260205_235253-4xvp6il5"#log_std_min=-6 + square_size=20
run_dir = "wandb/awac_training/wandb/run-20260209_154749-v6xtokdu" #Weight Clipping

config_path = os.path.join("configs/awac_odpr_config.py")

model_path = os.path.join(run_dir, "files/trained_models/model_best.pth")
#################################

############# flow model ####################
# Settings
flow_run_dir = "wandb/Switching_Administrator_training/wandb/run-20260121_142518-hifx4rkp"
# flow_run_dir = "wandb/Switching_Administrator_training_with_10ep_data/wandb/run-20260210_010613-jxrdf2lc" #+10ep

flow_config_path = os.path.join(flow_run_dir, "files/config.py")

flow_model_path = os.path.join(flow_run_dir, "files/trained_models/model_500.pth")

render = False
render_type = "video"

spec = importlib.util.spec_from_file_location("config", config_path)

config = importlib.util.module_from_spec(spec)
spec.loader.exec_module(config)

cfg = config.cfg

# flow_spec = importlib.util.spec_from_file_location("config", flow_config_path)

# flow_config = importlib.util.module_from_spec(flow_spec)
# flow_spec.loader.exec_module(flow_config)

# flow_cfg = flow_config.cfg

if cfg.log.wandb:
    # wandb.tensorboard.patch(root_logdir=f"logs/{start_time_log}")

    run = wandb.init(
        project=cfg.log.wandb_project, 
        save_code=True,
        mode=cfg.log.wandb_mode,
        name=f"{start_time_log}_awac_finetuning_odpr",
        dir=f"wandb/awac_finetuning_odpr",
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
capacity = 10000
sampler = ODPRSampler(capacity=capacity)
buffer = TensorDictReplayBuffer(storage=ListStorage(capacity), sampler=sampler)

critic_optimizer = torch.optim.Adam(model.critic.parameters(), lr=cfg.train.lr)
actor_optimizer = torch.optim.Adam(model.actor.parameters(), lr=cfg.train.lr)
model.load_model(model_path)
model.to(device)


trainer = AWAC(
    model=model,
    replay_buffer=buffer,
    actor_optimizer=actor_optimizer,
    critic_optimizer=critic_optimizer,
    batch_size=cfg.train.batch_size,
)

wandb.watch(model.actor, log="all", log_freq=100)
wandb.watch(model.critic, log="all", log_freq=100)

######### flow ##########
flow = GraphSituationFlow(
    obs_dim=cfg.model.obs_dim,
    h_dim=cfg.model.h_dim,
    n_flow_blocks=cfg.model.n_flow_blocks,
    n_flow_hidden_num=cfg.model.n_flow_hidden_num,
    threshold_type=cfg.model.threshold_type,
)
flow.load_model(flow_model_path)
flow.to(device)


# if not cfg.train.onpolicy_finetuning:
# expl_logs = expl.exploration_k_ep_orca(
#     buffer=buffer,
#     k=10,#cfg.train.preliminary_exp_n,
#     scenario=cfg.sim.train_scenario,
#     human_num=cfg.sim.human_num,
#     policy=cfg.humans.policy,
#     # k=100,
#     render=False,
# )

# if cfg.train.pre_explor:
with tqdm(range(100)) as pbar:
    for i in enumerate(pbar):
        expl_logs = expl.exploration_k_ep_with_flow_mode(
            buffer=buffer,
            flow=flow,
            model=model,
            scenario=cfg.sim.val_scenario,
            human_num=cfg.sim.human_num,
            policy=cfg.humans.test_policy,
            pbar=pbar,
            mode=cfg.train.finetune_mode,
            render=False,
        )



loss_list = []

val_logs = eval_policy(
    eval_env=env,
    model=model,
    transfunc=transfunc,
    convert_action=convert_action,
    scenario=cfg.sim.val_scenario,
    human_num=cfg.sim.human_num,
    policy=cfg.humans.test_policy,
    eval_episodes=100,#env.case_size["test"],
    phase="test",
    render=False,
    print_results=True
)
odpr = False
weight_history = torch.ones(capacity, dtype=torch.float64)
max_cdr = float("-inf")
with tqdm(range(cfg.train.num_odpr_update), desc=trainer.alg_name + " Training") as pbar:
    for i, ch in enumerate(pbar):
        if odpr:
            weight_history = calculate_odpr_weights(model, buffer, weight_history)
            buffer.sampler.apply_odpr_weights(weight_history, len(buffer))
        with tqdm(range(cfg.train.total_it), desc="Online Fine tuning") as pbar:
            for j, ch in enumerate(pbar):
                trainer.update(
                    update_actor=((cfg.train.total_it % cfg.train.actor_update_interval) == 0),
                    data_for_logging=(run, i*cfg.train.total_it + j + 1) if cfg.log.wandb else None,
                )

                # total_it += 1
                if ((j + 1) % cfg.train.target_update_interval) == 0:
                    trainer.update_target()

                if (j + 1) % cfg.eval.eval_interval == 0:
                    val_logs = eval_policy(
                        eval_env=env,
                        model=model,
                        transfunc=transfunc,
                        scenario=cfg.sim.val_scenario,
                        human_num=cfg.sim.human_num,
                        policy=cfg.humans.test_policy,
                        convert_action=convert_action,
                        eval_episodes=env.case_size["val"],
                        phase="val",
                        render=cfg.eval.val_render,
                        render_type=cfg.eval.render_type,
                        path="trajs/",
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
                            step=i*cfg.train.total_it + j + 1,
                        )

                        val_log_data = [i*cfg.train.total_it + j + 1] + list(val_logs)
                        val_table.add_data(*val_log_data)
                        if j + 1 == cfg.train.total_it:
                            run.log({"Validation Table": val_table})

                    update_best = val_logs[1] > max_cdr
                    if update_best:
                        best_model = copy.deepcopy(model.state_dict())
                        best_step_num = i*cfg.train.total_it + j + 1
                        max_cdr = val_logs[1]

                    if cfg.log.save_model:
                        model.save_model(trained_models_dir + "/model_{}.pth".format(i*cfg.train.total_it + j + 1))
                        if update_best:
                            model.save_model(trained_models_dir + "/model_best.pth")


if cfg.log.wandb:
    # run.log({"Validation Table": val_table})

    test_log_columns = ["bset_step_num"] + results_log_columns
    # test_log_data = [best_step_num] + list(test_logs)
    test_table = wandb.Table(columns=test_log_columns)
    # test_table.add_data(*test_log_data)

    run.log({"Test Table": test_table})
    wandb.finish()