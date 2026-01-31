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
import matplotlib
matplotlib.use("Agg")
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
from algo.meta_critic.eval import eval_policy
from algo.meta_critic.explorer import ExplorerCrowdSim
from utils.models import (
    SocialCritic,
)
from utils.state_integrators import (
    EmbeddedGaussianIntegrator,
)
from algo.awac.trainer import AWAC
from algo.awac.actor import SocialActorAWAC
from flow.model import GraphSituationFlow
from algo.meta_critic.actor import SocialActorMetaCriticAWAC
from algo.meta_critic.trainer_hotplug import MetaCriticAWAC
from algo.meta_critic.meta_critic import MetaCriticNet, MetaCriticGraphNet
from meta_rl_navigation import MetaRLNavigation

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
# run_dir = "wandb/awac_training/wandb/run-20260124_141502-b6znpdu2"
run_dir = "wandb/awac_training/wandb/run-20260127_123238-tac0gepj"
config_path = "./configs/meta_critic_awac_with_flow_config.py"

model_path = os.path.join(run_dir, "files/trained_models/model_best.pth")
#################################

############# flow model ####################
# Settings
flow_run_dir = "wandb/Switching_Administrator_training/wandb/run-20260121_142518-hifx4rkp"

flow_config_path = os.path.join(flow_run_dir, "files/config.py")

flow_model_path = os.path.join(flow_run_dir, "files/trained_models/model_500.pth")

render = False
render_type = "video"

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
        name=f"{start_time_log}_awac_meta_critic_finetuning/seed{str(cfg.train.random_seed)}",
        dir=f"wandb/awac_meta_critic_finetuning/seed{str(cfg.train.random_seed)}",
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

actor = SocialActorMetaCriticAWAC(
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

# meta_critic = MetaCriticNet(cfg.model.meta_critic_integrator_enc_hdims)
meta_critic = MetaCriticGraphNet(    
    cfg.model.projection_dim + cfg.model.action_dim + cfg.model.other_output_dim,
    1,
    h_dims=cfg.model.critic_h_dims,
    integrator=critic_integrator,
    single=False,)

model = MetaRLNavigation(actor=actor, critic=critic, meta_critic=meta_critic)

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
buffer_val = ReplayBuffer(storage=LazyTensorStorage(cfg.train.buffer_capacity))

critic_optimizer = torch.optim.Adam(model.critic.parameters(), lr=cfg.train.lr)
actor_optimizer = torch.optim.Adam(model.actor.parameters(), lr=cfg.train.lr)
meta_optimizer = torch.optim.Adam(model.meta_critic.parameters(), lr=cfg.train.lr)
checkpoint = torch.load(model_path, map_location=device)
model.load_state_dict(checkpoint, strict=False)
model.to(device)

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


trainer = MetaCriticAWAC(
    model=model,
    flow=flow,
    replay_buffer=buffer,
    replay_buffer_val=buffer_val,
    actor_optimizer=actor_optimizer,
    critic_optimizer=critic_optimizer,
    meta_critic_optimizer=meta_optimizer,
    batch_size=cfg.train.batch_size,
)

# if not cfg.train.onpolicy_finetuning:
#     expl_logs = expl.exploration_k_ep_orca(
#         buffer=buffer,
#         k=cfg.train.preliminary_exp_n,
#         scenario=cfg.sim.train_scenario,
#         human_num=cfg.sim.human_num,
#         policy=cfg.humans.policy,
#         # k=100,
#         render=False,
#     )

# if cfg.train.pre_explor:
#     with tqdm(range(cfg.train.pre_explor_itr)) as pbar:
#         for i in enumerate(pbar):
#             expl_logs = expl.exploration_k_ep_with_flow_mode(
#                 buffer=buffer,
#                 flow=flow,
#                 model=model,
#                 scenario=cfg.sim.val_scenario,
#                 human_num=cfg.sim.human_num,
#                 policy=cfg.humans.test_policy,
#                 pbar=pbar,
#                 mode=cfg.train.finetune_mode,
#                 render=False,
#             )

if cfg.train.pre_explor:
    with tqdm(range(cfg.train.pre_explor_itr)) as pbar:
        for i in enumerate(pbar):
            expl_logs = expl.exploration_k_ep_with_switching(
                buffer=buffer,
                buffer_val=buffer_val,
                flow=flow,
                model=model,
                scenario=cfg.sim.val_scenario,
                human_num=cfg.sim.human_num,
                policy=cfg.humans.test_policy,
                pbar=pbar,
                render=False,
            )

loss_list = []
val_logs = eval_policy(
    eval_env=env,
    model=model,
    transfunc=transfunc,
    scenario=cfg.sim.val_scenario,
    human_num=cfg.sim.human_num,
    policy=cfg.humans.finetune_policy,
    convert_action=convert_action,
    eval_episodes=env.case_size["val"],
    phase="val",
    render=cfg.eval.val_render,
    render_type=cfg.eval.render_type,
    print_results=True,
    path=f"trajs/awac_meta_critic_finetuning/seed{str(cfg.train.random_seed)}/human{str(cfg.sim.human_num)}/{cfg.train.finetune_mode}/itr0",
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
        step=0,
    )

    val_log_data = [0] + list(val_logs)
    val_table.add_data(*val_log_data)
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
with tqdm(range(cfg.train.finetune_total_it), desc=trainer.alg_name + " Training") as pbar:
    for i, ch in enumerate(pbar):
        with torch.no_grad():
            # if i < 1000:
            
            if not cfg.train.offline_learning:
                for j in range(cfg.train.fintuning_rollout_itr):
                    # expl_logs = expl.exploration_k_ep_with_flow_mode(
                    #     buffer=buffer,
                    #     flow=flow,
                    #     model=model,
                    #     scenario=cfg.sim.val_scenario,
                    #     human_num=cfg.sim.human_num,
                    #     policy=cfg.humans.test_policy,
                    #     pbar=pbar,
                    #     mode=cfg.train.finetune_mode,
                    #     render=False,
                    # )
                    expl_logs = expl.exploration_k_ep_with_switching(
                        buffer=buffer,
                        buffer_val=buffer_val,
                        flow=flow,
                        model=model,
                        scenario=cfg.sim.val_scenario,
                        human_num=cfg.sim.human_num,
                        policy=cfg.humans.test_policy,
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

        trainer.finetune(
            # update_actor=((cfg.train.total_it % cfg.train.actor_update_interval) == 0),
            data_for_logging=(run, i + 1) if cfg.log.wandb else None,
        )

        # total_it += 1
        if ((i + 1) % cfg.train.target_update_interval) == 0:
            trainer.update_target()

        if (i + 1) % cfg.eval.finetune_interval == 0:
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
                path=f"trajs/awac_meta_critic_finetuning/seed{str(cfg.train.random_seed)}/human{str(cfg.sim.human_num)}/{cfg.train.finetune_mode}/itr{str(i+1)}",
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
                if i + 1 == cfg.train.finetune_total_it:
                    run.log({"Validation Table": val_table})

if cfg.log.wandb:
    wandb.finish()