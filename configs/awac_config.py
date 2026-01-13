import numpy as np
from flexible_config.flexible_config_builder import FlexibleConfigBuilder


class Environment:
    time_limit = 30
    time_step = 0.25
    val_size = 100
    test_size = 500
    train_size = np.iinfo(np.uint32).max - 2000
    randomize_attributes = False
    robot_sensor_range = 5


class Reward:
    success_reward = 1
    collision_penalty = -0.25
    discomfort_dist = 0.2
    discomfort_penalty_factor = 0.5


class Simulation:
    train_val_scenario = "square_crossing"
    test_scenario = "circle_crossing"
    # train_val_scenario = "square_crossing"
    # test_scenario = "square_crossing"

    # rain_val_scenario = "corridor"
    # test_scenario = "corridor"

    # train_val_scenario = "monotonic"
    # test_scenario = "monotonic"

    # train_val_scenario = "alone"
    # test_scenario = "alone"

    square_width = 20
    circle_radius = 4
    human_num = 10
    nonstop_human = False
    centralized_planning = True
    random_p_num = False


class Humans:
    visible = True
    policy = "orca"
    radius = 0.3
    v_pref = 1
    sensor = "coordinates"


class Robot:
    visible = True
    policy = "orca_rc"
    radius = 0.3
    v_pref = 1
    sensor = "coordinates"


class Model:
    use_actor = True
    stochastic_actor = True
    projection_dim = 32
    gc = True
    obs_dim = 4
    r_obs_dim = 5
    action_dim = 2
    action_space = [-1.0, 1.0]
    max_action = 1.0

    actor_h_dims = [100, 100]
    critic_h_dims = [100, 100]

    actor_integrator_enc_hdims = [64]
    critic_integrator_enc_hdims = [64]


class Transfunc:
    with_peds_vel = True
    peds_vel_as_relative = True
    use_omega = True


class Train:
    random_seed = 17
    offline_learning = True
    lr = 3e-4
    preliminary_exp_n = 200
    total_it = 10000
    batch_size = 100
    buffer_capacity = 100000
    actor_update_interval = 2
    target_update_interval = 1
    polyak = 0.995
    training_alg = "AWAC"


class Evaluation:
    eval_interval = 1000
    final_eval_num = 500
    val_render = False
    render = False
    render_type = "video"


class Log:
    wandb_project = "AWAC_training"
    # wandb_mode = "offline"
    wandb_mode = "online"
    wandb = True
    save_model = True


b = FlexibleConfigBuilder()
b.add_from_class("env", Environment)
b.add_from_class("reward", Reward)
b.add_from_class("sim", Simulation)
b.add_from_class("humans", Humans)
b.add_from_class("robot", Robot)

b.add_from_class("model", Model)
b.add_from_class("transfunc", Transfunc)
b.add_from_class("train", Train)
b.add_from_class("eval", Evaluation)
b.add_from_class("log", Log)

cfg = b.parse()
