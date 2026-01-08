import numpy as np
from nested_config.nested_config_builder import NestedConfigBuilder


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
    test_scenario = "square_crossing"
    square_width = 20
    circle_radius = 4
    human_num = 5
    nonstop_human = False
    centralized_planning = True


class Humans:
    visible = True
    policy = "orca"
    radius = 0.3
    v_pref = 1
    sensor = "coordinates"


class Robot:
    visible = True
    policy = "orca"
    radius = 0.3
    v_pref = 1
    sensor = "coordinates"


b = NestedConfigBuilder()
b.add_from_class("env", Environment)
b.add_from_class("reward", Reward)
b.add_from_class("sim", Simulation)
b.add_from_class("humans", Humans)
b.add_from_class("robot", Robot)

cfg = b.parse()
