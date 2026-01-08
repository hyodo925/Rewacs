import numpy as np

from rewacs.envs.utils.action import ActionRot, ActionXY, ActionXYW
from rewacs.envs.utils.agent import Agent
from rewacs.envs.utils.state import JointState, ObservableState, RobotFullState


class Robot(Agent):
    def __init__(self, config, section):
        super().__init__(config, section)
        self.v_yaw = None

    def get_distance(self, r: ObservableState):
        return (r.px - self.px) ** 2 + (r.py - self.py) ** 2

    def act(self, ob):
        if self.policy is None:
            raise AttributeError("Policy attribute has to be set!")

        state = JointState(self.get_full_state(), ob)
        action = self.policy.predict(state)
        return action

    def get_state(self, ob):
        state = JointState(self.get_full_state(), ob)
        if self.policy is None:
            raise AttributeError("Policy attribute has to be set!")
        return self.policy.transform(state)

    def set(self, px, py, gx, gy, vx, vy, v_yaw, theta, radius=None, v_pref=None):
        self.px = px
        self.py = py
        self.sx = px
        self.sy = py
        self.gx = gx
        self.gy = gy
        self.vx = vx
        self.vy = vy
        self.v_yaw = v_yaw
        self.theta = theta
        if radius is not None:
            self.radius = radius
        if v_pref is not None:
            self.v_pref = v_pref

    def get_full_state(self):
        return RobotFullState(
            self.px,
            self.py,
            self.vx,
            self.vy,
            self.v_yaw,
            self.radius,
            self.gx,
            self.gy,
            self.v_pref,
            self.theta,
        )

    def get_position(self):
        return self.px, self.py

    def set_position(self, position):
        self.px = position[0]
        self.py = position[1]

    def get_goal_position(self):
        return self.gx, self.gy

    def get_start_position(self):
        return self.sx, self.sy

    def get_velocity(self):
        return self.vx, self.vy, self.v_yaw

    def set_velocity(self, velocity):
        self.vx = velocity[0]
        self.vy = velocity[1]
        self.v_yaw = velocity[2]

    def check_validity(self, action):
        if self.kinematics == "holonomic":
            assert isinstance(action, ActionXY) or isinstance(action, ActionXYW)
        else:
            assert isinstance(action, ActionRot)

    def compute_position(self, action, delta_t):
        self.check_validity(action)
        if self.kinematics == "holonomic":
            # px = self.px + action.vx * delta_t
            # py = self.py + action.vy * delta_t
            if isinstance(action, ActionXYW):
                theta = self.theta + action.vw * delta_t
            else:
                theta = self.theta
            px = (
                self.px
                + (action.vx * np.cos(theta) - action.vy * np.sin(theta)) * delta_t
            )
            py = (
                self.py
                + (action.vx * np.sin(theta) + action.vy * np.cos(theta)) * delta_t
            )
        else:
            theta = self.theta + action.r
            px = self.px + np.cos(theta) * action.v * delta_t
            py = self.py + np.sin(theta) * action.v * delta_t

        return px, py

    def step(self, action):
        """
        Perform an action and update the state
        """
        self.check_validity(action)
        pos = self.compute_position(action, self.time_step)
        self.px, self.py = pos
        if self.kinematics == "holonomic":
            # self.vx = action.vx
            # self.vy = action.vy
            if isinstance(action, ActionXYW):
                self.theta = (self.theta + action.vw * self.time_step) % (2 * np.pi)
            self.vx = action.vx * np.cos(self.theta) - action.vy * np.sin(self.theta)

            self.vy = action.vx * np.sin(self.theta) + action.vy * np.cos(self.theta)

        else:
            self.theta = (self.theta + action.r) % (2 * np.pi)
            self.vx = action.v * np.cos(self.theta)
            self.vy = action.v * np.sin(self.theta)
