import math

import torch


def wrap_to_pi(phi):
    return (phi + math.pi) % (2 * math.pi) - math.pi


def rotmat_minus_theta(theta):
    c = torch.cos(theta)
    s = torch.sin(theta)

    R = torch.stack(
        [
            torch.stack([c, -s], dim=-1),
            torch.stack([s, c], dim=-1),
        ],
        dim=-2,
    )
    return R


def world_to_rf_points(points_world, robot_pos_world, robot_theta):
    R = rotmat_minus_theta(robot_theta)
    shifted = points_world - robot_pos_world
    rf_points = torch.matmul(shifted.unsqueeze(-2), R).squeeze(-2)
    return rf_points


def world_to_rf_vel(vel_world, robot_theta):
    R = rotmat_minus_theta(robot_theta)
    rf_vel = torch.matmul(vel_world.unsqueeze(-2), R).squeeze(-2)
    return rf_vel


def world_to_rf_rel_vel(
    obj_pos_world,
    obj_vel_world,
    robot_pos_world,
    robot_vel_world,
    robot_theta,
    robot_omega=None,
    use_omega=False,
):
    r_world = obj_pos_world - robot_pos_world
    v_rel_world = obj_vel_world - robot_vel_world

    if use_omega:
        if robot_omega is None:
            raise ValueError("If use_omega=True, please provide robot_omega.")
        omega = robot_omega.unsqueeze(-1)
        omega_cross_r = torch.stack(
            [-omega[..., 0] * r_world[..., 1], omega[..., 0] * r_world[..., 0]], dim=-1
        )
        v_rel_world = v_rel_world - omega_cross_r

    return world_to_rf_vel(v_rel_world, robot_theta)


def bearing_to_point_in_robot_frame(point_world, robot_pos_world, robot_theta):
    p_b = world_to_rf_points(point_world, robot_pos_world, robot_theta)
    ang = torch.atan2(p_b[..., 1], p_b[..., 0])
    return wrap_to_pi(ang)


class GetRobotFrameObs:
    def __init__(self, with_peds_vel=True, peds_vel_as_relative=True, use_omega=True):
        super().__init__()

        self.with_peds_vel = with_peds_vel
        self.peds_vel_as_relative = peds_vel_as_relative
        self.use_omega = use_omega

    def __call__(
        self,
        robot_state,
        human_states,
    ):
        robot_state_tensor = torch.Tensor([robot_state.to_tuple()])
        human_states_tensor = torch.Tensor(
            [human_state.to_tuple() for human_state in human_states]
        )

        goal = robot_state_tensor[:, 6:8]
        robot_pos = robot_state_tensor[:, :2]
        robot_theta = robot_state_tensor[:, -1].reshape(-1, 1)
        robot_vel_xy = robot_state_tensor[:, 2:4]
        robot_vel_omega = robot_state_tensor[:, 4]
        rc_goal = world_to_rf_points(goal, robot_pos, robot_theta).squeeze(0)
        rc_goal_theta = bearing_to_point_in_robot_frame(goal, robot_pos, robot_theta)
        rc_r_vel = world_to_rf_vel(robot_vel_xy, robot_theta).squeeze(0)

        huma_pos = human_states_tensor[:, :2]
        huma_vel = human_states_tensor[:, 2:4]
        huma_pos_rc = world_to_rf_points(huma_pos, robot_pos, robot_theta).squeeze(0)

        if self.peds_vel_as_relative:
            huma_vel_rc = world_to_rf_rel_vel(
                obj_pos_world=huma_pos,
                obj_vel_world=huma_vel,
                robot_pos_world=robot_pos,
                robot_vel_world=robot_vel_xy,
                robot_theta=robot_theta,
                robot_omega=robot_vel_omega,
                use_omega=self.use_omega,
            ).squeeze(0)
        else:
            huma_vel_rc = world_to_rf_vel(huma_vel, robot_theta).squeeze(0)

        robot_obs = torch.cat(
            [rc_goal, rc_goal_theta, rc_r_vel],
            dim=1,
        )

        if self.with_peds_vel:
            human_obs = torch.cat(
                [huma_pos_rc, huma_vel_rc],
                dim=1,
            )
        else:
            human_obs = torch.cat(
                [huma_pos_rc],
                dim=1,
            )

        return robot_obs, human_obs
