"""Grasp setup and shared MPPI cost for two-arm tray balancing.

The arms hold a large tray by handles at two diagonally opposite corners (the left
arm the (-x, -y) corner, the right arm the (+x, +y) one) and keep a ball rolling on
it at the tray centre. Each grasp is a weld between the hand and the tray, set from
the initial posture, in which both grippers point down with the fingers closing
across the handle. The handles alone leave the tray free to turn about the
diagonal, so only the wrists can tilt it that way: run.py lets the MPPI explore
tool rotation as well as position, and holds no orientation.

Both arms minimize the same cost: the ball-on-plate objective of Franka_MPPI's
single-arm franka_balancing task (Zarzycki & Lawrynczuk 2021, Sensors 21(12) 3959,
Eq. 23, extended there with velocity, tilt, air gap, terminal and fall terms), on
the tray instead of a plate carried by one arm. Its control increment and proposal
terms are left out; a small term keeps the tray near its start position, which one
arm alone no longer fixes (run.py also sets the controller's joint lag penalty, which
keeps the arms from pulling against each other through the tray). The cost reads only the ball and the tray, never the arm
it is called for, so centralized MPPI scores both arms with it and each
decentralized MPPI optimizes it for the team.
"""
from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np

from ..robot import DualFranka

ARM_CORNER = {'left': np.array([-1.0, -1.0]), 'right': np.array([1.0, 1.0])}  # corner sign along x, y
HANDLE_GRASP = 0.035  # grasp point on the handle, distance out from the tray corner, m
HANDLE_R = 0.012      # handle radius, m


@dataclass
class BalanceConfig:
    # Weights of Franka_MPPI's BalanceWeights.
    w_position: float = 3.0    # ball offset, normalized by position_scale
    position_scale: float = 0.092  # usable half-width of Franka_MPPI's plate, m: the tray is larger,
                                   # and normalizing by its own size would make centring several times weaker
    w_velocity: float = 0.8    # ball speed on the tray, normalized by v_scale
    v_scale: float = 0.3       # m/s
    w_tilt: float = 0.06       # ((1 - n_z) / 0.02)^2, n the tray normal
    w_air_gap: float = 2.0     # ball lifted off the tray beyond 0.015 m, normalized by 0.03 m
    w_terminal: float = 2.0    # position and speed at the end of the horizon
    w_fall: float = 30.0       # share of the horizon with the ball lost, plus half if lost at all
    w_drift: float = 1.0       # tray centre displacement from its start, normalized by drift_scale
    drift_scale: float = 0.05  # m


class BalanceTask:
    def __init__(self, robot: DualFranka, cfg: BalanceConfig):
        self.robot, self.cfg = robot, cfg
        m = robot.model
        self.ball_pos = robot.sensor_slice('ball_pos')
        self.ball_vel = robot.sensor_slice('ball_vel')
        self.tray_pos = robot.sensor_slice('tray_pos')
        self.tray_normal = robot.sensor_slice('tray_normal')
        tray = m.geom('tray')
        self.half = tray.size[:2].copy()
        self.half_thick = float(tray.size[2])
        self.ball_r = float(m.geom('ball').size[0])
        self.usable = self.half - self.ball_r
        self.tray_qadr, self.tray_dadr = m.joint('tray').qposadr[0], m.joint('tray').dofadr[0]
        self.ball_qadr, self.ball_dadr = m.joint('ball').qposadr[0], m.joint('ball').dofadr[0]
        self.start = m.body('tray').pos.copy()

    # Grasp ----------------------------------------------------------------------
    def grasp_pose(self, arm: str) -> tuple[np.ndarray, np.ndarray]:
        """Fingertip point on the arm's handle and gripper orientation, with the tray at its start."""
        corner = ARM_CORNER[arm] * self.half
        out = corner / np.linalg.norm(corner)
        point = self.start + np.append(corner + HANDLE_GRASP * out, 0.0)
        across = np.array([-out[1], out[0], 0.0])  # finger closing axis (tool y)
        down = np.array([0.0, 0.0, -1.0])
        return point, np.column_stack([np.cross(across, down), across, down])

    def grasp(self, iterations: int = 500, damping: float = 0.05, settle: float = 0.5) -> np.ndarray:
        """Puts both grippers on their handles (damped least squares IK from the current
        posture), closes them on the handle, welds the hands to the tray there and lets
        the tray settle. Returns the left arm's posture, the MPPI home posture (the layout is symmetric)."""
        m, d, robot = self.robot.model, self.robot.data, self.robot
        for arm in ('left', 'right'):
            point, rot_goal = self.grasp_pose(arm)
            for _ in range(iterations):
                mujoco.mj_kinematics(m, d)
                mujoco.mj_comPos(m, d)
                jacp, jacr, rot = robot.jacobian_rot(arm)
                e = rot_goal @ rot.T
                err = np.concatenate([point - robot.ee(arm),
                                      0.5 * np.array([e[2, 1] - e[1, 2], e[0, 2] - e[2, 0], e[1, 0] - e[0, 1]])])
                jac = np.vstack([jacp, jacr])
                d.qpos[robot.joint_qadr[arm]] += jac.T @ np.linalg.solve(jac @ jac.T + damping ** 2 * np.eye(6), err)
            d.ctrl[robot.arm_act[arm]] = d.qpos[robot.joint_qadr[arm]]
            # Fingers at the handle surface: the servo sets each finger at 0.04 * ctrl / 255.
            d.qpos[robot.finger_qadr[arm]] = HANDLE_R
            d.ctrl[robot.grip_act[arm]] = 255.0 * HANDLE_R / 0.04
        mujoco.mj_forward(m, d)
        tray = m.body('tray').id
        for arm in ('left', 'right'):
            eq = m.equality(f'{arm}_grasp').id
            hand = m.body(f'{arm}_hand').id
            # Weld point (the grasp point) in the tray frame, then in the hand frame,
            # then the tray orientation relative to the hand.
            rel_quat = np.empty(4)
            mujoco.mju_negQuat(rel_quat, d.xquat[hand])
            mujoco.mju_mulQuat(rel_quat, rel_quat, d.xquat[tray])
            m.eq_data[eq, 0:3] = d.xmat[tray].reshape(3, 3).T @ (robot.ee(arm) - d.xpos[tray])
            m.eq_data[eq, 3:6] = d.xmat[hand].reshape(3, 3).T @ (robot.ee(arm) - d.xpos[hand])
            m.eq_data[eq, 6:10] = rel_quat
        mujoco.mj_forward(m, d)
        for _ in range(int(round(settle / m.opt.timestep))):
            mujoco.mj_step(m, d)
        return robot.q('left')

    def place_ball(self, offset, velocity):
        """Ball on the tray at ``offset`` (tray x, y) from its centre, rolling at ``velocity``."""
        m, d = self.robot.model, self.robot.data
        rot = d.xmat[m.body('tray').id].reshape(3, 3)
        pos = d.xpos[m.body('tray').id] + rot @ np.array([*offset, self.half_thick + self.ball_r + 0.0005])
        d.qpos[self.ball_qadr:self.ball_qadr + 7] = (*pos, 1.0, 0.0, 0.0, 0.0)
        d.qvel[self.ball_dadr:self.ball_dadr + 6] = 0.0
        d.qvel[self.ball_dadr:self.ball_dadr + 3] = rot[:, :2] @ np.asarray(velocity, dtype=float)
        mujoco.mj_forward(m, d)

    # State ------------------------------------------------------------------------
    def ball(self, data=None) -> tuple[np.ndarray, np.ndarray, float]:
        """Ball offset (x, y) and velocity (x, y) in the tray frame, and its gap above the tray, m."""
        s = (self.robot.data if data is None else data).sensordata
        rel = s[self.ball_pos]
        return rel[:2].copy(), s[self.ball_vel][:2].copy(), float(rel[2] - self.half_thick - self.ball_r)

    def tilt(self, data=None) -> float:
        """Tray tilt from horizontal, rad."""
        nz = (self.robot.data if data is None else data).sensordata[self.tray_normal][2]
        return float(np.arccos(np.clip(nz, -1.0, 1.0)))

    def fallen(self, data=None) -> bool:
        return self.ball(data)[2] < -0.05

    # MPPI cost ----------------------------------------------------------------------
    def cost(self, sensors: np.ndarray, arm: str, goal=None) -> np.ndarray:
        """Shared ball-on-tray cost; the same for both arms (``arm`` is not used)."""
        cfg = self.cfg
        rel = sensors[:, :, self.ball_pos]
        r, v = rel[..., :2], sensors[:, :, self.ball_vel][..., :2]
        gap = rel[..., 2] - self.half_thick - self.ball_r
        nz = sensors[:, :, self.tray_normal][..., 2]
        pos = np.sum((r / cfg.position_scale) ** 2, axis=-1)
        speed = np.sum((v / cfg.v_scale) ** 2, axis=-1)
        tilt = ((1.0 - nz) / 0.02) ** 2
        air_gap = (np.maximum(gap - 0.015, 0.0) / 0.03) ** 2
        drift = np.sum(((sensors[:, :, self.tray_pos] - self.start) / cfg.drift_scale) ** 2, axis=-1)
        lost = np.any(np.abs(r) > self.usable, axis=-1) | (gap < -0.035) | (gap > 0.085)
        running = (cfg.w_position * pos + cfg.w_velocity * speed + cfg.w_tilt * tilt
                   + cfg.w_air_gap * air_gap + cfg.w_drift * drift)
        terminal = cfg.w_terminal * (pos[:, -1] + speed[:, -1])
        loss = cfg.w_fall * (np.mean(lost, axis=1) + 0.5 * np.any(lost, axis=1))
        return np.mean(running, axis=1) + terminal + loss
