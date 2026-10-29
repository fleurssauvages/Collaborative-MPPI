"""Cartesian PID on the fingertip position of one arm.

The PID outputs a Cartesian velocity, v = Kp e + Ki int(e) - Kd xdot (derivative
on the measured velocity, so target jumps do not kick), mapped to joint
velocities by damped least squares with a null-space pull towards home. The
joint-position command integrates that velocity, which suits the position
actuators of the Panda model.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..robot import HOME_Q, DualFranka


@dataclass
class PIDConfig:
    kp: float = 3.0           # 1/s
    ki: float = 0.5           # 1/s^2
    kd: float = 0.1           # dimensionless
    max_speed: float = 0.5    # m/s
    integral_limit: float = 0.05  # m s
    damping: float = 0.02     # damped least squares
    null_gain: float = 0.5    # 1/s, pull towards the home posture


class CartesianPID:
    def __init__(self, robot: DualFranka, arm: str, cfg: PIDConfig, control_dt: float):
        self.robot, self.arm, self.cfg, self.dt = robot, arm, cfg, control_dt
        self.integral = np.zeros(3)
        self.command = robot.q(arm)

    def reset_integral(self):
        self.integral[:] = 0.0

    def step(self, target: np.ndarray) -> np.ndarray:
        cfg, robot, arm = self.cfg, self.robot, self.arm
        J = robot.jacobian(arm)
        error = np.asarray(target) - robot.ee(arm)
        self.integral = np.clip(self.integral + error * self.dt, -cfg.integral_limit, cfg.integral_limit)
        v = cfg.kp * error + cfg.ki * self.integral - cfg.kd * (J @ robot.dq(arm))
        speed = np.linalg.norm(v)
        if speed > cfg.max_speed:
            v *= cfg.max_speed / speed
        jinv = J.T @ np.linalg.inv(J @ J.T + cfg.damping ** 2 * np.eye(3))
        null = (np.eye(7) - jinv @ J) @ (cfg.null_gain * (HOME_Q - robot.q(arm)))
        self.command = np.clip(self.command + (jinv @ v + null) * self.dt,
                               robot.ctrl_low[robot.arm_act[arm]], robot.ctrl_high[robot.arm_act[arm]])
        return self.command.copy()
