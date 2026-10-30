"""MPPI over one or both Panda arms with Cartesian proposals.

Each controlled arm receives three proposal channels, interpreted as Cartesian
fingertip offsets (metres) and mapped to joint-position offsets with the
Jacobian pseudo-inverse along the nominal plan, as in Franka_MPPI. With
upright_gain > 0 the mapping also holds the tool pointing straight down: the
Jacobian gains the two tilt rows, whose target turns the tool z axis back to
vertical by that fraction per update, and the samples cannot tilt it. With
tip_z set, the tip height is held there the same way (air hockey): the samples
only explore x and y, and each update removes tip_z_gain of the height error. With
posture_gain > 0 the joints are also pulled towards HOME_Q in the Jacobian
null space, so the redundancy cannot drift into folded configurations. Rollouts,
costs and the weighted update stay in actuator (joint-target) space. Rollouts
run in parallel with mujoco.rollout.

The task objective is a per-arm cost ``arm_cost(sensors, arm, goal) -> (rollouts,)``
on the rollout sensors (rollouts x horizon x nsensordata); the controller adds
the floor and arm-arm clearance penalties.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable

import mujoco
import mujoco.rollout
import numpy as np

from ..robot import ARMS, HOME_Q, DualFranka
from ..samplers import make_sampler
from .lbps import optimize_lbps_temperature, weighted_control_sequence


@dataclass
class MPPIConfig:
    sampler: str = 'spike'
    rollouts: int = 32
    horizon: int = 20
    control_dt: float = 0.01
    noise_std: float = 0.01        # Cartesian exploration, metres
    ik_rcond: float = 1e-4
    ik_damping: float = 0.0        # damped least squares instead of the pseudo-inverse when > 0
    upright_gain: float = 0.0      # per-update fraction of tool tilt removed (0: orientation free)
    tip_z: float | None = None     # fixed tip height, metres (None: free)
    tip_z_gain: float = 0.5        # per-update fraction of the tip height error removed
    posture_gain: float = 0.0      # per-update fraction of the null-space distance to HOME_Q removed
    lbps_delta: float = 0.95
    lambda_temperature: float = 1.0  # fallback when LBPS cannot adapt
    w_clearance: float = 1e3
    clearance: float = 0.15        # minimum distance between the arms' proxy points, metres
    w_floor: float = 1e3
    floor_z: float = 0.05          # lowest fingertip height (floor or table top), metres
    threads: int = 4
    seed: int = 0


ArmCost = Callable[[np.ndarray, str, object], np.ndarray]


class MPPIController:
    """Optimizes the joint targets of ``arms``; other actuators follow a given plan."""

    def __init__(self, robot: DualFranka, arms, cfg: MPPIConfig, arm_cost: ArmCost):
        self.robot, self.arms, self.cfg, self.arm_cost = robot, tuple(arms), cfg, arm_cost
        m = robot.model
        self.substeps = max(1, int(round(cfg.control_dt / m.opt.timestep)))
        self.sampler = make_sampler(cfg.sampler, cfg.rollouts, cfg.horizon, 3 * len(self.arms),
                                    dt=cfg.control_dt)
        self.rng = np.random.default_rng(cfg.seed)
        self.nominal = np.tile(robot.data.ctrl, (cfg.horizon, 1))
        self.scratch = mujoco.MjData(m)
        self.thread_data = [mujoco.MjData(m) for _ in range(max(1, cfg.threads))]
        self.pool = mujoco.rollout.Rollout(nthread=max(1, cfg.threads))
        self.info: dict = {}

    def close(self):
        self.pool.close()

    def _proposals(self, nominal: np.ndarray) -> np.ndarray:
        cfg, robot = self.cfg, self.robot
        xyz = cfg.noise_std * self.sampler(self.rng)
        cand = np.broadcast_to(nominal, (cfg.rollouts, *nominal.shape)).copy()
        upright = cfg.upright_gain > 0
        rows = 5 if upright else 3
        jac = {arm: np.empty((cfg.horizon, rows, 7)) for arm in self.arms}
        tilt = {arm: np.zeros((cfg.horizon, 2)) for arm in self.arms}
        lift = {arm: np.zeros(cfg.horizon) for arm in self.arms}
        down = np.array([0.0, 0.0, -1.0])
        for t in range(cfg.horizon):
            for arm in self.arms:
                self.scratch.qpos[robot.joint_qadr[arm]] = nominal[t, robot.arm_act[arm]]
            mujoco.mj_kinematics(robot.model, self.scratch)
            mujoco.mj_comPos(robot.model, self.scratch)
            for arm in self.arms:
                if upright:
                    jac[arm][t], z = robot.jacobian_upright(arm, self.scratch)
                    tilt[arm][t] = cfg.upright_gain * np.cross(z, down)[:2]
                if cfg.tip_z is not None:
                    lift[arm][t] = cfg.tip_z_gain * (cfg.tip_z - robot.ee(arm, self.scratch)[2])
                else:
                    jac[arm][t] = robot.jacobian(arm, self.scratch)
        for k, arm in enumerate(self.arms):
            if cfg.ik_damping > 0:  # bounded joint offsets near singular poses
                jt = jac[arm].transpose(0, 2, 1)
                jinv = jt @ np.linalg.inv(jac[arm] @ jt + cfg.ik_damping ** 2 * np.eye(rows))  # (H, 7, rows)
            else:
                jinv = np.linalg.pinv(jac[arm], rcond=cfg.ik_rcond)
            if cfg.tip_z is not None:
                xyz[:, :, 3 * k + 2] = lift[arm]
            cand[:, :, robot.arm_act[arm]] += np.einsum('tjc,ntc->ntj', jinv[:, :, :3], xyz[:, :, 3 * k:3 * k + 3])
            if upright:
                cand[:, :, robot.arm_act[arm]] += np.einsum('tjc,tc->tj', jinv[:, :, 3:], tilt[arm])
            if cfg.posture_gain > 0:
                null = np.eye(7) - jinv @ jac[arm]
                pull = cfg.posture_gain * (HOME_Q - nominal[:, robot.arm_act[arm]])
                cand[:, :, robot.arm_act[arm]] += np.einsum('tij,tj->ti', null, pull)
        return np.clip(cand, robot.ctrl_low, robot.ctrl_high)

    def _costs(self, sensors: np.ndarray, goal) -> np.ndarray:
        cfg, robot = self.cfg, self.robot
        cost = np.zeros(len(sensors))
        for arm in self.arms:
            cost += self.arm_cost(sensors, arm, goal)
            low = np.maximum(cfg.floor_z - sensors[:, :, robot.sensor_ee[arm]][..., 2], 0.0)
            cost += cfg.w_floor * np.sum(low ** 2, axis=1)
        # Arm-arm clearance between every pair of proxy points.
        a = sensors[:, :, robot.sensor_arm['left']].reshape(*sensors.shape[:2], -1, 1, 3)
        b = sensors[:, :, robot.sensor_arm['right']].reshape(*sensors.shape[:2], 1, -1, 3)
        dmin = np.sqrt(np.sum((a - b) ** 2, axis=-1)).min(axis=(2, 3))
        cost += cfg.w_clearance * np.sum(np.maximum(cfg.clearance - dmin, 0.0) ** 2, axis=1)
        return cost

    def step(self, goal, others: np.ndarray | None = None) -> np.ndarray:
        """One MPPI update; returns the plan (horizon x nu). ``goal`` is passed to
        the arm cost; ``others`` gives the actuator plan used for every actuator
        this controller does not optimize."""
        cfg, robot = self.cfg, self.robot
        tic = time.perf_counter()
        nominal = self.nominal.copy()
        if others is not None:
            mine = np.concatenate([robot.arm_act[arm] for arm in self.arms])
            keep = np.setdiff1d(np.arange(robot.nu), mine)
            nominal[:, keep] = others[:, keep]
        cand = self._proposals(nominal)
        controls = np.repeat(cand, self.substeps, axis=1)
        _, sensors = self.pool.rollout(robot.model, self.thread_data, robot.state()[None], controls)
        sensors = sensors[:, self.substeps - 1::self.substeps]
        costs = self._costs(sensors, goal)
        lbps = optimize_lbps_temperature(costs, delta=cfg.lbps_delta,
                                         fallback_temperature=cfg.lambda_temperature)
        plan = weighted_control_sequence(costs, cand, lbps.temperature)
        # Warm start: shift by one control step and repeat the last command.
        self.nominal = np.vstack([plan[1:], plan[-1:]])
        self.info = {'temperature': lbps.temperature, 'ess': lbps.ess,
                     'best_cost': float(np.min(costs)), 'compute_ms': 1e3 * (time.perf_counter() - tic)}
        return plan


class CollaborativeMPPI:
    """Centralized: one MPPI over both arms. Decentralized: one MPPI per arm, each
    rolling out the other arm along that arm's previous plan."""

    def __init__(self, robot: DualFranka, arms, cfg: MPPIConfig, arm_cost: ArmCost,
                 layout: str = 'centralized'):
        if layout not in ('centralized', 'decentralized'):
            raise ValueError("layout must be 'centralized' or 'decentralized'")
        self.robot, self.arms, self.layout = robot, tuple(arms), layout
        if layout == 'centralized' or len(self.arms) == 1:
            self.parts = [MPPIController(robot, self.arms, cfg, arm_cost)]
        else:
            self.parts = [MPPIController(robot, (arm,), MPPIConfig(**{**cfg.__dict__, 'seed': cfg.seed + k}),
                                         arm_cost)
                          for k, arm in enumerate(self.arms)]

    def step(self, goal, held_ctrl: np.ndarray) -> np.ndarray:
        """Returns the actuator command for the next control step. ``held_ctrl``
        is assumed constant over the horizon for arms MPPI does not control."""
        h = self.parts[0].cfg.horizon
        others = np.tile(held_ctrl, (h, 1))
        # Simultaneous update: every MPPI sees the other arms' previous plans.
        for part in self.parts:
            for other in self.parts:
                if other is not part:
                    for arm in other.arms:
                        act = self.robot.arm_act[arm]
                        others[:, act] = other.nominal[:, act]
        command = held_ctrl.copy()
        for part in self.parts:
            plan = part.step(goal, others)
            for arm in part.arms:
                command[self.robot.arm_act[arm]] = plan[0, self.robot.arm_act[arm]]
        return command

    @property
    def info(self) -> dict:
        return {f'{"+".join(p.arms)}_{k}': v for p in self.parts for k, v in p.info.items()}

    def close(self):
        for part in self.parts:
            part.close()
