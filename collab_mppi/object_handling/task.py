"""Phases, MPPI cost and bookkeeping for the object handling task.

Robot 1 (the right arm, the giver) picks each box from the start table on its
right and holds it up in the air between the robots; robot 2 (the left arm, the
taker) grasps the other end of the box there, robot 1 lets go, and robot 2 puts
the box on its slot on the goal table on its left. Boxes go in order 0, 1, 2;
robot 1 fetches the next box while robot 2 places the previous one.

Each arm runs a small phase machine. A phase gives the fingertip target and the
gripper command; MPPI tracks the target (the gripper command is held over the
rollouts). Targets on a resting box follow the box; targets while carrying are
frozen when the phase starts, so they do not run away with the box. MPPI tracks
a setpoint that walks towards the phase target at v_free, or at v_held while the
gripper is closed, so a held box is not swung out of the fingers; phases end on
the fingertip error to the target itself.

giver: pick_above -> pick_down -> pick_close -> lift -> to_handover -> hold
       (until the taker has closed) -> release -> retreat -> next box
taker: wait (until the giver holds the box at the handover) -> take_above ->
       take_down -> take_close -> take_hold (until the giver has let go) ->
       carry -> place -> open -> retreat -> wait

Both grippers point straight down with the fingers closing along x; run.py
locks that orientation in the MPPI. A box that falls (below the table tops while
not held) is a drop: it goes back to its start position and both arms restart it.
"""
from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np

from ..robot import GRIP_CLOSED, GRIP_OPEN, DualFranka

GIVER, TAKER = 'right', 'left'
TABLE_TOP = 0.20
BOX_HALF = np.array([0.02, 0.12, 0.02])
N_BOXES = 3
# Tool orientation for both arms: pointing down, fingers closing along world x.
GRASP_ROT = np.array([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, -1.0]])
HOME_Q_GRASP = np.array([0.0, -0.3, 0.0, -2.2, 0.0, 1.9, -0.785])  # HOME_Q turned to GRASP_ROT
BASES = {'left': ((0.0, 0.45, 0.0), 0.0), 'right': ((0.0, -0.45, 0.0), 0.0)}


@dataclass
class HandlingConfig:
    handover: tuple = (0.52, 0.0, 0.40)  # box centre at the handover, m
    grasp_offset: float = 0.07  # each gripper holds the box this far from its centre along y, m
    approach: float = 0.10     # height above a grasp for the approach, m
    lift: float = 0.12         # lift after a grasp or a release, m
    place_clearance: float = 0.005  # box lowered to this height above the goal table, m
    tol_move: float = 0.02     # fingertip error that ends a free move, m
    tol_fine: float = 0.008    # fingertip error that ends a move onto a grasp or a placement, m
    t_close: float = 0.5       # gripper closing time, s
    t_open: float = 0.4        # gripper opening time, s
    v_free: float = 0.8        # setpoint speed with the gripper open, m/s
    v_held: float = 0.35       # setpoint speed with the gripper closed, m/s
    dt: float = 0.01           # control step, s
    w_target: float = 1.0
    w_terminal: float = 10.0


class ArmPhase:
    """Current phase of one arm, its target and gripper command."""

    def __init__(self, phase: str, t: float = 0.0):
        self.set(phase, t)

    def set(self, phase: str, t: float, frozen: np.ndarray | None = None):
        self.phase, self.since, self.frozen = phase, t, frozen


class HandlingTask:
    def __init__(self, robot: DualFranka, cfg: HandlingConfig):
        self.robot, self.cfg = robot, cfg
        m = robot.model
        self.box_sensor = [robot.sensor_slice(f'box{k}_pos') for k in range(N_BOXES)]
        self.box_qadr = [m.joint(f'box{k}').qposadr[0] for k in range(N_BOXES)]
        self.box_dadr = [m.joint(f'box{k}').dofadr[0] for k in range(N_BOXES)]
        self.start = [m.body(f'box{k}').pos.copy() for k in range(N_BOXES)]
        self.slots = [m.geom(f'slot{k}').pos.copy() for k in range(N_BOXES)]
        self.handover = np.asarray(cfg.handover, dtype=float)
        self.arm = {GIVER: ArmPhase('pick_above'), TAKER: ArmPhase('wait')}
        self.box = {GIVER: 0, TAKER: 0}  # box each arm works on
        self.delivered, self.drops = 0, 0
        self.setpoint = {arm: robot.ee(arm) for arm in (GIVER, TAKER)}
        self.events: list = []

    # Geometry -----------------------------------------------------------------
    def box_pos(self, k: int, data=None) -> np.ndarray:
        d = self.robot.data if data is None else data
        return d.sensordata[self.box_sensor[k]].copy()

    def grasp(self, arm: str, centre: np.ndarray) -> np.ndarray:
        """Fingertip point for the arm on a box centred at ``centre``: the giver holds the -y end."""
        side = -1.0 if arm == GIVER else 1.0
        return centre + np.array([0.0, side * self.cfg.grasp_offset, 0.0])

    @property
    def done(self) -> bool:
        return self.delivered >= N_BOXES

    # Phase machines -----------------------------------------------------------
    def target(self, arm: str) -> tuple[np.ndarray, float]:
        """Fingertip target and gripper command of the arm's current phase."""
        cfg, a = self.cfg, self.arm[arm]
        up = lambda h: np.array([0.0, 0.0, h])
        if a.frozen is not None:
            point = a.frozen
        elif arm == GIVER:
            box = self.box_pos(self.box[GIVER]) if self.box[GIVER] < N_BOXES else self.handover
            point = {'pick_above': self.grasp(GIVER, box) + up(cfg.approach),
                     'pick_down': self.grasp(GIVER, box),
                     'to_handover': self.grasp(GIVER, self.handover),
                     'hold': self.grasp(GIVER, self.handover),
                     'release': self.grasp(GIVER, self.handover),
                     'home': self.grasp(GIVER, self.handover) + np.array([0.0, -0.15, cfg.approach])}[a.phase]
        else:
            box = self.box_pos(min(self.box[TAKER], N_BOXES - 1))
            slot = self.slots[min(self.box[TAKER], N_BOXES - 1)].copy()
            slot[2] = TABLE_TOP + BOX_HALF[2] + cfg.place_clearance
            point = {'wait': self.grasp(TAKER, self.handover) + np.array([0.0, 0.15, cfg.approach]),
                     'take_above': self.grasp(TAKER, box) + up(cfg.approach),
                     'take_down': self.grasp(TAKER, box),
                     'carry': self.grasp(TAKER, slot) + up(cfg.approach),
                     'place': self.grasp(TAKER, slot),
                     'open': self.grasp(TAKER, slot),
                     'home': self.grasp(TAKER, self.handover) + np.array([0.0, 0.15, cfg.approach])}[a.phase]
        closed = a.phase in ('pick_close', 'lift', 'to_handover', 'hold',
                             'take_close', 'take_hold', 'carry', 'place')
        return point, GRIP_CLOSED if closed else GRIP_OPEN

    def goal(self) -> dict:
        """Setpoints for MPPI, each moved one control step towards its phase target."""
        cfg = self.cfg
        for arm in (GIVER, TAKER):
            target, grip = self.target(arm)
            step = target - self.setpoint[arm]
            reach = (cfg.v_held if grip == GRIP_CLOSED else cfg.v_free) * cfg.dt
            dist = np.linalg.norm(step)
            self.setpoint[arm] = target if dist <= reach else self.setpoint[arm] + step * (reach / dist)
        return {arm: self.setpoint[arm].copy() for arm in (GIVER, TAKER)}

    def _go(self, arm: str, phase: str, t: float, freeze: np.ndarray | None = None):
        self.arm[arm].set(phase, t, freeze)

    def update(self, t: float):
        """Advances both phase machines; returns a list of event strings."""
        cfg, robot = self.cfg, self.robot
        events = []
        events += self._check_drop(t)
        g, k = self.arm[GIVER], self.arm[TAKER]
        err = {arm: np.linalg.norm(robot.ee(arm) - self.target(arm)[0]) for arm in (GIVER, TAKER)}
        up = np.array([0.0, 0.0, cfg.lift])
        # Giver.
        if g.phase == 'pick_above' and err[GIVER] < cfg.tol_move:
            self._go(GIVER, 'pick_down', t)
        elif g.phase == 'pick_down' and err[GIVER] < cfg.tol_fine:
            self._go(GIVER, 'pick_close', t, self.target(GIVER)[0])
        elif g.phase == 'pick_close' and t - g.since >= cfg.t_close:
            self._go(GIVER, 'lift', t, g.frozen + up)
        elif g.phase == 'lift' and err[GIVER] < cfg.tol_move:
            self._go(GIVER, 'to_handover', t)
        elif g.phase == 'to_handover' and err[GIVER] < cfg.tol_move:
            self._go(GIVER, 'hold', t)
            events.append(f'box {self.box[GIVER]} at the handover')
        elif g.phase == 'hold' and k.phase == 'take_close' and t - k.since >= cfg.t_close:
            self._go(GIVER, 'release', t)
        elif g.phase == 'release' and t - g.since >= cfg.t_open:
            self._go(GIVER, 'retreat', t, robot.ee(GIVER) + np.array([0.0, -0.15, cfg.approach]))
            events.append(f'box {self.box[GIVER]} handed over')
        elif g.phase == 'retreat' and err[GIVER] < cfg.tol_move * 2:
            self.box[GIVER] += 1
            self._go(GIVER, 'pick_above' if self.box[GIVER] < N_BOXES else 'home', t)
        # Taker.
        if k.phase == 'wait' and g.phase == 'hold' and self.box[TAKER] == self.box[GIVER]:
            self._go(TAKER, 'take_above', t)
        elif k.phase == 'take_above' and err[TAKER] < cfg.tol_move:
            self._go(TAKER, 'take_down', t)
        elif k.phase == 'take_down' and err[TAKER] < cfg.tol_fine:
            self._go(TAKER, 'take_close', t, self.target(TAKER)[0])
        elif k.phase == 'take_close' and t - k.since >= cfg.t_close:
            self._go(TAKER, 'take_hold', t, k.frozen)
        elif k.phase == 'take_hold' and g.phase in ('retreat', 'pick_above', 'home'):
            self._go(TAKER, 'carry', t)
        elif k.phase == 'carry' and err[TAKER] < cfg.tol_move:
            self._go(TAKER, 'place', t)
        elif k.phase == 'place' and err[TAKER] < cfg.tol_fine:
            self._go(TAKER, 'open', t)
        elif k.phase == 'open' and t - k.since >= cfg.t_open:
            self._go(TAKER, 'retreat', t, robot.ee(TAKER) + up)
            self.delivered += 1
            events.append(f'box {self.box[TAKER]} placed ({self.delivered}/{N_BOXES})')
        elif k.phase == 'retreat' and err[TAKER] < cfg.tol_move * 2:
            self.box[TAKER] += 1
            self._go(TAKER, 'wait' if self.box[TAKER] < N_BOXES else 'home', t)
        return events

    def _check_drop(self, t: float) -> list:
        """A box on its way (picked, not yet placed) that falls below the tables goes back to its start."""
        held = {self.box[GIVER]} if self.arm[GIVER].phase in ('lift', 'to_handover', 'hold', 'release') else set()
        if self.arm[TAKER].phase in ('take_close', 'take_hold', 'carry', 'place'):
            held.add(self.box[TAKER])
        for b in held:
            if self.box_pos(b)[2] < TABLE_TOP - 0.02:
                self.reset_box(b)
                self.drops += 1
                self.box[GIVER], self.box[TAKER] = b, b
                self._go(GIVER, 'pick_above', t)
                self._go(TAKER, 'wait', t)
                return [f'box {b} dropped, back to the start table']
        return []

    def reset_box(self, k: int):
        m, d = self.robot.model, self.robot.data
        d.qpos[self.box_qadr[k]:self.box_qadr[k] + 7] = (*self.start[k], 1.0, 0.0, 0.0, 0.0)
        d.qvel[self.box_dadr[k]:self.box_dadr[k] + 6] = 0.0
        mujoco.mj_forward(m, d)

    # MPPI cost ----------------------------------------------------------------
    def arm_cost(self, sensors: np.ndarray, arm: str, goal: dict) -> np.ndarray:
        """Squared fingertip-target distance over the horizon plus a terminal term."""
        err = np.sum((sensors[:, :, self.robot.sensor_ee[arm]] - goal[arm]) ** 2, axis=2)
        return self.cfg.w_target * err.mean(axis=1) + self.cfg.w_terminal * err[:, -1]

    def placement_errors(self) -> list:
        """Horizontal distance of each box from its goal slot, m."""
        return [float(np.linalg.norm(self.box_pos(k)[:2] - self.slots[k][:2])) for k in range(N_BOXES)]
