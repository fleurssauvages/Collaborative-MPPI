"""Phases, MPPI cost and bookkeeping for the object handling task.

Robot 1 (the right arm, the giver) picks each box from the start table on its
right and holds it up in the air between the robots; robot 2 (the left arm, the
taker) grasps the other end of the box there, robot 1 lets go, and robot 2 puts
the box on its slot on the goal table on its left. Boxes go in order 0, 1, 2;
robot 1 fetches the next box while robot 2 places the previous one.

Each arm has two phases, one per gripper state: reach (open) and carry (closed).
A phase gives the fingertip target and the gripper command; MPPI tracks a
setpoint that follows it (the gripper command is held over the rollouts).

reach box k: the target is the arm's grasp point on box k, approached from above
  (it rises by `approach` once the setpoint is more than descend[1] from it
  horizontally). Robot 2 waits at a rest point until robot 1 holds the box at the
  handover; either arm rests once all boxes are done. Until the fingers are fully
  open (after a release) the setpoint stays put. Ends when the fingertip is at the
  grasp point.
carry box k: the setpoint walks at v_held along a path, robot 1's up from the grasp
  point and to the handover, robot 2's to above the slot and down onto it. It only
  advances once the fingers have closed on the box, and for robot 2 once robot 1's
  fingers are open again. Robot 1's carry ends when robot 2 holds the box, robot
  2's when the box is on its slot; fingers that close on nothing go back to reach.

Both grippers point straight down with the fingers closing along x; run.py
locks that orientation in the MPPI. A box that falls (below the table tops while
held) is a drop: it goes back to its start position and both arms restart it.
"""
from __future__ import annotations

import copy
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


def plan_model(model: mujoco.MjModel) -> mujoco.MjModel:
    """Copy of the model for the MPPI rollouts, with the boxes floating and colliding
    with nothing: the cost only reads the arms, the boxes are light, and the 18
    contacts per finger on a held box made every rollout about 3x slower."""
    plan = copy.deepcopy(model)
    for k in range(N_BOXES):
        plan.body_gravcomp[plan.body(f'box{k}').id] = 1.0
        plan.geom_contype[plan.geom(f'box{k}').id] = 0
        plan.geom_conaffinity[plan.geom(f'box{k}').id] = 0
    return plan


@dataclass
class HandlingConfig:
    handover: tuple = (0.52, 0.0, 0.40)  # box centre at the handover, m
    grasp_offset: float = 0.07  # each gripper holds the box this far from its centre along y, m
    approach: float = 0.10     # approach height above a grasp point, m
    descend: tuple = (0.015, 0.05)  # horizontal distances to the grasp point: fully down, fully up, m
    lift: float = 0.12         # robot 1 lifts a picked box this much before heading to the handover, m
    place_clearance: float = 0.005  # box lowered to this height above the goal table, m
    tol_fine: float = 0.008    # fingertip error at a grasp point that closes the gripper, m
    holds_finger: float = 0.024  # finger opening below which closed fingers hold the box, m
    empty_finger: float = 0.012  # finger opening below which closed fingers missed the box, m
    open_finger: float = 0.038   # finger opening above which the fingers are open, m
    v_free: float = 0.8        # setpoint speed with the gripper open, m/s
    v_held: float = 0.35       # setpoint speed along a carry path, m/s
    dt: float = 0.01           # control step, s
    w_target: float = 1.0
    w_terminal: float = 10.0


class ArmPhase:
    """Phase of one arm ('reach' or 'carry'), its box, and the carry path and progress."""

    def __init__(self, phase: str, box: int = 0):
        self.set(phase, box)

    def set(self, phase: str, box: int, path: np.ndarray | None = None):
        self.phase, self.box, self.path, self.s = phase, box, path, 0.0

    def path_point(self) -> np.ndarray:
        """Point at arc length s along the path."""
        seg = np.diff(self.path, axis=0)
        lengths = np.linalg.norm(seg, axis=1)
        s = self.s
        for start, d, length in zip(self.path[:-1], seg, lengths):
            if s <= length:
                return start + d * (s / max(length, 1e-9))
            s -= length
        return self.path[-1].copy()

    @property
    def path_done(self) -> bool:
        return self.s >= np.linalg.norm(np.diff(self.path, axis=0), axis=1).sum()


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
        self.rest = {GIVER: self.grasp(GIVER, self.handover) + np.array([0.0, -0.15, cfg.approach]),
                     TAKER: self.grasp(TAKER, self.handover) + np.array([0.0, 0.15, cfg.approach])}
        self.arm = {GIVER: ArmPhase('reach'), TAKER: ArmPhase('reach')}
        self.delivered, self.drops = 0, 0
        self.setpoint = {arm: robot.ee(arm) for arm in (GIVER, TAKER)}

    # Geometry -----------------------------------------------------------------
    def box_pos(self, k: int, data=None) -> np.ndarray:
        d = self.robot.data if data is None else data
        return d.sensordata[self.box_sensor[k]].copy()

    def grasp(self, arm: str, centre: np.ndarray) -> np.ndarray:
        """Fingertip point for the arm on a box centred at ``centre``: the giver holds the -y end."""
        side = -1.0 if arm == GIVER else 1.0
        return centre + np.array([0.0, side * self.cfg.grasp_offset, 0.0])

    def release_point(self, k: int) -> np.ndarray:
        slot = self.slots[k].copy()
        slot[2] = TABLE_TOP + BOX_HALF[2] + self.cfg.place_clearance
        return self.grasp(TAKER, slot)

    def fingers(self, arm: str) -> float:
        return float(self.robot.data.qpos[self.robot.finger_qadr[arm]].mean())

    @property
    def done(self) -> bool:
        return self.delivered >= N_BOXES

    def giver_at_handover(self, k: int) -> bool:
        g = self.arm[GIVER]
        return g.phase == 'carry' and g.box == k and g.path_done

    def taker_holds(self, k: int) -> bool:
        a = self.arm[TAKER]
        return a.phase == 'carry' and a.box == k and self.fingers(TAKER) < self.cfg.holds_finger

    # Phases -------------------------------------------------------------------
    def reach_point(self, arm: str) -> np.ndarray | None:
        """Exact grasp point the arm reaches for, or None while it rests."""
        k = self.arm[arm].box
        if k >= N_BOXES or (arm == TAKER and not self.giver_at_handover(k)):
            return None
        return self.grasp(arm, self.box_pos(k))

    def target(self, arm: str) -> tuple[np.ndarray, float]:
        """Fingertip target and gripper command of the arm's current phase."""
        cfg, a = self.cfg, self.arm[arm]
        if a.phase == 'carry':
            return a.path_point(), GRIP_CLOSED
        if self.fingers(arm) < cfg.open_finger:  # still letting go: stay
            return self.setpoint[arm].copy(), GRIP_OPEN
        point = self.reach_point(arm)
        if point is None:
            return self.rest[arm].copy(), GRIP_OPEN
        far = np.linalg.norm(self.setpoint[arm][:2] - point[:2])
        rise = cfg.approach * np.clip((far - cfg.descend[0]) / (cfg.descend[1] - cfg.descend[0]), 0.0, 1.0)
        return point + np.array([0.0, 0.0, rise]), GRIP_OPEN

    def goal(self) -> dict:
        """Setpoints for MPPI: carry paths as they progress, reach targets approached at v_free."""
        cfg = self.cfg
        for arm in (GIVER, TAKER):
            target, _ = self.target(arm)
            if self.arm[arm].phase == 'carry':
                self.setpoint[arm] = target
                continue
            step = target - self.setpoint[arm]
            reach = cfg.v_free * cfg.dt
            dist = np.linalg.norm(step)
            self.setpoint[arm] = target if dist <= reach else self.setpoint[arm] + step * (reach / dist)
        return {arm: self.setpoint[arm].copy() for arm in (GIVER, TAKER)}

    def update(self, t: float):
        """Advances both arms; returns a list of event strings."""
        cfg, robot = self.cfg, self.robot
        events = self._check_drop(t)
        up = np.array([0.0, 0.0, cfg.lift])
        for arm in (GIVER, TAKER):
            a = self.arm[arm]
            if a.phase == 'reach':
                point = self.reach_point(arm)
                if point is not None and self.fingers(arm) >= cfg.open_finger \
                        and np.linalg.norm(robot.ee(arm) - point) < cfg.tol_fine:
                    here = self.setpoint[arm].copy()
                    if arm == GIVER:
                        path = [here, here + up, self.grasp(GIVER, self.handover)]
                    else:
                        drop = self.release_point(a.box)
                        path = [here, drop + np.array([0.0, 0.0, cfg.approach]), drop]
                    a.set('carry', a.box, np.array(path))
                continue
            # Carry: advance along the path once the box is held (robot 2: and robot 1 has let go).
            if self.fingers(arm) < cfg.empty_finger:  # closed on nothing: reach for the box again
                events.append(f'robot {1 if arm == GIVER else 2} missed box {a.box}')
                a.set('reach', a.box)
                continue
            held = self.fingers(arm) < cfg.holds_finger
            if arm == TAKER:
                g = self.arm[GIVER]  # robot 1 has let go of this box: moved on, with its fingers opened since
                held = held and g.box > a.box and (g.phase == 'carry' or self.fingers(GIVER) >= cfg.open_finger)
            if held and not a.path_done:
                a.s += cfg.v_held * cfg.dt
                if arm == GIVER and a.path_done:
                    events.append(f'box {a.box} at the handover')
            if arm == GIVER and self.taker_holds(a.box):
                events.append(f'box {a.box} handed over')
                a.set('reach', a.box + 1)
            elif arm == TAKER and a.path_done and np.linalg.norm(robot.ee(arm) - a.path[-1]) < cfg.tol_fine:
                self.delivered += 1
                events.append(f'box {a.box} placed ({self.delivered}/{N_BOXES})')
                a.set('reach', a.box + 1)
        return events

    def _check_drop(self, t: float) -> list:
        """A held box that falls below the tables goes back to its start; both arms restart it."""
        held = {a.box for a in self.arm.values() if a.phase == 'carry'}
        for b in held:
            if self.box_pos(b)[2] < TABLE_TOP - 0.02:
                self.reset_box(b)
                self.drops += 1
                self.arm[GIVER].set('reach', b)
                self.arm[TAKER].set('reach', b)
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
