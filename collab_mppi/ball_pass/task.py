"""Roles, MPPI costs and pass bookkeeping for the table ball pass.

Roles follow from the ball state alone (position b, velocity v). The owner is
the arm whose half of the table holds the ball; u is the ball speed towards
the other arm.

- u < -v_receive (ball coming at the owner): owner receives, other is ready.
- u > v_release (ball sent):                 owner is ready, other receives.
- otherwise (ball slow in the owner's half):  owner kicks, other receives.

kick:    aim from the ball's current position through the table centre (0, 0),
         straight across when the ball is near the centre. Fingertip behind the
         ball on that line; reward ball speed and the arm's push force along it,
         penalize sideways ball speed.
receive: fingertip on the arm's catch line, in the ball's lane; ball speed driven
         to 0 once the ball is within brake_zone of that line (braking earlier would
         pull the arm out to the midline, where its next kick is weak).
ready:   fingertip back to a waiting point above the table.

Every role pays a strong penalty for any part of the arm (fingertip and the
clearance proxies on links 3-7 and the hand) closer than zone_margin to the
middle no-go zone (the scene's middle_zone geom, visual only). A ball
that stops inside the zone, where neither arm can reach it, is a stall and goes
back to the arm on its side.

A pass counts when the ball is at least pass_margin into the other half and the
receiving arm touches it there, whether it traps the ball or plays it straight
back. A ball that falls off the table is a drop; it is placed back in front of
the arm whose end it left.
"""
from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np

from ..robot import STICK_R, DualFranka

TABLE_TOP = 0.20
TABLE_HALF_X = 0.5
BALL_R = 0.035
OUTWARD = {'left': 1.0, 'right': -1.0}  # direction from each arm towards the other
OTHER = {'left': 'right', 'right': 'left'}


@dataclass
class PassConfig:
    v_release: float = 0.6     # the kicker lets go above this speed, m/s
    v_receive: float = 0.15    # an incoming ball faster than this is received, slower is fetched, m/s
    pass_margin: float = 0.05  # a touch counts as a pass this far past the midline, m
    zone_margin: float = 0.02  # arm points stay this far short of the middle zone, m
    v_stall: float = 0.03      # a ball slower than this inside the zone is stalled, m/s
    stall_hold: float = 0.5    # time stalled before the ball is re-placed, s
    aim_radius: float = 0.05   # closer to the centre than this, kick straight across, m
    catch_x: float = 0.30      # distance of each catch line from the midline, m
    brake_zone: float = 0.10   # braking counts when the ball is this close to the catch line, m
    kick_x: float = 0.25       # where a dropped ball is placed, distance from the midline, m
    ready_height: float = 0.15  # waiting point above the table, m
    contact_height: float = BALL_R  # stick tip height above the table when pushing (ball centre), m
    w_approach: float = 1.0
    w_terminal: float = 5.0
    w_speed: float = 1.0       # reward per m/s of ball speed along the aim
    w_push: float = 0.02       # reward per N of push force along the aim
    w_lane: float = 0.5        # penalty on squared sideways ball speed
    w_brake: float = 1.0
    w_off_table: float = 100.0
    w_zone: float = 1e3        # per metre of intrusion, per arm point and step


def ball_roles(b: np.ndarray, v: np.ndarray, cfg: PassConfig) -> dict:
    owner = 'left' if b[0] < 0 else 'right'
    u = OUTWARD[owner] * v[0]
    if u < -cfg.v_receive:
        return {owner: 'receive', OTHER[owner]: 'ready'}
    if u > cfg.v_release:
        return {owner: 'ready', OTHER[owner]: 'receive'}
    return {owner: 'kick', OTHER[owner]: 'receive'}


class PassTask:
    def __init__(self, robot: DualFranka, cfg: PassConfig):
        self.robot, self.cfg = robot, cfg
        self.ball_pos = robot.sensor_slice('ball_pos')
        self.ball_vel = robot.sensor_slice('ball_vel')
        # Contact sensors report the force on the arm; the push on the ball is its negative.
        self.push = {arm: robot.sensor_slice(f'{arm}_push') for arm in ('left', 'right')}
        self.contact_z = TABLE_TOP + cfg.contact_height
        self.zone_half_x = float(robot.model.geom('middle_zone').size[0])

    def ball(self, data=None):
        d = self.robot.data if data is None else data
        return d.sensordata[self.ball_pos].copy(), d.sensordata[self.ball_vel].copy()

    def goal(self) -> dict:
        """Roles and kick aim from the current ball state, fixed over the MPPI horizon."""
        b, v = self.ball()
        roles = ball_roles(b, v, self.cfg)
        owner = 'left' if b[0] < 0 else 'right'
        dist = np.linalg.norm(b[:2])
        aim = -b[:2] / dist if dist > self.cfg.aim_radius else np.array([OUTWARD[owner], 0.0])
        return {'roles': roles, 'aim': aim}

    def role_point(self, arm: str, goal: dict, b: np.ndarray) -> np.ndarray:
        """Fingertip goal for the arm's role given ball position(s) b (..., 3)."""
        cfg, out, role = self.cfg, OUTWARD[arm], goal['roles'][arm]
        p = np.empty(np.broadcast_shapes(b.shape, (3,)))
        if role == 'kick':
            p[..., :2] = b[..., :2] - (BALL_R + STICK_R + 0.005) * goal['aim']
            p[..., 2] = self.contact_z
        elif role == 'receive':
            p[..., 0] = -out * cfg.catch_x
            p[..., 1] = b[..., 1]
            p[..., 2] = self.contact_z
        else:
            p[..., 0] = -out * cfg.catch_x
            p[..., 1] = 0.0
            p[..., 2] = TABLE_TOP + cfg.ready_height
        return p

    def arm_cost(self, sensors: np.ndarray, arm: str, goal: dict) -> np.ndarray:
        cfg, role = self.cfg, goal['roles'][arm]
        ee = sensors[:, :, self.robot.sensor_ee[arm]]
        b, v = sensors[:, :, self.ball_pos], sensors[:, :, self.ball_vel]
        err = np.sum((ee - self.role_point(arm, goal, b)) ** 2, axis=2)
        cost = cfg.w_approach * err.mean(axis=1) + cfg.w_terminal * err[:, -1]
        if role == 'kick':
            aim = goal['aim']
            side = np.array([-aim[1], aim[0]])
            push = -sensors[:, :, self.push[arm]][..., :2] @ aim
            cost -= cfg.w_speed * np.mean(v[..., :2] @ aim, axis=1)
            cost -= cfg.w_push * np.mean(np.maximum(push, 0.0), axis=1)
            cost += cfg.w_lane * np.mean((v[..., :2] @ side) ** 2, axis=1)
        elif role == 'receive':
            near = -OUTWARD[arm] * b[..., 0] > cfg.catch_x - cfg.brake_zone
            cost += cfg.w_brake * np.mean(near * np.sum(v ** 2, axis=2), axis=1)
        off = np.maximum(np.abs(b[..., 0]) - TABLE_HALF_X, 0.0)
        cost += cfg.w_off_table * np.sum(off ** 2, axis=1)
        points_x = sensors[:, :, self.robot.sensor_arm[arm]][..., 0::3]
        intrusion = np.maximum(OUTWARD[arm] * points_x + self.zone_half_x + cfg.zone_margin, 0.0)
        return cost + cfg.w_zone * np.sum(intrusion, axis=(1, 2))

    def touching(self, arm: str, data=None) -> bool:
        d = self.robot.data if data is None else data
        return bool(np.any(d.sensordata[self.push[arm]] != 0.0))


class PassTracker:
    """Counts passes (ball touched by the receiver in its half), drops and stalls, and re-places lost balls."""

    def __init__(self, robot: DualFranka, task: PassTask, passes: int):
        if passes < 0:
            raise ValueError('passes must be nonnegative (0 repeats until --steps)')
        self.robot, self.task, self.target = robot, task, passes
        self.ball_qadr = robot.model.joint('ball').qposadr[0]
        self.ball_dadr = robot.model.joint('ball').dofadr[0]
        self.passes, self.drops, self.stalls, self.stall_since = 0, 0, 0, None
        self.holder = self.half(task.ball()[0])
        self.pass_times: list = []

    @staticmethod
    def half(b) -> str:
        return 'left' if b[0] < 0 else 'right'

    @property
    def done(self) -> bool:
        return self.target > 0 and self.passes >= self.target

    def place_ball(self, arm: str):
        m, d = self.robot.model, self.robot.data
        d.qpos[self.ball_qadr:self.ball_qadr + 7] = (-OUTWARD[arm] * self.task.cfg.kick_x, 0.0,
                                                     TABLE_TOP + BALL_R, 1.0, 0.0, 0.0, 0.0)
        d.qvel[self.ball_dadr:self.ball_dadr + 6] = 0.0
        mujoco.mj_forward(m, d)

    def update(self, t: float) -> str | None:
        """Returns 'pass', 'drop', 'stall' or None."""
        b, v = self.task.ball()
        if b[2] < TABLE_TOP - 0.05:
            arm = self.half(b)
            self.place_ball(arm)
            self.drops += 1
            self.holder, self.stall_since = arm, None
            return 'drop'
        if abs(b[0]) < self.task.zone_half_x and np.linalg.norm(v[:2]) < self.task.cfg.v_stall:
            if self.stall_since is None:
                self.stall_since = t
            if t - self.stall_since >= self.task.cfg.stall_hold:
                arm = self.half(b)
                self.place_ball(arm)
                self.stalls += 1
                self.holder, self.stall_since = arm, None
                return 'stall'
        else:
            self.stall_since = None
        receiver = self.half(b)
        if (receiver == self.holder or abs(b[0]) < self.task.cfg.pass_margin
                or not self.task.touching(receiver)):
            return None
        self.holder = receiver
        self.passes += 1
        self.pass_times.append(t)
        return 'pass'

