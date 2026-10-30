"""Roles, MPPI costs and pass bookkeeping for two-arm air hockey.

The arms pass a frictionless puck, struck with the disc at the stick tip.

Roles follow from the puck state (position b, velocity v), player style: catch
the puck deep in the own half and bring it back to rest on the set point at the
centre of that half, in one phase, then strike. The owner is the arm whose half
of the table holds the puck; u is the puck speed towards the other arm.

- u > v_release (puck sent): owner is ready, other catches.
- otherwise the owner catches, and kicks once the puck rests within set_radius of
  its set point along x (slower than v_settle; a puck held against a rail cannot
  be pushed off it). A puck that is not coming in but is too close to the no-go
  zone for the tip to get between them is kicked straight away. The kick holds
  until the puck is sent, comes back in, or leaves the half. While the owner
  catches the other arm is ready; while it kicks the other arm catches.

catch:   while the puck comes in (faster than v_receive towards the arm's end) or
         is in the other half, fingertip waits deep on the catch line catch_x, in
         the puck's lane (or just behind a puck that got past it). Otherwise
         fingertip behind the puck on the line to the set point. Once the puck
         reaches the waiting tip, or is no longer coming in, puck position is
         driven to the set point and puck velocity to a target field
         k_set * (set point - puck), capped at v_set, which fades to rest on the
         set point (AiRLIHockey prepares the puck with a target puck state after
         contact rather than a target position). Nothing rewards stopping the
         puck earlier, so it runs deep and the recentering has room.
Kick and catch push along x instead when the point behind the puck lies past a
rail, out of the tip's reach.
kick:    aim from the puck's current position through the table centre (0, 0),
         straight across when the puck is near the centre. Fingertip behind the
         puck on that line; reward puck speed (capped at v_cap) and the arm's push
         force along it, penalize sideways puck speed. Two terms adapted from the
         air hockey challenge give every rollout a signal before contact: an
         approach-from-behind reward cos^2(tip->puck, aim) * exp(-k * gap)
         (MushroomRL AirHockeyHit), and a shot reward from AiRLIHockey's mallet-level
         MPC, which tracks a tip velocity v_shot * aim at contact: 1 - |tip velocity -
         v_shot * aim|^2 / v_shot^2 (floored at 0) while the tip is close behind the
         puck. It is a bounded reward rather than the paper's error cost: as a cost,
         standing slow near the puck is penalized and MPPI, which has no hard contact
         constraint, keeps the tip away from it. With cone > 0 the puck speed only counts
         while the puck heads within that angle of the aim, and is penalized
         otherwise (Polimi's triangle reward).
ready:   fingertip back to a waiting point on the catch line. The tip stays at
         table height in every role (run.py locks it there in the MPPI).

Every role pays a strong penalty for any part of the arm (fingertip and the
clearance proxies on links 3-7 and the hand) closer than zone_margin to the
middle no-go zone (the scene's middle_zone geom, visual only), and for the
stick tip pressing against a side rail (role points stay inside them). A puck
that stops inside the zone, where neither arm can reach it, is a stall and goes
back to the arm on its side.

A pass counts when the puck is at least pass_margin into the other half and the
receiving arm touches it there, whether it traps the puck or plays it straight
back. A puck that falls off the table is a drop; it is placed back on the set
point of the arm whose end it left.
"""
from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np

from ..robot import TIP_R, DualFranka

TABLE_TOP = 0.20
TABLE_HALF_X = 0.5
PUCK_R, PUCK_HALF_H = 0.035, 0.012  # puck radius and half-height
OUTWARD = {'left': 1.0, 'right': -1.0}  # direction from each arm towards the other
OTHER = {'left': 'right', 'right': 'left'}


@dataclass
class PassConfig:
    v_release: float = 0.6     # the kicker lets go above this speed, m/s
    v_receive: float = 0.15    # a puck faster than this towards an arm's end is coming in, m/s
    pass_margin: float = 0.05  # a touch counts as a pass this far past the midline, m
    zone_margin: float = 0.02  # arm points stay this far short of the middle zone, m
    v_stall: float = 0.03      # a puck slower than this inside the zone is stalled, m/s
    stall_hold: float = 0.5    # time stalled before the puck is re-placed, s
    aim_radius: float = 0.05   # closer to the centre than this, kick straight across, m
    catch_x: float = 0.35      # distance of each catch line from the midline, m
    set_x: float = 0.25        # set point (and where a dropped puck is placed), distance from the midline, m
    set_radius: float = 0.05   # the puck is set when this close to the set point along x, m
    v_settle: float = 0.15     # and slower than this, m/s
    w_set: float = 100.0       # puck distance to the set point, squared
    w_field: float = 5.0       # squared error to the target puck velocity field
    k_set: float = 3.0         # its gain towards the set point, 1/s
    v_set: float = 0.5         # and its speed cap, m/s
    contact_height: float = PUCK_HALF_H + 0.003  # stick tip disc centre above the table when pushing, m
    w_approach: float = 1.0
    w_terminal: float = 5.0
    w_speed: float = 1.0       # reward per m/s of puck speed along the aim
    w_push: float = 0.02       # reward per N of push force along the aim
    w_lane: float = 0.5        # penalty on squared sideways puck speed
    v_cap: float = np.inf      # puck speed reward saturates here, m/s
    cone: float = 0.0          # half-angle of the accepted puck heading around the aim, rad (0: off)
    w_align: float = 0.0       # approach-from-behind reward
    align_k: float = 8.0       # its decay per metre of gap between stick tip and puck surface, 1/m
    w_shot: float = 0.0        # shot reward: tip velocity close to v_shot * aim near the puck
    v_shot: float = 1.5        # tip velocity at contact, m/s
    shot_radius: float = 0.12  # tip-puck centre distance within which the tip velocity counts, m
    w_off_table: float = 100.0
    w_zone: float = 1e3        # per metre of intrusion, per arm point and step
    rail_margin: float = 0.01  # stick tip disc stays this far inside the side rails, m
    w_rail: float = 1e3        # per metre past that limit, per step


class PassTask:
    def __init__(self, robot: DualFranka, cfg: PassConfig):
        self.robot, self.cfg = robot, cfg
        self.puck_pos = robot.sensor_slice('puck_pos')
        self.puck_vel = robot.sensor_slice('puck_vel')
        # Contact sensors report the force on the arm; the push on the puck is its negative.
        self.push = {arm: robot.sensor_slice(f'{arm}_push') for arm in ('left', 'right')}
        self.contact_z = TABLE_TOP + cfg.contact_height
        self.zone_half_x = float(robot.model.geom('middle_zone').size[0])
        rail = robot.model.geom('rail_left')
        self.tip_y_max = float(rail.pos[1] - rail.size[1]) - TIP_R - cfg.rail_margin
        self.kicker = None  # arm whose kick is under way
        # Closer to the midline than this, the tip cannot get behind the puck to pull it back.
        self.reach_depth = self.zone_half_x + cfg.zone_margin + PUCK_R + TIP_R + 0.025

    def set_point(self, arm: str) -> np.ndarray:
        return np.array([-OUTWARD[arm] * self.cfg.set_x, 0.0])

    def roles(self, b: np.ndarray, v: np.ndarray) -> dict:
        cfg = self.cfg
        owner = 'left' if b[0] < 0 else 'right'
        u = OUTWARD[owner] * v[0]
        incoming = u < -cfg.v_receive
        if self.kicker != owner or u > cfg.v_release or incoming:
            self.kicker = None
        if u > cfg.v_release:
            return {owner: 'ready', OTHER[owner]: 'catch'}
        if self.kicker is None and not incoming and (-OUTWARD[owner] * b[0] < self.reach_depth or (
                abs(b[0] - self.set_point(owner)[0]) < cfg.set_radius and np.linalg.norm(v[:2]) < cfg.v_settle)):
            self.kicker = owner
        if self.kicker == owner:
            return {owner: 'kick', OTHER[owner]: 'catch'}
        return {owner: 'catch', OTHER[owner]: 'ready'}

    def coming_in(self, arm: str, b: np.ndarray, v: np.ndarray) -> np.ndarray:
        """Puck (..., 3) in the other half, or moving towards the arm's end faster than v_receive."""
        return (OUTWARD[arm] * b[..., 0] > 0) | (-OUTWARD[arm] * v[..., 0] > self.cfg.v_receive)

    def puck(self, data=None):
        d = self.robot.data if data is None else data
        return d.sensordata[self.puck_pos].copy(), d.sensordata[self.puck_vel].copy()

    def goal(self) -> dict:
        """Roles and kick aim from the current puck state, fixed over the MPPI horizon.
        Call once per control step: it advances the kick state."""
        b, v = self.puck()
        roles = self.roles(b, v)
        owner = 'left' if b[0] < 0 else 'right'
        dist = np.linalg.norm(b[:2])
        aim = -b[:2] / dist if dist > self.cfg.aim_radius else np.array([OUTWARD[owner], 0.0])
        return {'roles': roles, 'aim': self.reachable(b[:2], aim, np.array([OUTWARD[owner], 0.0]))}

    def reachable(self, b, push, fallback):
        """push where the tip can stand behind the puck b (..., 2) on that line, else fallback."""
        behind_y = b[..., 1] - (PUCK_R + TIP_R + 0.005) * push[..., 1]
        return np.where((np.abs(behind_y) <= self.tip_y_max)[..., None], push, fallback)

    def role_point(self, arm: str, goal: dict, b: np.ndarray, v: np.ndarray) -> np.ndarray:
        """Fingertip goal for the arm's role given puck position(s) b and velocities v (..., 3)."""
        cfg, out, role = self.cfg, OUTWARD[arm], goal['roles'][arm]
        p = np.empty(np.broadcast_shapes(b.shape, (3,)))
        if role == 'kick':
            p[..., :2] = b[..., :2] - (PUCK_R + TIP_R + 0.005) * goal['aim']
            p[..., 2] = self.contact_z
        elif role == 'catch':
            # Behind the puck on the line to the set point ...
            to_set = self.set_point(arm) - b[..., :2]
            dist = np.linalg.norm(to_set, axis=-1, keepdims=True)
            push = np.where(dist > 1e-3, to_set / np.maximum(dist, 1e-3), [out, 0.0])
            along_x = np.zeros_like(push)
            along_x[..., 0] = np.where(to_set[..., 0] >= 0, 1.0, -1.0)
            push = self.reachable(b[..., :2], push, along_x)
            p[..., :2] = b[..., :2] - (PUCK_R + TIP_R + 0.005) * push
            # ... or, while it comes in, deep on the catch line in its lane (behind it if it got past).
            depth = np.minimum(np.maximum(cfg.catch_x, -out * b[..., 0] + PUCK_R + TIP_R + 0.005), TABLE_HALF_X)
            wait = self.coming_in(arm, b, v)
            p[..., 0] = np.where(wait, -out * depth, p[..., 0])
            p[..., 1] = np.where(wait, b[..., 1], p[..., 1])
            p[..., 2] = self.contact_z
        else:
            p[..., 0] = -out * cfg.catch_x
            p[..., 1] = 0.0
            p[..., 2] = self.contact_z
        p[..., 1] = np.clip(p[..., 1], -self.tip_y_max, self.tip_y_max)
        return p

    def arm_cost(self, sensors: np.ndarray, arm: str, goal: dict) -> np.ndarray:
        cfg, role = self.cfg, goal['roles'][arm]
        ee = sensors[:, :, self.robot.sensor_ee[arm]]
        b, v = sensors[:, :, self.puck_pos], sensors[:, :, self.puck_vel]
        err = np.sum((ee - self.role_point(arm, goal, b, v)) ** 2, axis=2)
        cost = cfg.w_approach * err.mean(axis=1) + cfg.w_terminal * err[:, -1]
        if role == 'kick':
            aim = goal['aim']
            side = np.array([-aim[1], aim[0]])
            push = -sensors[:, :, self.push[arm]][..., :2] @ aim
            along, across = v[..., :2] @ aim, v[..., :2] @ side
            speed = np.minimum(along, cfg.v_cap)
            if cfg.cone > 0:
                heading_ok = np.abs(across) <= np.tan(cfg.cone) * along
                speed = np.where(heading_ok, speed, -np.hypot(along, across))
            cost -= cfg.w_speed * np.mean(speed, axis=1)
            cost -= cfg.w_push * np.mean(np.maximum(push, 0.0), axis=1)
            cost += cfg.w_lane * np.mean(across ** 2, axis=1)
            if cfg.w_align or cfg.w_shot:
                to_puck = b[..., :2] - ee[..., :2]
                dist = np.linalg.norm(to_puck, axis=2)
                behind = (to_puck @ aim) / np.maximum(dist, 1e-6)
                if cfg.w_align:
                    gap = np.maximum(dist - PUCK_R - TIP_R, 0.0)
                    align = np.maximum(behind, 0.0) ** 2 * np.exp(-cfg.align_k * gap)
                    cost -= cfg.w_align * np.mean(align, axis=1)
                if cfg.w_shot:
                    tip_v = sensors[:, :, self.robot.sensor_ee_vel[arm]][..., :2]
                    match = 1.0 - np.sum((tip_v - cfg.v_shot * aim) ** 2, axis=2) / cfg.v_shot ** 2
                    near = (dist < cfg.shot_radius) & (behind > 0)
                    cost -= cfg.w_shot * np.mean(near * np.maximum(match, 0.0), axis=1)
        elif role == 'catch':
            # Set the puck once it reaches the waiting tip or stops coming in, not earlier.
            at_tip = -OUTWARD[arm] * b[..., 0] > cfg.catch_x - PUCK_R - TIP_R - 0.03
            active = (OUTWARD[arm] * b[..., 0] < 0) & (at_tip | ~self.coming_in(arm, b, v))
            miss = active * np.sum((b[..., :2] - self.set_point(arm)) ** 2, axis=2)
            cost += cfg.w_set * (miss.mean(axis=1) + miss[:, -1])
            field = cfg.k_set * (self.set_point(arm) - b[..., :2])
            speed = np.linalg.norm(field, axis=2, keepdims=True)
            field *= np.minimum(1.0, cfg.v_set / np.maximum(speed, 1e-9))
            cost += cfg.w_field * np.mean(active * np.sum((v[..., :2] - field) ** 2, axis=2), axis=1)
        off = np.maximum(np.abs(b[..., 0]) - TABLE_HALF_X, 0.0)
        cost += cfg.w_off_table * np.sum(off ** 2, axis=1)
        points_x = sensors[:, :, self.robot.sensor_arm[arm]][..., 0::3]
        intrusion = np.maximum(OUTWARD[arm] * points_x + self.zone_half_x + cfg.zone_margin, 0.0)
        cost += cfg.w_zone * np.sum(intrusion, axis=(1, 2))
        # A tip pressed against a rail wedges the arm away from its joint targets.
        return cost + cfg.w_rail * np.sum(np.maximum(np.abs(ee[..., 1]) - self.tip_y_max, 0.0), axis=1)

    def touching(self, arm: str, data=None) -> bool:
        d = self.robot.data if data is None else data
        return bool(np.any(d.sensordata[self.push[arm]] != 0.0))


class PassTracker:
    """Counts passes (puck touched by the receiver in its half), drops and stalls, and re-places lost pucks."""

    def __init__(self, robot: DualFranka, task: PassTask, passes: int):
        if passes < 0:
            raise ValueError('passes must be nonnegative (0 repeats until --steps)')
        self.robot, self.task, self.target = robot, task, passes
        self.puck_qadr = robot.model.joint('puck').qposadr[0]
        self.puck_dadr = robot.model.joint('puck').dofadr[0]
        self.passes, self.drops, self.stalls, self.stall_since = 0, 0, 0, None
        self.holder = self.half(task.puck()[0])
        self.pass_times: list = []

    @staticmethod
    def half(b) -> str:
        return 'left' if b[0] < 0 else 'right'

    @property
    def done(self) -> bool:
        return self.target > 0 and self.passes >= self.target

    def place_puck(self, arm: str):
        m, d = self.robot.model, self.robot.data
        d.qpos[self.puck_qadr:self.puck_qadr + 7] = (-OUTWARD[arm] * self.task.cfg.set_x, 0.0,
                                                     TABLE_TOP + PUCK_HALF_H, 1.0, 0.0, 0.0, 0.0)
        d.qvel[self.puck_dadr:self.puck_dadr + 6] = 0.0
        mujoco.mj_forward(m, d)

    def update(self, t: float) -> str | None:
        """Returns 'pass', 'drop', 'stall' or None."""
        b, v = self.task.puck()
        if b[2] < TABLE_TOP - 0.05:
            arm = self.half(b)
            self.place_puck(arm)
            self.drops += 1
            self.holder, self.stall_since = arm, None
            return 'drop'
        if abs(b[0]) < self.task.zone_half_x and np.linalg.norm(v[:2]) < self.task.cfg.v_stall:
            if self.stall_since is None:
                self.stall_since = t
            if t - self.stall_since >= self.task.cfg.stall_hold:
                arm = self.half(b)
                self.place_puck(arm)
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

