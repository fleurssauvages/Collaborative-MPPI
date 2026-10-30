"""Two Pandas pass a puck to each other across a table, with centralized or decentralized MPPI.

    python -m collab_mppi.air_hockey.run --sampler spike --layout centralized --viewer --realtime
    python -m collab_mppi.air_hockey.run --sampler pink --layout decentralized --passes 6
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import mujoco
import numpy as np

from ..controllers import CollaborativeMPPI
from ..robot import ARMS, ASSETS, PROXY_BODIES, TIP_HALF_H, DualFranka
from ..sim import Display, add_mppi_args, add_output_args, mppi_config
from .task import TABLE_TOP, PassConfig, PassTask, PassTracker

SCENE = ASSETS / 'air_hockey_scene.xml'


def make_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--passes', type=int, default=20, help='passes to complete (0 repeats until --steps)')
    p.add_argument('--steps', type=int, default=6000, help='control steps')
    p.add_argument('--control-dt', type=float, default=0.01)
    p.add_argument('--v-release', type=float, default=PassConfig.v_release,
                   help='puck speed at which the kicker lets go, m/s')
    p.add_argument('--w-speed', type=float, default=PassConfig.w_speed, help='kick reward per m/s of puck speed')
    p.add_argument('--w-push', type=float, default=PassConfig.w_push, help='kick reward per N of push force')
    p.add_argument('--v-cap', type=float, default=PassConfig.v_cap, help='puck speed reward cap, m/s')
    p.add_argument('--cone', type=float, default=PassConfig.cone,
                   help='accepted puck heading half-angle around the aim, rad (0: off)')
    p.add_argument('--w-align', type=float, default=PassConfig.w_align, help='approach-from-behind reward')
    p.add_argument('--w-shot', type=float, default=PassConfig.w_shot,
                   help='shot reward: tip velocity close to the shot velocity near the puck')
    p.add_argument('--v-shot', type=float, default=PassConfig.v_shot, help='tip velocity at contact, m/s')
    p.add_argument('--upright-gain', type=float, default=0.5,
                   help='fraction of stick tilt removed per MPPI update (0 leaves orientation free)')
    p.add_argument('--posture-gain', type=float, default=0.1,
                   help='fraction of the null-space distance to the home posture removed per MPPI update')
    add_mppi_args(p, horizon=20, noise_std=0.02)
    add_output_args(p, 'air_hockey')
    return p


def run(args) -> dict:
    robot = DualFranka(1.5, SCENE)
    m, d = robot.model, robot.data
    substeps = max(1, int(round(args.control_dt / m.opt.timestep)))
    task = PassTask(robot, PassConfig(v_release=args.v_release, w_speed=args.w_speed, w_push=args.w_push,
                                       v_cap=args.v_cap, cone=args.cone, w_align=args.w_align,
                                       w_shot=args.w_shot, v_shot=args.v_shot))
    tracker = PassTracker(robot, task, args.passes)
    mppi = CollaborativeMPPI(robot, ARMS, mppi_config(args, floor_z=TABLE_TOP + TIP_HALF_H,
                                                      upright_gain=args.upright_gain,
                                                      posture_gain=args.posture_gain, ik_damping=0.05,
                                                      tip_z=task.contact_z),
                             task.arm_cost,
                             layout=args.layout)
    display = Display(m, d, args)

    log = {k: [] for k in ('time', 'ctrl', 'qpos', 'qvel', 'puck_pos', 'puck_vel', 'left_ee', 'right_ee',
                           'left_role', 'right_role', 'passes', 'drops', 'stalls', 'clearance', 'compute_ms')}
    proxy = {arm: [m.body(f'{arm}_{b}').id for b in PROXY_BODIES]
             for arm in ARMS}
    print(f'air hockey layout={args.layout} sampler={args.sampler} passes={args.passes or "inf"}', flush=True)
    try:
        for step in range(args.steps):
            display.start_step()
            t = step * args.control_dt
            event = tracker.update(t)
            if event == 'pass':
                print(f'  t={t:6.2f}s pass {tracker.passes}/{args.passes or "inf"} to {tracker.holder}', flush=True)
            elif event == 'drop':
                print(f'  t={t:6.2f}s drop {tracker.drops}, puck back to {tracker.holder}', flush=True)
            elif event == 'stall':
                print(f'  t={t:6.2f}s puck stopped in the middle zone ({tracker.stalls}), back to {tracker.holder}',
                      flush=True)
            if tracker.done:
                break
            b, v = task.puck()
            goal = task.goal()
            roles = goal['roles']
            tic = time.perf_counter()
            d.ctrl[:] = mppi.step(goal, d.ctrl.copy())
            compute = 1e3 * (time.perf_counter() - tic)
            for _ in range(substeps):
                mujoco.mj_step(m, d)

            a = d.xpos[proxy['left']][:, None]
            c = d.xpos[proxy['right']][None]
            clearance = float(np.sqrt(((a - c) ** 2).sum(-1)).min())
            for key, value in (('time', t), ('ctrl', d.ctrl.copy()), ('qpos', d.qpos.copy()),
                               ('qvel', d.qvel.copy()), ('puck_pos', b), ('puck_vel', v),
                               ('left_ee', robot.ee('left')), ('right_ee', robot.ee('right')),
                               ('left_role', roles['left']), ('right_role', roles['right']),
                               ('passes', tracker.passes), ('drops', tracker.drops), ('stalls', tracker.stalls),
                               ('clearance', clearance), ('compute_ms', compute)):
                log[key].append(value)
            # The side view shows the left arm on the left of the screen.
            if not display.update((f'left: {roles["left"]}', f'right: {roles["right"]}')):
                break
            if args.print_every and step % args.print_every == 0:
                print(f't={t:6.2f}s puck x={b[0]:+.2f} y={b[1]:+.2f} vx={v[0]:+.2f} '
                      f'left={roles["left"]:8s} right={roles["right"]:8s} '
                      f'passes={tracker.passes} drops={tracker.drops} compute={compute:5.1f}ms', flush=True)
    finally:
        mppi.close()
        display.close()

    duration = float(log['time'][-1]) if log['time'] else 0.0
    result = {
        'success': tracker.done,
        'duration_s': duration,
        'passes': tracker.passes,
        'drops': tracker.drops,
        'stalls': tracker.stalls,
        'mean_pass_interval_s': float(np.mean(np.diff([0.0] + tracker.pass_times))) if tracker.pass_times
        else float('nan'),
        'min_clearance_m': float(np.min(log['clearance'])) if log['clearance'] else float('nan'),
        'mean_compute_ms': float(np.mean(log['compute_ms'])) if log['compute_ms'] else 0.0,
    }
    print(f'success={result["success"]} time={duration:.2f}s passes={tracker.passes} drops={tracker.drops} '
          f'stalls={tracker.stalls} pass_interval={result["mean_pass_interval_s"]:.2f}s min_clearance={result["min_clearance_m"]:.3f}m '
          f'compute={result["mean_compute_ms"]:.1f}ms', flush=True)
    if args.save:
        Path(args.save).parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(args.save, **{k: np.asarray(v) for k, v in log.items()},
                            layout=args.layout, sampler=args.sampler, passes_requested=args.passes,
                            pass_times=np.asarray(tracker.pass_times), success=result['success'])
        print(f'Saved {args.save}')
    return result


def main():
    run(make_parser().parse_args())


if __name__ == '__main__':
    main()
