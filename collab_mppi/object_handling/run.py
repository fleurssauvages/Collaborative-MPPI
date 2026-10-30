"""Robot 1 hands three boxes to robot 2, which puts them on a second table, with centralized or decentralized MPPI.

    python -m collab_mppi.object_handling.run --sampler spike --layout centralized --viewer --realtime
    python -m collab_mppi.object_handling.run --sampler pink --layout decentralized --mp4 results/handling.mp4
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import mujoco
import numpy as np

from ..controllers import CollaborativeMPPI
from ..robot import ARMS, ASSETS, DualFranka
from ..sim import Display, add_mppi_args, add_output_args, mppi_config
from .task import BASES, GIVER, GRASP_ROT, HOME_Q_GRASP, N_BOXES, TABLE_TOP, TAKER, HandlingConfig, HandlingTask

SCENE = ASSETS / 'object_handling_scene.xml'
# Front view: from beyond the tables, looking back at both robots.
FRONT_VIEW = ((0.35, 0.0, 0.3), 2.0, 180.0, -30.0)


def make_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--steps', type=int, default=6000, help='control steps')
    p.add_argument('--control-dt', type=float, default=0.01)
    p.add_argument('--orient-gain', type=float, default=0.5,
                   help='fraction of the gripper orientation error removed per MPPI update')
    p.add_argument('--posture-gain', type=float, default=0.1,
                   help='fraction of the null-space distance to the home posture removed per MPPI update')
    p.add_argument('--smooth', type=float, default=0.2,
                   help='low-pass factor on the applied arm commands (1: none); MPPI jitter shakes held boxes loose')
    add_mppi_args(p, clearance=0.07)
    add_output_args(p, 'object_handling')
    return p


def run(args) -> dict:
    robot = DualFranka(scene_xml=SCENE, tool='gripper', bases=BASES)
    robot.reset(HOME_Q_GRASP)
    m, d = robot.model, robot.data
    substeps = max(1, int(round(args.control_dt / m.opt.timestep)))
    task = HandlingTask(robot, HandlingConfig(dt=args.control_dt))
    mppi = CollaborativeMPPI(robot, ARMS, mppi_config(args, floor_z=TABLE_TOP + 0.012,
                                                      orient_target={arm: GRASP_ROT for arm in ARMS},
                                                      orient_gain=args.orient_gain, posture_gain=args.posture_gain,
                                                      ik_damping=0.05),
                             task.arm_cost, layout=args.layout)
    display = Display(m, d, args, view=FRONT_VIEW)

    log = {k: [] for k in ('time', 'ctrl', 'qpos', 'left_ee', 'right_ee', 'left_target', 'right_target',
                           'left_phase', 'right_phase', 'boxes', 'compute_ms')}
    print(f'object handling layout={args.layout} sampler={args.sampler}', flush=True)
    try:
        for step in range(args.steps):
            display.start_step()
            t = step * args.control_dt
            for event in task.update(t):
                print(f'  t={t:6.2f}s {event}', flush=True)
            if task.done:
                break
            goal = task.goal()
            command = d.ctrl.copy()
            for arm in ARMS:
                command[robot.grip_act[arm]] = task.target(arm)[1]
            tic = time.perf_counter()
            planned = mppi.step(goal, command)
            compute = 1e3 * (time.perf_counter() - tic)
            for arm in ARMS:
                act = robot.arm_act[arm]
                planned[act] = d.ctrl[act] + args.smooth * (planned[act] - d.ctrl[act])
            d.ctrl[:] = planned
            for _ in range(substeps):
                mujoco.mj_step(m, d)

            for key, value in (('time', t), ('ctrl', d.ctrl.copy()), ('qpos', d.qpos.copy()),
                               ('left_ee', robot.ee('left')), ('right_ee', robot.ee('right')),
                               ('left_target', goal['left']), ('right_target', goal['right']),
                               ('left_phase', task.arm['left'].phase), ('right_phase', task.arm['right'].phase),
                               ('boxes', np.array([task.box_pos(k) for k in range(N_BOXES)])),
                               ('compute_ms', compute)):
                log[key].append(value)
            if not display.update():
                break
            if args.print_every and step % args.print_every == 0:
                status = ' '.join(f'{name}={task.arm[arm].phase} ({1e3 * np.linalg.norm(robot.ee(arm) - task.target(arm)[0]):.0f}mm)'
                                  for name, arm in (('robot1', GIVER), ('robot2', TAKER)))
                print(f't={t:6.2f}s {status} compute={compute:5.1f}ms', flush=True)
    finally:
        mppi.close()
        display.close()

    placement = task.placement_errors()
    result = {
        'success': task.done,
        'duration_s': float(log['time'][-1]) if log['time'] else 0.0,
        'delivered': task.delivered,
        'drops': task.drops,
        'placement_error_m': placement,
        'mean_compute_ms': float(np.mean(log['compute_ms'])) if log['compute_ms'] else 0.0,
    }
    print(f'success={result["success"]} time={result["duration_s"]:.2f}s delivered={task.delivered}/{N_BOXES} '
          f'drops={task.drops} placement_error=' + ' '.join(f'{1e3 * e:.0f}mm' for e in placement)
          + f' compute={result["mean_compute_ms"]:.1f}ms', flush=True)
    if args.save:
        Path(args.save).parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(args.save, **{k: np.asarray(v) for k, v in log.items()},
                            layout=args.layout, sampler=args.sampler, success=result['success'],
                            slots=np.asarray(task.slots))
        print(f'Saved {args.save}')
    return result


def main():
    run(make_parser().parse_args())


if __name__ == '__main__':
    main()
