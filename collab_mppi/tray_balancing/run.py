"""Two Pandas hold a large tray by two opposite corners and balance a ball at its centre, with centralized or decentralized MPPI.

    python -m collab_mppi.tray_balancing.run --sampler spike --layout centralized --viewer --realtime
    python -m collab_mppi.tray_balancing.run --sampler pink --layout decentralized --mp4 results/tray.mp4
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
from .task import BalanceConfig, BalanceTask

SCENE = ASSETS / 'tray_balancing_scene.xml'
# Side view, high enough for the ball on the tray to stay visible.
TOP_VIEW = ((0.0, 0.0, 0.3), 1.5, 90.0, -30.0)


def make_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--steps', type=int, default=2000, help='control steps')
    p.add_argument('--control-dt', type=float, default=0.01)
    # Initial ball state of Franka_MPPI's balancing task, in the tray frame.
    p.add_argument('--offset', type=float, nargs=2, default=(0.035, -0.025), help='initial ball offset, m')
    p.add_argument('--velocity', type=float, nargs=2, default=(0.09, 0.0), help='initial ball velocity, m/s')
    p.add_argument('--rot-noise-std', type=float, default=0.03, help='rotational exploration, rad')
    p.add_argument('--w-lag', type=float, default=20.0,
                   help='penalty on squared joint target - joint position (the arms pulling against each other)')
    p.add_argument('--posture-gain', type=float, default=0.1,
                   help='fraction of the null-space distance to the home posture removed per MPPI update')
    add_mppi_args(p)
    add_output_args(p, 'tray_balancing')
    return p


def run(args) -> dict:
    robot = DualFranka(scene_xml=SCENE, tool='gripper')
    m, d = robot.model, robot.data
    substeps = max(1, int(round(args.control_dt / m.opt.timestep)))
    task = BalanceTask(robot, BalanceConfig())
    robot.home_q = task.grasp()
    task.place_ball(args.offset, args.velocity)
    mppi = CollaborativeMPPI(robot, ARMS, mppi_config(args, rot_noise_std=args.rot_noise_std, w_lag=args.w_lag,
                                                      posture_gain=args.posture_gain, ik_damping=0.05),
                             task.cost, layout=args.layout)
    display = Display(m, d, args, view=TOP_VIEW)

    log = {k: [] for k in ('time', 'ctrl', 'qpos', 'ball_xy', 'ball_v', 'ball_gap', 'tray_pos', 'tilt',
                           'left_ee', 'right_ee', 'compute_ms')}
    print(f'tray balancing layout={args.layout} sampler={args.sampler}', flush=True)
    fallen = False
    try:
        for step in range(args.steps):
            display.start_step()
            t = step * args.control_dt
            tic = time.perf_counter()
            d.ctrl[:] = mppi.step(None, d.ctrl.copy())
            compute = 1e3 * (time.perf_counter() - tic)
            for _ in range(substeps):
                mujoco.mj_step(m, d)

            xy, v, gap = task.ball()
            for key, value in (('time', t), ('ctrl', d.ctrl.copy()), ('qpos', d.qpos.copy()),
                               ('ball_xy', xy), ('ball_v', v), ('ball_gap', gap),
                               ('tray_pos', d.sensordata[task.tray_pos].copy()), ('tilt', task.tilt()),
                               ('left_ee', robot.ee('left')), ('right_ee', robot.ee('right')),
                               ('compute_ms', compute)):
                log[key].append(value)
            if not display.update((f'ball {1e3 * np.linalg.norm(xy):.0f} mm',
                                   f'tilt {np.degrees(task.tilt()):.1f} deg')):
                break
            if args.print_every and step % args.print_every == 0:
                print(f't={t:6.2f}s ball x={1e3 * xy[0]:+5.0f}mm y={1e3 * xy[1]:+5.0f}mm '
                      f'speed={np.linalg.norm(v):.3f}m/s tilt={np.degrees(task.tilt()):4.1f}deg '
                      f'tray_drift={1e3 * np.linalg.norm(d.sensordata[task.tray_pos] - task.start):3.0f}mm '
                      f'compute={compute:5.1f}ms', flush=True)
            if task.fallen():
                fallen = True
                print(f'  t={t:6.2f}s the ball fell off the tray', flush=True)
                break
    finally:
        mppi.close()
        display.close()

    err = np.linalg.norm(log['ball_xy'], axis=1) if log['ball_xy'] else np.array([np.nan])
    drift = np.linalg.norm(np.asarray(log['tray_pos']) - task.start, axis=1) if log['tray_pos'] else [np.nan]
    result = {
        'success': not fallen and len(log['time']) == args.steps,
        'duration_s': float(log['time'][-1]) if log['time'] else 0.0,
        'fallen': fallen,
        'mean_xy_error_m': float(np.mean(err)),
        'final_xy_error_m': float(err[-1]),
        'max_tilt_deg': float(np.degrees(np.max(log['tilt']))) if log['tilt'] else float('nan'),
        'max_tray_drift_m': float(np.max(drift)),
        'mean_compute_ms': float(np.mean(log['compute_ms'])) if log['compute_ms'] else 0.0,
    }
    print(f'success={result["success"]} time={result["duration_s"]:.2f}s '
          f'mean_error={1e3 * result["mean_xy_error_m"]:.1f}mm final_error={1e3 * result["final_xy_error_m"]:.1f}mm '
          f'max_tilt={result["max_tilt_deg"]:.1f}deg max_drift={1e3 * result["max_tray_drift_m"]:.0f}mm '
          f'compute={result["mean_compute_ms"]:.1f}ms', flush=True)
    if args.save:
        Path(args.save).parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(args.save, **{k: np.asarray(v) for k, v in log.items()},
                            layout=args.layout, sampler=args.sampler, success=result['success'])
        print(f'Saved {args.save}')
    return result


def main():
    run(make_parser().parse_args())


if __name__ == '__main__':
    main()
