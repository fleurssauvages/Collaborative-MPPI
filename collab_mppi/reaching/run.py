"""Two Pandas facing each other reach Cartesian targets, each driven by MPPI or a Cartesian PID.

    python -m collab_mppi.reaching.run --sampler spike --layout centralized --viewer
    python -m collab_mppi.reaching.run --left mppi --right pid --scenario cross
    python -m collab_mppi.reaching.run --left pid --right pid --mp4 results/pid.mp4
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import mujoco
import numpy as np

from ..controllers import CartesianPID, CollaborativeMPPI, PIDConfig
from ..robot import ARMS, PROXY_BODIES, DualFranka
from ..sim import Display, add_mppi_args, add_output_args, mppi_config
from .targets import SCENARIOS, TargetSequencer, load_targets


def target_cost(robot: DualFranka, w_target: float = 1.0, w_terminal: float = 10.0):
    """Squared fingertip-target distance over the horizon plus a terminal term."""
    def cost(sensors, arm, targets):
        err = np.sum((sensors[:, :, robot.sensor_ee[arm]] - targets[arm]) ** 2, axis=2)
        return w_target * err.mean(axis=1) + w_terminal * err[:, -1]
    return cost


def make_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--left', choices=('mppi', 'pid'), default='mppi', help='left arm controller')
    p.add_argument('--right', choices=('mppi', 'pid'), default='mppi', help='right arm controller')
    p.add_argument('--scenario', choices=tuple(SCENARIOS), default='meet')
    p.add_argument('--targets-file', default=None, help='JSON {"left": [[x,y,z],...], "right": [...]}')
    p.add_argument('--base-distance', type=float, default=1.5, help='distance between the two bases, m')
    p.add_argument('--steps', type=int, default=3000, help='control steps')
    p.add_argument('--control-dt', type=float, default=0.01)
    p.add_argument('--tolerance', type=float, default=0.025, help='target validation radius, m')
    p.add_argument('--hold', type=float, default=0.5, help='time to hold inside the radius, s')
    p.add_argument('--cycles', type=int, default=1,
                   help='times to repeat each target list (0 repeats until --steps)')
    add_mppi_args(p)
    # PID
    p.add_argument('--kp', type=float, default=3.0)
    p.add_argument('--ki', type=float, default=0.5)
    p.add_argument('--kd', type=float, default=0.1)
    p.add_argument('--max-speed', type=float, default=0.5, help='PID Cartesian speed limit, m/s')
    add_output_args(p, 'reaching')
    return p


def run(args) -> dict:
    robot = DualFranka(args.base_distance)
    m, d = robot.model, robot.data
    substeps = max(1, int(round(args.control_dt / m.opt.timestep)))
    targets = load_targets(args.scenario, args.targets_file)
    seq = {arm: TargetSequencer(targets[arm], args.tolerance, args.hold, args.cycles) for arm in ARMS}
    choice = {'left': args.left, 'right': args.right}
    mppi_arms = [arm for arm in ARMS if choice[arm] == 'mppi']
    mppi = None
    if mppi_arms:
        mppi = CollaborativeMPPI(robot, mppi_arms, mppi_config(args), target_cost(robot), layout=args.layout)
    pid_cfg = PIDConfig(kp=args.kp, ki=args.ki, kd=args.kd, max_speed=args.max_speed)
    pids = {arm: CartesianPID(robot, arm, pid_cfg, args.control_dt) for arm in ARMS if choice[arm] == 'pid'}
    display = Display(m, d, args)

    log = {k: [] for k in ('time', 'ctrl', 'qpos', 'left_ee', 'right_ee', 'left_target',
                           'right_target', 'clearance', 'compute_ms')}
    proxy = {arm: [m.body(f'{arm}_{b}').id for b in PROXY_BODIES]
             for arm in ARMS}
    print(f'left={args.left} right={args.right} layout={args.layout} sampler={args.sampler} '
          f'scenario={args.scenario if not args.targets_file else args.targets_file}', flush=True)
    try:
        for step in range(args.steps):
            display.start_step()
            t = step * args.control_dt
            for arm in ARMS:
                if seq[arm].update(robot.ee(arm), t):
                    print(f'  t={t:6.2f}s {arm} reached target {seq[arm].index}/{seq[arm].total or "inf"}',
                          flush=True)
                    if arm in pids:
                        pids[arm].reset_integral()
                robot.set_target_marker(arm, seq[arm].current)
            if all(s.done for s in seq.values()):
                break
            goal = {arm: seq[arm].current for arm in ARMS}
            command = d.ctrl.copy()
            for arm, pid in pids.items():
                command[robot.arm_act[arm]] = pid.step(goal[arm])
            compute = 0.0
            if mppi is not None:
                tic = time.perf_counter()
                mppi_command = mppi.step(goal, command)
                compute = 1e3 * (time.perf_counter() - tic)
                for arm in mppi_arms:
                    command[robot.arm_act[arm]] = mppi_command[robot.arm_act[arm]]
            d.ctrl[:] = command
            for _ in range(substeps):
                mujoco.mj_step(m, d)

            a = d.xpos[proxy['left']][:, None]
            b = d.xpos[proxy['right']][None]
            clearance = float(np.sqrt(((a - b) ** 2).sum(-1)).min())
            for key, value in (('time', t), ('ctrl', d.ctrl.copy()), ('qpos', d.qpos.copy()),
                               ('left_ee', robot.ee('left')), ('right_ee', robot.ee('right')),
                               ('left_target', goal['left']), ('right_target', goal['right']),
                               ('clearance', clearance), ('compute_ms', compute)):
                log[key].append(value)
            if not display.update():
                break
            if args.print_every and step % args.print_every == 0:
                errs = ' '.join(f'{arm}={1e3 * np.linalg.norm(robot.ee(arm) - goal[arm]):5.1f}mm' for arm in ARMS)
                print(f't={t:6.2f}s err {errs} clearance={clearance:.3f}m compute={compute:5.1f}ms', flush=True)
    finally:
        if mppi is not None:
            mppi.close()
        display.close()

    ctrl = np.asarray(log['ctrl'])
    smooth = {arm: float(np.linalg.norm(np.diff(ctrl[:, robot.arm_act[arm]], axis=0), axis=1).mean())
              for arm in ARMS} if len(ctrl) > 1 else {arm: float('nan') for arm in ARMS}
    result = {
        'success': all(s.done for s in seq.values()),
        'duration_s': float(log['time'][-1]) if log['time'] else 0.0,
        'reached': {arm: f'{seq[arm].index}/{seq[arm].total or "inf"}' for arm in ARMS},
        'min_clearance_m': float(np.min(log['clearance'])) if log['clearance'] else float('nan'),
        'command_smoothness_l2': smooth,
        'mean_compute_ms': float(np.mean(log['compute_ms'])) if log['compute_ms'] else 0.0,
    }
    print(f'success={result["success"]} time={result["duration_s"]:.2f}s reached={result["reached"]} '
          f'min_clearance={result["min_clearance_m"]:.3f}m smoothness='
          + ' '.join(f'{arm}={v:.4f}' for arm, v in smooth.items())
          + f' compute={result["mean_compute_ms"]:.1f}ms', flush=True)
    if args.save:
        Path(args.save).parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(args.save, **{k: np.asarray(v) for k, v in log.items()},
                            left_controller=args.left, right_controller=args.right, layout=args.layout,
                            sampler=args.sampler, cycles=args.cycles, success=result['success'],
                            left_targets=targets['left'], right_targets=targets['right'])
        print(f'Saved {args.save}')
    return result


def main():
    run(make_parser().parse_args())


if __name__ == '__main__':
    main()
