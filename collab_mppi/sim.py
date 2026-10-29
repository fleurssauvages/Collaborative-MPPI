"""Shared command-line options, viewer, camera and MP4 output for the task runners."""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import mujoco

from .controllers import MPPIConfig
from .samplers import SAMPLERS

RESULTS = Path(__file__).resolve().parent.parent / 'results'
# Viewer and MP4 camera: side view from above the shared workspace.
CAMERA_LOOKAT, CAMERA_DISTANCE, CAMERA_AZIMUTH, CAMERA_ELEVATION = (0.0, 0.0, 0.35), 1.5, 90.0, -38.0


def set_camera(camera):
    camera.lookat[:] = CAMERA_LOOKAT
    camera.distance, camera.azimuth, camera.elevation = CAMERA_DISTANCE, CAMERA_AZIMUTH, CAMERA_ELEVATION


def add_mppi_args(p: argparse.ArgumentParser, **defaults):
    p.add_argument('--layout', choices=('centralized', 'decentralized'), default='centralized',
                   help='with two MPPI arms: one MPPI over both, or one per arm')
    p.add_argument('--sampler', choices=SAMPLERS, default='spike')
    p.add_argument('--rollouts', type=int, default=32)
    p.add_argument('--horizon', type=int, default=20)
    p.add_argument('--noise-std', type=float, default=0.01, help='Cartesian exploration, m')
    p.add_argument('--clearance', type=float, default=0.15, help='arm-arm proxy clearance, m')
    p.add_argument('--threads', type=int, default=4)
    p.add_argument('--seed', type=int, default=0)
    p.set_defaults(**defaults)


def mppi_config(args, **overrides) -> MPPIConfig:
    return MPPIConfig(sampler=args.sampler, rollouts=args.rollouts, horizon=args.horizon,
                      control_dt=args.control_dt, noise_std=args.noise_std, clearance=args.clearance,
                      threads=args.threads, seed=args.seed, **overrides)


def add_output_args(p: argparse.ArgumentParser, task: str):
    default_save = RESULTS / task / 'last_run.npz'
    p.add_argument('--viewer', action='store_true')
    p.add_argument('--viewer-ui', action='store_true', help='show the MuJoCo viewer side panels')
    p.add_argument('--realtime', action='store_true', help='pace the viewer to wall-clock time')
    p.add_argument('--save', default=str(default_save),
                   help=f'NPZ recording path (default: results/{task}/last_run.npz)')
    p.add_argument('--no-save', dest='save', action='store_const', const=None, help='do not write an NPZ')
    p.add_argument('--mp4', default=None, help='render an MP4 (960x540)')
    p.add_argument('--print-every', type=int, default=100)


class Display:
    """Optional passive viewer and MP4 writer, updated once per control step."""

    def __init__(self, model, data, args):
        self.data, self.args = data, args
        self.viewer = self.renderer = self.writer = None
        self.tick = time.perf_counter()
        if args.viewer:
            from mujoco import viewer as mj_viewer
            self.viewer = mj_viewer.launch_passive(model, data, show_left_ui=args.viewer_ui,
                                                   show_right_ui=args.viewer_ui)
            set_camera(self.viewer.cam)
        if args.mp4:
            import imageio.v2 as imageio
            self.renderer = mujoco.Renderer(model, height=540, width=960)
            self.camera = mujoco.MjvCamera()
            set_camera(self.camera)
            Path(args.mp4).parent.mkdir(parents=True, exist_ok=True)
            self.writer = imageio.get_writer(args.mp4, fps=round(1 / args.control_dt), codec='libx264',
                                             quality=8, macro_block_size=1)

    def start_step(self):
        self.tick = time.perf_counter()

    def update(self) -> bool:
        """Draws the current state; returns False once the viewer window is closed."""
        if self.writer is not None:
            self.renderer.update_scene(self.data, self.camera)
            self.writer.append_data(self.renderer.render())
        if self.viewer is not None:
            if not self.viewer.is_running():
                return False
            self.viewer.sync()
            if self.args.realtime:
                time.sleep(max(0.0, self.args.control_dt - (time.perf_counter() - self.tick)))
        return True

    def close(self):
        if self.writer is not None:
            self.writer.close()
            print(f'Saved {self.args.mp4}')
        if self.viewer is not None:
            self.viewer.close()
