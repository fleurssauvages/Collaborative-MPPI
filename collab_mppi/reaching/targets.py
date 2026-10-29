"""Cartesian fingertip target sequences in the world frame, one list per arm.

The left base is at (-0.75, 0, 0) facing +x, the right base at (0.75, 0, 0)
facing -x; the shared workspace is around x = 0.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

SCENARIOS = {
    # Both fingertips repeatedly approach the centre and come within 16 cm.
    'meet': {
        'left':  [(-0.30, -0.20, 0.45), (-0.08, 0.00, 0.50), (-0.30, 0.20, 0.45), (-0.08, 0.00, 0.50)],
        'right': [(0.30, 0.20, 0.45), (0.08, 0.00, 0.50), (0.30, -0.20, 0.45), (0.08, 0.00, 0.50)],
    },
    # The arms swap lanes across the midline, passing each other near the centre.
    'cross': {
        'left':  [(-0.25, -0.25, 0.50), (0.00, 0.25, 0.50), (-0.25, -0.25, 0.50)],
        'right': [(0.25, 0.25, 0.50), (0.00, -0.25, 0.50), (0.25, 0.25, 0.50)],
    },
    # Independent reaching in each arm's own half of the workspace.
    'separate': {
        'left':  [(-0.35, -0.25, 0.50), (-0.30, 0.00, 0.62), (-0.35, 0.25, 0.50)],
        'right': [(0.35, 0.25, 0.50), (0.30, 0.00, 0.62), (0.35, -0.25, 0.50)],
    },
}


def load_targets(scenario: str, path: str | None = None) -> dict:
    """Built-in scenario, or a JSON file {"left": [[x,y,z], ...], "right": [...]}."""
    raw = json.loads(Path(path).read_text()) if path else SCENARIOS[scenario]
    return {arm: np.asarray(raw[arm], dtype=np.float64).reshape(-1, 3) for arm in ('left', 'right')}


class TargetSequencer:
    """Advances an arm to its next target after holding within tolerance.

    The target list is repeated `cycles` times; 0 repeats until the run ends.
    """

    def __init__(self, targets: np.ndarray, tolerance: float, hold_s: float, cycles: int = 1):
        if cycles < 0:
            raise ValueError('cycles must be nonnegative (0 repeats until --steps)')
        self.targets, self.tolerance, self.hold_s, self.cycles = targets, tolerance, hold_s, cycles
        self.index, self.inside_since, self.reached_times = 0, None, []

    @property
    def total(self) -> int | None:
        return len(self.targets) * self.cycles if self.cycles else None

    @property
    def done(self) -> bool:
        return self.total is not None and self.index >= self.total

    @property
    def current(self) -> np.ndarray:
        index = min(self.index, self.total - 1) if self.total is not None else self.index
        return self.targets[index % len(self.targets)]

    def update(self, ee: np.ndarray, t: float) -> bool:
        """Returns True when the current target was just validated."""
        if self.done:
            return False
        if np.linalg.norm(ee - self.current) > self.tolerance:
            self.inside_since = None
            return False
        if self.inside_since is None:
            self.inside_since = t
        if t - self.inside_since < self.hold_s:
            return False
        self.reached_times.append(t)
        self.index += 1
        self.inside_since = None
        return True
