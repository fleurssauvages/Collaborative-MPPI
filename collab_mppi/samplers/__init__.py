"""Unit-RMS MPPI proposal samplers: standard, pink, poisson and spike.

Each sampler returns perturbations of shape (n, horizon, channels) whose RMS is
one before scaling, so a single exploration sigma has the same meaning for all.
Standard, pink, Poisson and Spike follow the Franka_MPPI / Spike-MPPI laws.
"""
from __future__ import annotations

import math

import numpy as np

from .pink_noise import sample_pink_noise
from .poisson_kernels import StaticPoissonSampler, resolve_poisson_sampler
from .spike_motor_pool import SpikeMotorPoolSampler

SAMPLERS = ('standard', 'pink', 'poisson', 'spike')


def _twitch_kernel(dt, rise_s, decay_s, duration_s):
    steps = max(1, int(math.ceil(duration_s / dt)))
    t = (np.arange(steps, dtype=np.float64) + 1.0) * dt
    k = np.maximum(np.exp(-t / decay_s) - np.exp(-t / rise_s), 0.0)
    return k / max(float(k.max()), 1e-12)


class StandardSampler:
    """Stationary AR(1) temporally correlated Gaussian proposal."""

    def __init__(self, n, h, channels, *, rho=0.25, **_):
        self.shape, self.rho = (n, h, channels), float(rho)

    def __call__(self, rng):
        x = rng.standard_normal(self.shape)
        beta = math.sqrt(max(0.0, 1.0 - self.rho * self.rho))
        for t in range(1, self.shape[1]):
            x[:, t] = self.rho * x[:, t - 1] + beta * x[:, t]
        return x


class PinkSampler:
    def __init__(self, n, h, channels, **_):
        self.shape = (n, h, channels)

    def __call__(self, rng):
        return sample_pink_noise(rng, self.shape)


class PoissonSampler:
    """Signed marked Poisson events decoded by a common twitch kernel."""

    def __init__(self, n, h, channels, *, dt, base_rate_hz=8.0, levels=6, twitch_rise_s=0.016,
                 twitch_decay_s=0.064, twitch_duration_s=0.2, **_):
        strengths = np.arange(1, levels + 1, dtype=np.float64)
        strengths /= strengths.sum()
        amplitudes = np.cumsum(strengths)
        amplitudes[-1] = 1.0
        prior = np.full(levels, 1.0 / levels)
        kernel = _twitch_kernel(dt, twitch_rise_s, twitch_decay_s, twitch_duration_s)
        rate = np.full((h, channels), base_rate_hz / 2.0)
        marks = np.broadcast_to(prior, (h, channels, levels)).copy()
        self.sampler = StaticPoissonSampler(rate, rate.copy(), marks, amplitudes, dt, n, kernel,
                                            backend=resolve_poisson_sampler('auto'))
        # Expected-RMS normalization of the production controller.
        k2 = np.cumsum(kernel * kernel)
        remaining = np.minimum(np.arange(h, 0, -1), len(k2))
        variance = (k2[remaining - 1].sum() / h * dt * base_rate_hz
                    * float(np.sum(prior * amplitudes ** 2)))
        self.scale = 1.0 / math.sqrt(max(variance, 1e-12))

    def __call__(self, rng):
        out, _ = self.sampler.sample_projected_identity(rng)
        return np.array(out, dtype=np.float64) * self.scale


class SpikeSampler:
    """Antagonistic motor-pool proposal (common drive, recruitment, renewal, twitches)."""

    def __init__(self, n, h, channels, *, dt, base_rate_hz=8.0, levels=6, twitch_rise_s=0.016,
                 twitch_decay_s=0.064, twitch_duration_s=0.2, drive_sigma=0.2, drive_tau_s=0.05, **_):
        self.sampler = SpikeMotorPoolSampler(
            n=n, h=h, nu=channels, dt=dt, levels=levels, base_rate_hz=base_rate_hz,
            twitch_rise_s=twitch_rise_s, twitch_decay_s=twitch_decay_s,
            twitch_duration_s=twitch_duration_s, drive_sigma=drive_sigma,
            drive_tau_s=drive_tau_s, separate_antagonists=False, backend='auto')
        self.scale = 1.0 / max(self.sampler.estimate_unit_rms(), 1e-12)

    def __call__(self, rng):
        out, _ = self.sampler.sample(rng)
        return out * self.scale


def make_sampler(name: str, n: int, h: int, channels: int, **params):
    """Callable rng -> (n, h, channels) unit-RMS perturbations."""
    classes = {'standard': StandardSampler, 'pink': PinkSampler,
               'poisson': PoissonSampler, 'spike': SpikeSampler}
    if name not in classes:
        raise ValueError(f'unknown sampler {name!r}; choose from {SAMPLERS}')
    return classes[name](n, h, channels, **params)
