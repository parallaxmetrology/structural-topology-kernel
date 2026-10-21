"""Deterministic synthetic microstructure-like fields for golden tests."""
import numpy as np
from scipy.ndimage import gaussian_filter


def _grains(n=40, size=256, seed=0):
    rng = np.random.default_rng(seed)
    pts = rng.uniform(0, size, (n, 2))
    yy, xx = np.mgrid[0:size, 0:size]
    d = np.stack([np.hypot(yy - y, xx - x) for y, x in pts])
    s = np.sort(d, 0)
    return np.clip((s[1] - s[0]) / 6.0, 0, 1)  # bright grain interiors, dark boundaries


def _lamellae(size=256, period=9.0, angle=0.6):
    yy, xx = np.mgrid[0:size, 0:size]
    u = xx * np.cos(angle) + yy * np.sin(angle)
    return 0.5 + 0.5 * np.sign(np.sin(2 * np.pi * u / period))


def _particles(n=120, size=256, seed=1):
    rng = np.random.default_rng(seed)
    f = np.zeros((size, size))
    yy, xx = np.mgrid[0:size, 0:size]
    for y, x, r in zip(rng.uniform(0, size, n), rng.uniform(0, size, n), rng.uniform(2, 7, n)):
        f[np.hypot(yy - y, xx - x) < r] = 1.0
    return gaussian_filter(f, 1.0)


def _porous(size=256, seed=2):
    rng = np.random.default_rng(seed)
    return (gaussian_filter(rng.normal(size=(size, size)), 6) > 0).astype(float)


FIELDS = {"grains": _grains, "lamellae": _lamellae, "particles": _particles, "porous_binary": _porous}
