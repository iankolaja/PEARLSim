"""Rod-worth curves and the estimated critical rod position (docs dataset_v2_design.md §1c).

A worth curve w(z) >= 0 is the reactivity removed by inserting the rods to tip position z, relative to fully
withdrawn (z = 369.47 cm, the top of the bed; fully inserted is 60 cm). Measured curves are noisy, so the
points are first projected onto the closest non-decreasing curve in insertion (pool-adjacent-violators,
weighted by 1/sigma^2) and then interpolated with a monotone cubic (PCHIP), so the curve can be inverted.

For a step with reactivity rho_step (pcm) at rod position z_act, the critical position solves
    w(z_crit) = rho_step + w(z_act)
reported with a status (``critical``, ``subcritical_rods_out``, ``supercritical_rods_in``) and the rods'
shutdown margin W_tot - rho_step - w(z_act).
"""

from dataclasses import dataclass

import numpy as np
from scipy.interpolate import PchipInterpolator

WITHDRAWN = 369.47
INSERTED = 60.0


def isotonic(values, weights=None):
    """Weighted least-squares non-decreasing fit (pool-adjacent-violators)."""
    values = np.asarray(values, dtype=float)
    weights = np.ones_like(values) if weights is None else np.asarray(weights, dtype=float)
    blocks = []  # [mean, weight, count]
    for v, w in zip(values, weights, strict=True):
        blocks.append([v, w, 1])
        while len(blocks) > 1 and blocks[-2][0] > blocks[-1][0]:
            v2, w2, n2 = blocks.pop()
            v1, w1, n1 = blocks.pop()
            blocks.append([(v1 * w1 + v2 * w2) / (w1 + w2), w1 + w2, n1 + n2])
    return np.concatenate([np.full(n, v) for v, _, n in blocks])


@dataclass
class WorthCurve:
    """Monotone rod-worth curve from measured (z, w, sigma) points; must include fully withdrawn (w = 0)."""

    z: np.ndarray
    w: np.ndarray
    sigma: np.ndarray = None

    def __post_init__(self):
        z, w = np.asarray(self.z, dtype=float), np.asarray(self.w, dtype=float)
        order = np.argsort(-z)  # increasing insertion
        self.z, w = z[order], w[order]
        sigma = None if self.sigma is None else np.asarray(self.sigma, dtype=float)[order]
        if not np.isclose(self.z[0], WITHDRAWN):
            raise ValueError("a worth curve needs the fully withdrawn point")
        weights = None if sigma is None else 1.0 / np.maximum(sigma, 1e-9) ** 2
        self.w = isotonic(w - w[0], weights)
        self.w -= self.w[0]
        self.sigma = sigma
        # PCHIP needs strictly increasing x; insertion depth d = WITHDRAWN - z
        self._depth = WITHDRAWN - self.z
        self._curve = PchipInterpolator(self._depth, self.w)

    @property
    def total(self):
        return float(self.w[-1])

    def __call__(self, z):
        depth = np.clip(WITHDRAWN - np.asarray(z, dtype=float), self._depth[0], self._depth[-1])
        return self._curve(depth)

    def position_for_worth(self, worth):
        """Rod tip position z where w(z) = worth, for 0 <= worth <= total (bisection on the monotone curve)."""
        if not 0.0 <= worth <= self.total:
            raise ValueError(f"worth {worth} outside [0, {self.total}]")
        lo, hi = self._depth[0], self._depth[-1]
        for _ in range(60):
            mid = 0.5 * (lo + hi)
            if self._curve(mid) < worth:
                lo = mid
            else:
                hi = mid
        return float(WITHDRAWN - 0.5 * (lo + hi))


def critical_position(curve, rho_step, z_act):
    """(z_crit or None, status, shutdown margin in pcm) for a step with reactivity rho_step at rods z_act."""
    target = rho_step + float(curve(z_act))
    margin = curve.total - target
    if target < 0:
        return None, "subcritical_rods_out", margin
    if target > curve.total:
        return None, "supercritical_rods_in", margin
    return curve.position_for_worth(target), "critical", margin
