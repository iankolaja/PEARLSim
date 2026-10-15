"""Metrics in physical units: reactivity errors in pcm, stratified by how close to critical.

R^2 over the whole v1 data is dominated by deeply subcritical start-up states
(rho down to -675,000 pcm), so the headline numbers are pcm errors within
|rho| bands (review #4), compared with the per-step noise of the data itself.
"""

import numpy as np

from pearlsim.ml.features import reactivity

BANDS_PCM = (500, 2000, 5000)
# v1 total per-step reactivity noise (Monte Carlo + pebble reshuffle), 1 sigma;
# docs/review/noise_floor.py. v2 records carry per-step replica spreads instead.
V1_NOISE_FLOOR_PCM = 265.0


def _summary(error):
    error = np.asarray(error, dtype=float)
    if error.size == 0:
        return {"n": 0, "rmse": None, "mae": None, "bias": None}
    return {"n": int(error.size), "rmse": float(np.sqrt(np.mean(error ** 2))),
            "mae": float(np.mean(np.abs(error))), "bias": float(np.mean(error))}


def reactivity_metrics(k_true, k_pred, bands_pcm=BANDS_PCM, noise_pcm=None):
    """Reactivity error (pcm) overall and within |rho_true| < band, plus R^2 of rho for reference.

    noise_pcm: per-sample (or scalar) 1-sigma noise of the true values. The
    ``excess_rmse`` of a band is sqrt(max(rmse^2 - mean noise^2, 0)), the error not
    explained by noise in the reference data.
    """
    rho_true, rho_pred = reactivity(k_true) * 1e5, reactivity(k_pred) * 1e5
    error = rho_pred - rho_true
    noise = None if noise_pcm is None else np.broadcast_to(np.asarray(noise_pcm, dtype=float), error.shape)
    result = {"all": _summary(error)}
    for band in bands_pcm:
        inside = np.abs(rho_true) < band
        entry = _summary(error[inside])
        if noise is not None and entry["n"]:
            entry["noise_rms"] = float(np.sqrt(np.mean(noise[inside] ** 2)))
            entry["excess_rmse"] = float(np.sqrt(max(entry["rmse"] ** 2 - entry["noise_rms"] ** 2, 0.0)))
        result[f"abs_rho_lt_{band}"] = entry
    result["outside_largest_band"] = _summary(error[np.abs(rho_true) >= max(bands_pcm)])
    ss_res, ss_tot = np.sum(error ** 2), np.sum((rho_true - rho_true.mean()) ** 2)
    result["r2_rho"] = float(1 - ss_res / ss_tot) if ss_tot > 0 else None
    return result


def horizon_curve(k_true, k_pred, mask, band_pcm=2000):
    """Near-critical reactivity RMSE (pcm) for each horizon; arrays are (N, H+1)."""
    curve = []
    for h in range(k_true.shape[1]):
        available = mask[:, h]
        inside = available & (np.abs(reactivity(k_true[:, h]) * 1e5) < band_pcm)
        error = (reactivity(k_pred[inside, h]) - reactivity(k_true[inside, h])) * 1e5
        curve.append({"horizon": h, **_summary(error)})
    return curve


def regression_metrics(y_true, y_pred):
    """RMSE, MAE and R^2 in the target's own units."""
    y_true, y_pred = np.asarray(y_true, dtype=float), np.asarray(y_pred, dtype=float)
    error = y_pred - y_true
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    return {**_summary(error), "r2": float(1 - np.sum(error ** 2) / ss_tot) if ss_tot > 0 else None}
