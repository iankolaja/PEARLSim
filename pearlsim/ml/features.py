"""Features and target transforms computed on whole sequences, before windowing.

Everything here operates on a full sequence history, so a feature at row i never
depends on where a window starts (review #6: v1 recomputed power per pebble on the
8-row steering window with fabricated pre-history).
"""

import fnmatch

import numpy as np
import pandas as pd

V1_CORE_PEBBLES = 250_131  # pebbles in the v1 gFHR model (zone maps of gFHR_benchmark_1975)

# Discharge measurements as used in the paper (dependent inputs)
DEPENDENT_FEATURES = (["R1_avg_FIMA_last_pass", "R2_avg_FIMA_last_pass", "R3_avg_FIMA_last_pass",
                       "R4_avg_FIMA_last_pass"] + [f"burnup_bin{b}_count" for b in range(1, 10)]
                      + ["avg_FIMA", "num_discarded"])
# v2 alternative to the hard burnup bins: Gaussian-smoothed discharge burnup (records.smoothed_discharge_burnup)
DEPENDENT_FEATURES_SMOOTH = (DEPENDENT_FEATURES[:4] + [f"burnup_smooth_bin{b}_frac" for b in range(1, 10)]
                             + ["burnup_smooth_p10", "burnup_smooth_p50", "burnup_smooth_p90",
                                "discharge_fuel_pebbles", "avg_FIMA", "num_discarded"])
CONTROL_FEATURES = ["power", "burnup_step", "graphite_insertion_fraction", "control_rod_position",
                    "fima_discharge_threshold"]

# Paper Table "MAPE applied to each dependent input variable" (fractions); sigma = MAPE / sqrt(2/pi)
PAPER_NOISE_MAPE = {"burnup_bin*_count": 0.05, "avg_FIMA": 0.025, "num_discarded": 0.05,
                    "R*_avg_FIMA_last_pass": 0.10,
                    # provisional, not from the paper: smoothed fractions as bin counts, quantiles as
                    # avg_FIMA, the discharged fuel count as num_discarded
                    "burnup_smooth_bin*_frac": 0.05, "burnup_smooth_p*": 0.025, "discharge_fuel_pebbles": 0.05}


def v1_power_per_fuel_pebble(frame, num_pebbles=V1_CORE_PEBBLES, residence_steps=10):
    """v1 estimate of power per fuel pebble from the insertion history (``add_computed_features``).

    The fuel fraction in core is the mean fuel insertion fraction over the current and
    previous ``residence_steps - 1`` steps, with the history before the first row taken
    equal to the first row. v2 records have the exact count (``power_per_fuel_pebble``).
    """
    fuel = 1.0 - frame["graphite_insertion_fraction"].to_numpy(dtype=float)
    padded = np.concatenate([np.full(residence_steps - 1, fuel[0]), fuel])
    fraction = np.convolve(padded, np.ones(residence_steps) / residence_steps, mode="valid")
    return frame["power"].to_numpy(dtype=float) / (num_pebbles * fraction)


BED_BOTTOM, BED_HEIGHT = 60.0, 309.47  # cm, gFHR pebble bed (rod tip 369.47 = withdrawn, 60 = inserted)
# Total rod worth in delta-k (k_out - k_in). M3 runs at 9 core states (fresh cores at 10-100% fuel density, run-in
# and equilibrium; dataset_v2_design.md §1c): 0.035-0.044, fresh ~0.042, burned ~0.036, while the pcm worth
# varies 2,100-4,900 mostly through 1/(k_in k_out).
ROD_WORTH_TOTAL_DK = 0.039


def rod_worth_feature(frame, total_dk=ROD_WORTH_TOTAL_DK, steps_per_transit=10, initial_front=1.0):
    """First-order rod worth (delta-k) from the control history (operator-observable), as a transformer input.

    dk(t) = total_dk x S(x), with S(x) = x - sin(2 pi x) / (2 pi) the textbook S-curve and x the fraction of
    the fuelled bed height the rods reach from above. The fuel front (fraction of the bed height holding fuel)
    starts at ``initial_front`` (1 = full core, 0 = all-graphite start) and rises one axial zone
    (1 / steps_per_transit) per step that inserts fuel (graphite_insertion_fraction < 1).
    In delta-k the rods are worth about the same in every state measured; converting to reactivity needs k
    (rho worth = dk / (k_in k_out)), which the forecaster predicts. Returns the inserted delta-k per row.
    """
    fuel = 1.0 - frame["graphite_insertion_fraction"].to_numpy(dtype=float)
    front = np.empty_like(fuel)
    level = initial_front
    for i, f in enumerate(fuel):
        level = min(1.0, level + (1.0 / steps_per_transit if f > 0 else 0.0))
        front[i] = level
    top = BED_BOTTOM + front * BED_HEIGHT
    z = frame["control_rod_position"].to_numpy(dtype=float)
    with np.errstate(invalid="ignore", divide="ignore"):
        x = np.clip(np.where(front > 0, (top - z) / (top - BED_BOTTOM), 0.0), 0.0, 1.0)
    return total_dk * (x - np.sin(2 * np.pi * x) / (2 * np.pi))


def reactivity(k):
    """rho = (k - 1) / k (dimensionless; multiply by 1e5 for pcm)."""
    k = np.asarray(k, dtype=float)
    return (k - 1.0) / k


def near_critical_weights(rho, band_pcm=None, outside_weight=1.0):
    """Sample weights: 1 within |rho| < band_pcm, outside_weight elsewhere (0 drops those samples)."""
    rho = np.asarray(rho, dtype=float)
    if band_pcm is None:
        return np.ones_like(rho)
    return np.where(np.abs(rho) * 1e5 < band_pcm, 1.0, outside_weight)


class NoiseModel:
    """Multiplicative Gaussian measurement noise: x * N(1, MAPE / sqrt(2/pi)), per column pattern."""

    def __init__(self, mape=None):
        self.mape = dict(PAPER_NOISE_MAPE if mape is None else mape)

    def sigma(self, column):
        for pattern, mape in self.mape.items():
            if fnmatch.fnmatchcase(column, pattern):
                return mape / np.sqrt(2.0 / np.pi)
        return 0.0

    def sigmas(self, columns):
        return np.array([self.sigma(c) for c in columns])

    def apply(self, values, columns, rng):
        """Noisy copy of ``values`` (..., len(columns)); columns without a pattern are unchanged."""
        sigma = self.sigmas(columns)
        values = np.asarray(values, dtype=float)
        return values * (1.0 + sigma * rng.standard_normal(values.shape))


class Standardizer:
    """Per-column mean/std scaling fitted on training data (constant columns keep std 1)."""

    def fit(self, values):
        values = np.asarray(values, dtype=float).reshape(-1, np.shape(values)[-1])
        self.mean = values.mean(axis=0)
        std = values.std(axis=0)
        self.std = np.where(std > 0, std, 1.0)
        return self

    def transform(self, values):
        return (np.asarray(values, dtype=float) - self.mean) / self.std

    def inverse(self, values):
        return np.asarray(values, dtype=float) * self.std + self.mean


class MeshPCA:
    """PCA of a mesh (columns matching a prefix) fitted on training sequences only.

    Scores are standardised with their training std (review #12: raw PCA scores have
    variance equal to the eigenvalue, so the first component would dominate the loss).
    """

    def __init__(self, prefix, n_components=5):
        self.prefix, self.n_components = prefix, n_components

    def columns(self, frame):
        return [c for c in frame.columns if c.startswith(self.prefix) and c[len(self.prefix):][:1] == "R"]

    def fit(self, frames):
        from sklearn.decomposition import PCA

        self.mesh_columns = self.columns(frames[0])
        data = np.concatenate([f[self.mesh_columns].to_numpy(dtype=float) for f in frames])
        self.pca = PCA(n_components=self.n_components).fit(data)
        self.score_std = self.pca.transform(data).std(axis=0)
        return self

    @property
    def names(self):
        return [f"{self.prefix}_pc{j + 1}" for j in range(self.n_components)]

    def add_scores(self, frame):
        """Copy of frame with standardised PC score columns ``<prefix>_pc<j>`` added."""
        scores = self.pca.transform(frame[self.mesh_columns].to_numpy(dtype=float)) / self.score_std
        return pd.concat([frame, pd.DataFrame(scores, columns=self.names, index=frame.index)], axis=1)

    def reconstruct(self, scores, components=None):
        """Mesh from standardised scores (N, n_components); components: indices to keep (others zeroed)."""
        scores = np.asarray(scores, dtype=float) * self.score_std
        if components is not None:
            keep = np.zeros(self.n_components, dtype=bool)
            keep[list(components)] = True
            scores = np.where(keep, scores, 0.0)
        return self.pca.inverse_transform(scores)
