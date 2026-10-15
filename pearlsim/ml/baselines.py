"""Ridge and gradient-boosting baselines on flattened windows (direct, one model per horizon and target)."""

import numpy as np


def flatten(windows):
    """(N, W*F + H*C) design matrix: the past window followed by the planned future controls."""
    n = len(windows)
    return np.concatenate([windows.past.reshape(n, -1), windows.future_controls.reshape(n, -1)], axis=1)


class FlatBaseline:
    """One scikit-learn regressor per (horizon, target), fitted where that target is available."""

    def __init__(self, kind="ridge", **params):
        self.kind, self.params = kind, params

    def _regressor(self):
        if self.kind == "ridge":
            from sklearn.linear_model import RidgeCV
            from sklearn.pipeline import make_pipeline
            from sklearn.preprocessing import StandardScaler

            return make_pipeline(StandardScaler(), RidgeCV(alphas=self.params.get("alphas", np.logspace(-4, 3, 15))))
        if self.kind == "hgb":
            from sklearn.ensemble import HistGradientBoostingRegressor

            return HistGradientBoostingRegressor(max_iter=self.params.get("max_iter", 400),
                                                 learning_rate=self.params.get("learning_rate", 0.05),
                                                 random_state=self.params.get("random_state", 0))
        raise ValueError(f"unknown baseline {self.kind!r}")

    def fit(self, windows, weights=None):
        x = flatten(windows)
        _, horizons, targets = windows.targets.shape
        self.models = {}
        for h in range(horizons):
            for j in range(targets):
                available = windows.mask[:, h, j]
                w = None if weights is None else np.asarray(weights)[available]
                model = self._regressor()
                fit_params = {}
                if w is not None:
                    key = model.steps[-1][0] + "__sample_weight" if hasattr(model, "steps") else "sample_weight"
                    fit_params[key] = w
                self.models[h, j] = model.fit(x[available], windows.targets[available, h, j], **fit_params)
        return self

    def predict(self, windows):
        x = flatten(windows)
        out = np.zeros(windows.targets.shape, dtype=float)
        for (h, j), model in self.models.items():
            out[:, h, j] = model.predict(x)
        return out
