# ML experiment configs

Run one config with `python -m pearlsim.ml.experiment experiments/<name>.yaml` (results go to
`ml_runs/<name>/`). Keys left out take the defaults in `pearlsim/ml/experiment.py` (`DEFAULTS`).
Neural-network configs are meant for batch jobs (`experiments/run_experiment.sub`), not for the
interactive node.

v1 data (`sequence_data/`) is a development set only (docs/refactor_plan.md Phase E): its results
are never reported as v2 results and the two are never mixed.

| Config | Purpose |
|---|---|
| `v1_ridge_k_near_critical.yaml` | corrected evaluation, ridge on k fitted within ±5000 pcm (review baseline, ~1,200 pcm expected near critical) |
| `v1_hgb_k_near_critical.yaml` | same with gradient boosting (~1,000 pcm expected) |
| `v1_paper_port_lstm.yaml` | E5: the paper's per-target LSTMs on the paper's leaky split and alignment, to check the PyTorch port against the published R² |
| `v1_lstm.yaml`, `v1_gru.yaml`, `v1_tcn.yaml`, `v1_transformer.yaml` | corrected evaluation of each sequence model |

The v1 test families were drawn once with `choose_test_families(families, 0.2, seed=2026)` and are
written into each corrected config, so they stay fixed while models and settings are tuned on the
validation folds.
