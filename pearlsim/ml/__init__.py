"""Machine learning on PEARLSim sequences (PyTorch models, scikit-learn baselines).

Only ``pearlsim.ml.models`` and ``pearlsim.ml.train`` import torch; the data,
feature, window, split and metric modules don't, so they load quickly.

Conventions (docs/refactor_plan.md Phase E):

* A *sequence* is one simulated operating history, one row per depletion step.
* Row i holds the controls u_i applied during step i, the discharge measurements
  D_i taken before step i (pebbles that left the core at the start of the step),
  and the step's results (k_eff, meshes).
* A window ending at row i uses rows i-W+1 .. i. Its targets are the reactivity of
  step i (and of later steps for multi-horizon models, given the planned future
  controls) and the next measurements D_{i+1} (review #3).
* Splits are by sequence family, never within a sequence (review #1); the test
  families are fixed in the experiment config before any tuning (review #2).
"""
