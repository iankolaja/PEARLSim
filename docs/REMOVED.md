# Removed components and how to recover them

The v2 refactor removes code and data that no longer belong in PEARLSim. Nothing is lost:

| Source | Holds |
|---|---|
| Git tag `v1-as-run` (commit `dd60745`) | every tracked file as it generated the v1 dataset |
| Branch `v1` | `v1-as-run` plus the previously untracked v1 inputs and scripts (run decks, steering instruction files, `ml_model_wrappers.py`, helper scripts) |
| Cleanup archive `/global/scratch/users/ikolaja/cleanup/PEARLSim_v1/` | untracked files moved out of the working tree, at the same relative paths. `MANIFEST.tsv` lists path, size, sha256 (or `skipped` for large data moved without hashing) and mtime. Moves are made with `scripts/archive_to_cleanup.py`. |

```bash
git show v1-as-run:pearlsim/core.py                    # view a file as it was
git checkout v1-as-run -- pearlsim/pebble_model.py      # restore a removed tracked file
git diff v1-as-run -- pearlsim/core.py                  # what the refactor changed
git switch v1                                            # the frozen v1 line, incl. formerly untracked inputs
# untracked material: copy back from the cleanup archive (same relative path); check MANIFEST.tsv
```

## 1. Gamma-spectrum measurement of discharged pebbles (published separately)

Serpent decay and photon runs producing discharged-pebble gamma spectra for the burnup-measurement
paper. That paper has its own published repository of key notebooks.

| Removed | Tracked? | Recover from |
|---|---|---|
| `pearlsim/serpent_spectrum_tools.py` | yes | `v1-as-run` |
| star imports of it in `results_processing.py`, `gather_sequence_results_*.py`; lazy exports in `pearlsim/__init__.py` | yes | `v1-as-run` |
| `produce_decay_spectra.py`, `decay_step{1,2}{,_batch}_template.serpent` | no | branch `v1`, or the cleanup archive |
| `decay.points`, `process_hxf_data.ipynb` | no | cleanup archive |
| `HxF_sample_pebbles/`, `HxF_sample_pebbles_dataset1/`, `HxF_sample_pebbles_dataset2/` (~114 GB HxF pebble data; moved without hashing) | no | cleanup archive |

## 2. ML pebble-wise depletion ("hyperfidelity"), tabled

ML models of per-pebble current, flux and cross sections (random forests), with OpenMC MicroXS
pebble depletion, as a fast stand-in for explicit pebble-wise depletion. Training data came from
Serpent runs with sample-pebble detectors and `set mdep`.

Code removed from files that remain (recover with `git show v1-as-run:<file>`):
- `pearlsim/core.py`: `Core.generate_sample_pebbles` (sample-pebble detectors and `set mdep` blocks);
  the `num_training_data` parameter and training branch of `Core.generate_input`; the unused
  `pebble_model` parameter of `Core.insert`.
- `pearlsim/simulation.py`: `Simulation.pebble_model`; keywords `load_pebble_model`,
  `load_velocity_profile`, `simulate_pebbles`, `sample_pebbles_from_step`, `generate_training_data`;
  the pebble-model pickle written by `save`. **Behaviour note:** `initialize` no longer calls
  `Pebble_Model.initialize_pebbles`, which drew from `np.random`. A deck that does not reseed after
  `initialize` therefore produces different pebble shuffles than v1.
- `pearlsim/ml_utilities.py`: the pebble branches of `read_det_file` (`read_pebbles`,
  `pebble_energy_grid`); `extract_xs_from_mdx`, `match_flux_to_xs_and_flatten`,
  `expand_pebble_features_by_energy`, `flatten_pebble_data`, `generate_pebble_burnup_model`,
  `interpolate_core_flux_3d`, `interpolate_core_power`, `extract_ml_samples_from_training_step`,
  `log_scale_data`, `create_forecast_dataset`, `create_training_split`; constants
  `PEBBLE_SURFACE_AREA`, `PEBBLE_KERNEL_VOLUME`.
- `PEARLSim.py`: the `ml_model_wrappers` import and `pebble_model.num_cores`.
- Review items made moot: #27, #32, #33.

Whole files (`pebble_model.py`, `depletion_utilities.py`, `pearlsim_chain.xml`,
`pebble_gFHR_template.serpent`, notebooks) and untracked artifacts: see the change log.

## 3. v1 one-step LSTM steering (superseded by the multi-step planner)

Keywords `steer`, `steer_constant_ascent`, `steer_updated` and helpers `get_reactivity_impacts*` in
`simulation.py`. *(Removal pending, Phase A.)*

## Change log
- Gamma-spectrum measurement code removed (section 1).
- Pebble-wise ML code paths removed from `core.py`, `simulation.py`, `ml_utilities.py`, `PEARLSim.py`
  (section 2).
