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

Whole files:

| Removed | Tracked? | Recover from |
|---|---|---|
| `pearlsim/pebble_model.py`, `pearlsim/depletion_utilities.py`, `pearlsim/isotope_rename_key.py` | yes | `v1-as-run` |
| `pearlsim_chain.xml` (OpenMC depletion chain), `pebble_gFHR_template.serpent` | yes | `v1-as-run` |
| `create_current_model.ipynb`, `generate_current_dataset.ipynb` | yes | `v1-as-run` |
| `pearlsim/ml_model_wrappers.py`, `gen_xs_openmc.py`, `train_burnup_models{,_2}.{py,sub}` | no | branch `v1`, or the cleanup archive |
| `create_current_model-Copy1.ipynb`, `create_current_model_object.ipynb`, `create_burnup_model{,_backup,_object}.ipynb`, `create_xs_model.ipynb`, `generate_microxs_models.ipynb`, `generate_kernel_burnup_data{,_backup}.ipynb` | no | cleanup archive |
| `ml_models/{current,flux}_rfr_{feature,target}_{mean,std}.csv` | no | branch `v1`, or the cleanup archive |
| `depletion_results.h5`, `depletion_results_2.h5`, `u235endf7.ace`, `pebble_current_matrix/` | no | cleanup archive |
| `burnup_models/` (4.3 GB), `pebble_xs_models/` (3.9 GB), `detector_data/` (0.6 GB), `ml_models/{current,flux}_{model,rfr}.pkl` (12.8 GB); moved without hashing | no | cleanup archive |

## 3. v1 one-step LSTM steering (superseded by the multi-step planner)

One-step LSTM-guided run-in steering. It produced the `gFHR_automated_*`, `gFHR_ascension_*` and
`gFHR_runin_*` v1 sequences. Known defects at removal: review #5, #6, #14–#16.

| Removed | Recover from |
|---|---|
| keywords `steer`, `steer_constant_ascent`, `steer_updated` and helpers `get_reactivity_impacts`, `get_reactivity_impacts_updated`, `get_reactivity_impacts_delayed` in `pearlsim/simulation.py` | `v1-as-run` |
| steering instruction files `gFHR_power_ascension*.csv`, `run_in_*.json` and the decks that use them | branch `v1` |

## Change log
- Gamma-spectrum measurement code removed (section 1).
- Pebble-wise ML code paths removed from `core.py`, `simulation.py`, `ml_utilities.py`, `PEARLSim.py`
  (section 2).
- Pebble-wise ML files deleted and untracked artifacts archived (section 2).
- v1 steering removed (section 3).
