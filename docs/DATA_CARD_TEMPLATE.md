# {{Dataset name}} data card

Fill every `{{...}}` field. Values marked *(from records)* come from the gathered table
and its `.meta.json`; don't type them by hand.

## Summary
- **What:** time series of a zone-based depletion model of the Kairos gFHR pebble-bed reactor:
  operating controls, core inventory and discharged-fuel features per step, with Serpent 2 k_eff and
  flux/power meshes as targets.
- **Dataset version:** {{e.g. v2.0}} (`dataset_version` in every record).
- **Size:** {{sequences}} sequences, {{steps}} steps, {{columns}} columns *(from records)*.
- **Intended use:** training and evaluating core-state forecasters and planners. Not for safety
  analysis or licensing.
- **Contact:** {{name, email}}.

## Model
- **Code:** PEARLSim {{version / git commit}} *(from records)*; Serpent {{VERSION, COMPILE_DATE}}
  *(from records)*, executable {{path}}.
- **Geometry:** gFHR template {{file and commit}}; 4 radial × 10 axial zones; mesh detectors of 8 × 20
  equal-volume cells (3 energy groups for flux).
- **Nuclear data:** {{profile, e.g. endf8_lib80x}}; libraries {{acelib / declib / nfylib}}; thermal
  scattering {{tables}}. See `docs/nuclear_data.md` and its change log.
- **Temperatures:** {{uncoupled fixed values, or the power-coupled model of `pearlsim/thermal.py`
  with its calibration source}}.
- **Statistics:** {{n replicas}} reshuffled replicas per step, each `set pop {{N cycles skip}}`;
  per-step Monte Carlo and shuffle noise {{from the pilot study}}.

## Generation
- **Sequences:** {{families, shares, control ranges, hold lengths}}.
- **Initial states:** {{…}}.
- **Held-out test set:** {{chosen before generation; list sequences}}.
- **Seeds:** every step's Python and Serpent seeds are recorded; any step can be regenerated.

## Contents
- **One row per depletion step.** Column groups:
  - identity: `sequence`, `step`, `days`;
  - controls: `power`, `burnup_step`, `graphite_insertion_fraction`, `control_rod_position`,
    `fima_discharge_threshold`, `fima_discharge_rel_std`;
  - temperatures: `T_*`;
  - inventory: `fuel_pebbles`, `graphite_pebbles`, `fuel_fraction`, `R*_fuel_fraction`,
    `held_pebbles`, `power_per_fuel_pebble`;
  - discharge features: `R*_avg_FIMA_last_pass`, `burnup_bin*_count`, `avg_FIMA`, `num_discarded`;
  - targets: `keff_bos`/`keff_eos` (mean, `_unc`, `_rep*`, `_rep_std`), `rho_*_pcm`,
    `fluxR*Z*E*`, `powerR*Z*`, `zone_power_fracR*Z*`, `peaking_*`;
  - provenance: `serpent_version`, `serpent_pop`, `running_time_min`, ….
- **Full records** (`<sim>_record_<i>.json`) also hold mesh uncertainties and replica spreads.
- **Units:** power W; time days; temperatures K; k_eff dimensionless; reactivity pcm.
  Mesh columns are cell averages: `powerR*Z*` in W/cm³ and `fluxR*Z*E*` in n/cm²/s (Serpent's
  per-cell integrals divided by the cell volume, 87,500.6 cm³ for every subzone cell).
  `mesh_power_W` is the fission power scored in the mesh.

## Known limitations
- Zone-averaged depletion: pebbles in a zone share one composition; replicas average the
  shuffle-dependent flux.
- {{Thermal model: quasi-static, uniform, fixed coolant flow; no xenon/temperature transients within
  a step.}}
- {{Rods and power are independent inputs; the critical rod position is an estimate (M3).}}
- {{Differences from v1 data: #7, #36, #37 fixed; per-step seeds; replicas; see
  `docs/dataset_v2_design.md` §1d.}}

## Citation and license
- **Cite:** {{paper}}, {{dataset DOI}}.
- **License:** data {{e.g. CC BY 4.0}}; code MIT.
- **Acknowledgements:** {{computing allocation (Savio, co_nuclear), funding}}.
