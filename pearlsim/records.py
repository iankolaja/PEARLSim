"""Per-step records: everything the ML and the bookkeeping need about one depletion step.

A record is built from the files a step leaves in its run directory, so the same
code serves a running simulation (which writes ``<sim>_record_<i>.json`` after each
Serpent step) and v1 runs gathered after the fact:

* ``<sim>_operating_<i>.json``: controls, temperatures, seeds;
* ``zone_map<i>.json`` (and ``<sim>_held_inventory<i>.json`` from v2): the exact
  pebble inventory transported in this step;
* ``<sim>_discharge_inventory<i>.json``, ``<sim>_discard_inventory<i>.json``;
* per replica, ``<deck>_res.m`` and ``<deck>_det<k>.m`` (k = last burnup point);
* ``<sim>_replica_spread_<i>.json`` for n > 1.

Mesh units: Serpent writes mesh detector results integrated over each cell
(docs/serpent/detectors.md); the subzone power mesh (``dr -8 void``, fission
energy production normalised by ``set power``) is in W. ``read_det_file`` divides
every value by the cell's geometric volume, so the meshes here (and the v1
``fluxR*Z*E*``/``powerR*Z*`` columns) are cell averages: power in W/cm3, flux in
n/cm2/s. ``power.mesh_power_W`` multiplies the volumes back. The subzone mesh
(8 radial x 20 axial) has equal-area rings and equal heights, so its cells have
equal volume (87 500.6 cm3); the 4 x 10 zone powers are 2 x 2 block sums (the zone
grid's bounds are every second subzone bound; group_definitions.RADIUS_GRID_4/HEIGHT_GRID_10).
"""

import json
import os
import platform
import re
import shlex
import shutil
import subprocess

import numpy as np

from pearlsim import replicas

SCHEMA_VERSION = 2  # 2: adds "discharge_smooth"

# Smoothed discharge burnup: same range and edges as v1's burnup_bin features (BURNUP_BINS_9)
SMOOTH_BURNUP_EDGES = np.linspace(0.0, 22.5, 10)  # FIMA, %
SMOOTH_BURNUP_QUANTILES = (0.1, 0.5, 0.9)
DEFAULT_FIMA_REL_STD = 0.02576  # Simulation default for fima_discharge_rel_std

# _res.m scalars copied into the record's "serpent" block (first occurrence in the file)
RES_METADATA = {"VERSION": str, "COMPILE_DATE": str, "XS_DATA_FILE_PATH": str, "DECAY_DATA_FILE_PATH": str,
                "NFY_DATA_FILE_PATH": str, "POP": int, "CYCLES": int, "SKIP": int, "SEED": int,
                "OMP_THREADS": int, "MPI_TASKS": int}
CONTROLS = ("power", "burnup_step", "graphite_insertion_fraction", "control_rod_position",
            "fima_discharge_threshold", "fima_discharge_rel_std")


def smoothed_discharge_burnup(discharge, rel_std, edges=SMOOTH_BURNUP_EDGES, quantiles=SMOOTH_BURNUP_QUANTILES):
    """Discharged-fuel burnup distribution with each group spread over a Gaussian.

    v1's ``burnup_bin<i>_count`` puts all pebbles of a discharged group (one average FIMA
    per zone x burnup group) into the single bin that average falls in, so a count jumps
    by a whole group when the average crosses an edge. Here each group of ``count``
    pebbles at FIMA F is spread as N(F, (rel_std F)^2): the within-group spread the
    discard decision already assumes (zone.Zone.remove_threshold). Pebble counts are
    conserved (mass below the first edge goes to the first bin, above the last edge to
    the last bin), and rel_std = 0 gives v1's hard bins. Measurement noise is not added
    here; the ML noise model (pearlsim.ml.features.NoiseModel) applies it later.

    Returns ``discharge_fuel_pebbles``, ``burnup_smooth_bin<i>_frac`` (summing to 1) and
    ``burnup_smooth_p<q>`` (FIMA %, quantiles of the mixture). With no fuel discharged the
    fractions and quantiles are 0, as v1 fills ``avg_FIMA``.
    """
    from scipy.stats import norm

    groups = [(float(v["count"]), float(v["FIMA"])) for key, v in discharge.items()
              if "graph" not in key and float(v["count"]) > 0]
    n_bins = len(edges) - 1
    total = sum(count for count, _ in groups)
    out = {"discharge_fuel_pebbles": total}
    if total == 0:
        out.update({f"burnup_smooth_bin{i + 1}_frac": 0.0 for i in range(n_bins)})
        out.update({f"burnup_smooth_p{round(q * 100)}": 0.0 for q in quantiles})
        return out

    counts = np.array([c for c, _ in groups])
    fima = np.array([f for _, f in groups])
    sigma = rel_std * fima

    def mixture_cdf(x):
        """Fraction of discharged fuel pebbles with burnup <= x (x: 1-D array)."""
        x = np.asarray(x, dtype=float)[:, None]
        with np.errstate(divide="ignore", invalid="ignore"):
            smooth = norm.cdf((x - fima) / np.where(sigma > 0, sigma, 1.0))
        cdf = np.where(sigma > 0, smooth, (x >= fima).astype(float))
        return cdf @ counts / total

    inner = mixture_cdf(edges[1:-1])
    bounds = np.concatenate([[0.0], inner, [1.0]])
    for i, frac in enumerate(np.diff(bounds)):
        out[f"burnup_smooth_bin{i + 1}_frac"] = float(frac)
    grid = np.linspace(0.0, max(edges[-1], float((fima + 6 * sigma).max())), 6001)
    cdf = mixture_cdf(grid)
    for q in quantiles:
        out[f"burnup_smooth_p{round(q * 100)}"] = float(grid[min(np.searchsorted(cdf, q), len(grid) - 1)])
    return out


def record_file(run_name, step):
    return f"{run_name}_record_{step}.json"


def reactivity_pcm(k):
    return (k - 1.0) / k * 1e5


def read_res_metadata(file_name):
    """Serpent version, libraries, population, seed and timings from a _res.m file."""
    found, running_time = {}, None
    pattern = re.compile(r"^(\w+)\s+\(idx, [^)]*\)\s*=\s*(.*?)\s*;\s*$")
    with open(file_name) as f:
        for line in f:
            match = pattern.match(line)
            if not match:
                continue
            key, value = match.groups()
            if key in RES_METADATA and key not in found:
                found[key] = value.strip("'") if RES_METADATA[key] is str else RES_METADATA[key](value)
            elif key == "RUNNING_TIME":
                running_time = float(value)  # cumulative: the last one is the whole run
    found["RUNNING_TIME_MIN"] = running_time
    return found


def detect_replicas(directory, run_name, step):
    """Number of replicas whose _res.m exists for a step (1 for v1 single-run names)."""
    n = 0
    while os.path.isfile(os.path.join(directory, f"{replicas.stem(run_name, step, n)}_res.m")):
        n += 1
    return n or 1


def _replica_outputs(stem):
    from pearlsim.group_definitions import ENERGY_GRID_3, HEIGHT_GRID_20, RADIUS_GRID_8
    from pearlsim.ml_utilities import read_det_file
    from pearlsim.results_processing import read_res_file

    keff, unc = read_res_file(f"{stem}_res.m", "ANA_KEFF", burnup_step="all")
    det_file = f"{stem}_det{len(keff) - 1}.m"
    if not os.path.exists(det_file):
        det_file = f"{stem}_det{len(keff) - 2}.m"
    meshes = read_det_file(det_file, meshes_to_read=["subzone_3group_flux", "subzone_power"],
                           mesh_grids_rze=[(RADIUS_GRID_8, HEIGHT_GRID_20, ENERGY_GRID_3),
                                           (RADIUS_GRID_8, HEIGHT_GRID_20, [0.0])])
    return {"keff": keff, "keff_unc": unc, "meshes": meshes, "det_file": os.path.basename(det_file),
            "serpent": read_res_metadata(f"{stem}_res.m")}


def _k_block(runs, index):
    values = [run["keff"][index] for run in runs]
    mean, unc, std = replicas.mean_over_replicas(values, [run["keff_unc"][index] for run in runs])
    return {"mean": float(mean), "unc": float(unc), "rho_pcm": float(reactivity_pcm(mean)),
            "replicas": values, "replica_std": float(std) if len(runs) > 1 else None}


def _mesh_block(runs, name):
    mean, unc, std = replicas.mean_over_replicas([run["meshes"][name]["data"] for run in runs],
                                                 [run["meshes"][name]["unc"] for run in runs])
    return {"data": mean, "unc": unc, "replica_std": std if len(runs) > 1 else None}


def cell_volumes(radius_grid, height_grid):
    """Volumes (cm3) of the cells of a cylindrical r-z mesh, shape (len(radius_grid)-1, len(height_grid)-1)."""
    radius_grid, height_grid = np.asarray(radius_grid, dtype=float), np.asarray(height_grid, dtype=float)
    return np.outer(np.pi * np.diff(radius_grid ** 2), np.diff(height_grid))


def power_summary(power_density, volumes=None):
    """Mesh power, zone power fractions and peaking factors from the (8, 20) subzone power density (W/cm3).

    Peaking factors are ratios of volume-averaged power densities (max over mean).
    """
    from pearlsim.group_definitions import HEIGHT_GRID_20, RADIUS_GRID_8

    density = np.asarray(power_density, dtype=float)
    volumes = cell_volumes(RADIUS_GRID_8, HEIGHT_GRID_20) if volumes is None else np.asarray(volumes, dtype=float)
    power = density * volumes  # W per cell
    total = power.sum()
    if not total:
        return {"mesh_power_W": 0.0, "zone_power_fraction": None, "peaking_mesh": None,
                "peaking_radial": None, "peaking_axial": None, "peaking_zone": None}

    def peaking(cell_power, cell_volume):
        averaged = cell_power / cell_volume
        return float(averaged.max() / (cell_power.sum() / cell_volume.sum()))

    zone_power = power.reshape(4, 2, 10, 2).sum(axis=(1, 3))
    zone_volume = volumes.reshape(4, 2, 10, 2).sum(axis=(1, 3))
    return {"mesh_power_W": float(total),
            "zone_power_fraction": (zone_power / total).tolist(),
            "peaking_mesh": peaking(power, volumes),
            "peaking_radial": peaking(power.sum(axis=1), volumes.sum(axis=1)),
            "peaking_axial": peaking(power.sum(axis=0), volumes.sum(axis=0)),
            "peaking_zone": peaking(zone_power, zone_volume)}


def inventory_summary(zone_map, held=None):
    """Exact pebble counts by type, overall and per radial channel, from a zone map."""
    channels = {}
    for zone_id, inventory in zone_map.items():
        channel = int(zone_id.split("R")[1].split("Z")[0])
        fuel = sum(n for key, n in inventory.items() if "fuel" in key)
        graphite = sum(n for key, n in inventory.items() if "fuel" not in key)
        totals = channels.setdefault(channel, [0, 0])
        totals[0] += fuel
        totals[1] += graphite
    fuel = sum(c[0] for c in channels.values())
    graphite = sum(c[1] for c in channels.values())
    return {"fuel_pebbles": fuel, "graphite_pebbles": graphite,
            "fuel_fraction": fuel / (fuel + graphite) if fuel + graphite else None,
            "channel_fuel_fraction": [channels[c][0] / sum(channels[c]) if sum(channels[c]) else None
                                      for c in sorted(channels)],
            "held_pebbles": None if held is None else sum(held.values())}


def _load_json(path):
    if not os.path.isfile(path):
        return None
    with open(path) as f:
        return json.load(f)


def _number(value):
    """v1 stored power as text (review #20)."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return value


def code_version():
    """pearlsim package version and the git commit of its source tree (None outside a checkout)."""
    from importlib import metadata

    try:
        version = metadata.version("pearlsim")
    except metadata.PackageNotFoundError:
        version = None
    package_dir = os.path.dirname(os.path.abspath(__file__))
    commit = dirty = None
    try:
        result = subprocess.run(["git", "-C", package_dir, "rev-parse", "HEAD"], capture_output=True, text=True,
                                timeout=60)
        if result.returncode == 0:
            commit = result.stdout.strip()
            dirty = subprocess.run(["git", "-C", package_dir, "diff", "--quiet", "HEAD", "--", "."],
                                   timeout=60).returncode != 0
    except (OSError, subprocess.TimeoutExpired):
        pass
    return {"pearlsim_version": version, "git_commit": commit, "git_dirty": dirty}


def run_provenance(serpent_command, serpent_settings, dataset_version=None):
    """What produced a run: dataset version, code version, Serpent executable and settings, host and job."""
    executable = shutil.which(shlex.split(serpent_command)[0])
    settings = {key: value for key, value in serpent_settings.items() if key != "seed"}
    return {"dataset_version": dataset_version, **code_version(),
            "serpent_command": serpent_command,
            "serpent_executable": os.path.realpath(executable) if executable else None,
            "serpent_settings": settings,
            "host": platform.node(), "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            "python": platform.python_version(), "numpy": np.__version__}


def build_record(directory, run_name, step, num_replicas=None, days=None, provenance=None):
    """The per-step record (a JSON-serialisable dict) of one step in ``directory``."""
    from pearlsim.results_processing import extract_lstm_input_features

    path = lambda name: os.path.join(directory, name)  # noqa: E731
    operating = _load_json(path(f"{run_name}_operating_{step}.json"))
    if operating is None:
        raise FileNotFoundError(path(f"{run_name}_operating_{step}.json"))
    if num_replicas is None:
        num_replicas = detect_replicas(directory, run_name, step)
    runs = [_replica_outputs(path(replicas.stem(run_name, step, r))) for r in replicas.replica_ids(num_replicas)]

    controls = {key: _number(operating.get(key)) for key in CONTROLS}
    zone_map = _load_json(path(f"zone_map{step}.json"))
    inventory = None
    if zone_map is not None:
        inventory = inventory_summary(zone_map, _load_json(path(f"{run_name}_held_inventory{step}.json")))
        if inventory["fuel_pebbles"] and isinstance(controls["power"], float):
            inventory["power_per_fuel_pebble"] = controls["power"] / inventory["fuel_pebbles"]
    discharge = discharge_smooth = None
    if os.path.isfile(path(f"{run_name}_discharge_inventory{step}.json")):
        _, features = extract_lstm_input_features(directory, step, run_name=run_name, merge=False)
        discharge = {key: float(value) for key, value in features.items()}
        rel_std = controls["fima_discharge_rel_std"]
        discharge_smooth = smoothed_discharge_burnup(
            _load_json(path(f"{run_name}_discharge_inventory{step}.json")),
            rel_std if isinstance(rel_std, float) else DEFAULT_FIMA_REL_STD)
    spread = _load_json(path(f"{run_name}_replica_spread_{step}.json"))
    spread_summary = None
    if spread:
        worst = max(spread, key=lambda name: spread[name]["max"])
        spread_summary = {"max_rel_std": spread[worst]["max"], "material": worst,
                          "nuclide": spread[worst]["max_nuclide"]}
        for nuclide in replicas.SPREAD_NUCLIDES:
            values = [s[nuclide] for s in spread.values() if nuclide in s]
            spread_summary[f"max_rel_std_{nuclide.replace('<lib>', '')}"] = max(values) if values else None

    flux, power = _mesh_block(runs, "subzone_3group_flux"), _mesh_block(runs, "subzone_power")
    power_block = power_summary(power["data"][:, :, 0])
    if isinstance(controls["power"], float) and controls["power"]:
        # fission energy scored in the mesh over the normalisation power (TOT_POWER); close to 1
        power_block["mesh_power_over_set_power"] = power_block["mesh_power_W"] / controls["power"]
    as_list = lambda block: {k: None if v is None else np.asarray(v).tolist() for k, v in block.items()}  # noqa: E731
    return {
        "schema_version": SCHEMA_VERSION,
        "run_name": run_name,
        "step": step,
        "days": days,
        "controls": controls,
        "nuclear_data": operating.get("nuclear_data", "endf7_v1"),
        "thermal_coupling": operating.get("thermal_coupling", False),
        "temperatures": operating.get("temperatures"),
        "seeds": {"base_seed": operating.get("base_seed"), "serpent_seed": operating.get("serpent_seed"),
                  "serpent_seed_in_output": [run["serpent"].get("SEED") for run in runs]},
        "replicas": num_replicas,
        "keff": {"bos": _k_block(runs, 0), "eos": _k_block(runs, -1)},
        "power": power_block,
        "meshes": {"subzone_3group_flux": as_list(flux), "subzone_power": as_list(power),
                   "det_file": runs[0]["det_file"]},
        "inventory": inventory,
        "discharge": discharge,
        "discharge_smooth": discharge_smooth,
        "replica_spread": spread_summary,
        "serpent": [run["serpent"] for run in runs],
        "provenance": provenance or {},
    }


def write_record(directory, run_name, step, **kwargs):
    record = build_record(directory, run_name, step, **kwargs)
    with open(os.path.join(directory, record_file(run_name, step)), "w") as f:
        json.dump(record, f)
    return record


def flatten(record, meshes=True):
    """One table row from a record. Mesh columns keep the v1 names (fluxR<r>Z<z>E<e>, powerR<r>Z<z>)."""
    row = {"run_name": record["run_name"], "step": record["step"], "days": record["days"],
           "replicas": record["replicas"], "nuclear_data": record["nuclear_data"],
           "thermal_coupling": record["thermal_coupling"]}
    row.update(record["controls"])
    for key, value in (record["temperatures"] or {}).items():
        row[f"T_{key}"] = value
    row["base_seed"] = record["seeds"]["base_seed"]
    for point, block in record["keff"].items():
        row[f"keff_{point}"] = block["mean"]
        row[f"keff_{point}_unc"] = block["unc"]
        row[f"rho_{point}_pcm"] = block["rho_pcm"]
        row[f"keff_{point}_rep_std"] = block["replica_std"]
        for r, value in enumerate(block["replicas"]):
            row[f"keff_{point}_rep{r}"] = value
    row["final_analog_keff"] = record["keff"]["eos"]["mean"]  # v1 name of the target
    power = record["power"]
    for key in ("mesh_power_W", "mesh_power_over_set_power", "peaking_mesh", "peaking_radial", "peaking_axial",
                "peaking_zone"):
        row[key] = power.get(key)
    if power["zone_power_fraction"] is not None:
        for r, column in enumerate(power["zone_power_fraction"]):
            for z, value in enumerate(column):
                row[f"zone_power_fracR{r + 1}Z{z + 1}"] = value
    inventory = record["inventory"] or {}
    for key in ("fuel_pebbles", "graphite_pebbles", "fuel_fraction", "held_pebbles", "power_per_fuel_pebble"):
        row[key] = inventory.get(key)
    for c, value in enumerate(inventory.get("channel_fuel_fraction") or []):
        row[f"R{c + 1}_fuel_fraction"] = value
    row.update(record["discharge"] or {})
    row.update(record.get("discharge_smooth") or {})  # schema >= 2
    for key, value in (record["replica_spread"] or {}).items():
        if key.startswith("max_rel_std"):
            row[f"replica_{key}"] = value
    serpent = record["serpent"][0]
    row["serpent_version"] = serpent.get("VERSION")
    row["serpent_pop"] = serpent.get("POP")
    row["serpent_cycles"] = serpent.get("CYCLES")
    row["serpent_skip"] = serpent.get("SKIP")
    row["running_time_min"] = sum(s.get("RUNNING_TIME_MIN") or 0.0 for s in record["serpent"])
    if meshes:
        flux = np.asarray(record["meshes"]["subzone_3group_flux"]["data"])
        for r in range(flux.shape[0]):
            for z in range(flux.shape[1]):
                for e in range(flux.shape[2]):
                    row[f"fluxR{r + 1}Z{z + 1}E{e + 1}"] = flux[r, z, e]
        power_mesh = np.asarray(record["meshes"]["subzone_power"]["data"])
        for r in range(power_mesh.shape[0]):
            for z in range(power_mesh.shape[1]):
                row[f"powerR{r + 1}Z{z + 1}"] = power_mesh[r, z, 0]
    return row
