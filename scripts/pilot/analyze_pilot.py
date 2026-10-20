"""Split per-step noise into Monte Carlo and pebble-reshuffle parts from build_pilot_decks.py runs.

    python scripts/pilot/analyze_pilot.py PILOT_DIR [--prefix eq]

For each history level h, runs x_ij (shuffle i = 0..4, Serpent seed j = 0..1) of a quantity x give
  sigma_MC^2      = mean_i (x_i0 - x_i1)^2 / 2                 (same pebbles, different random numbers)
  sigma_shuffle^2 = var_i(mean_j x_ij) - sigma_MC^2 / 2        (clipped at 0)
for reactivity (pcm), every power and flux mesh cell (relative), and zone-level summaries (radial and
axial power fractions, peaking factor). Also compares Serpent's reported k error with the observed sigma_MC,
and the equal-cost options of dataset_v2_design.md §2: sigma^2 = sigma_MC^2/m + sigma_sh^2 (m x histories,
one shuffle) vs (sigma_MC^2 + sigma_sh^2)/n (n shuffles averaged). Writes PILOT_DIR/pilot_summary.{json,md}.
"""

import argparse
import glob
import json
import os
import re
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from pearlsim.group_definitions import ENERGY_GRID_3, HEIGHT_GRID_20, RADIUS_GRID_8  # noqa: E402
from pearlsim.ml_utilities import read_det_file  # noqa: E402

NAME = re.compile(r"_sh(\d+)_(h\d+)_ms(\d+)$")


def ana_keff(path):
    with open(path) as f:
        for line in f:
            if line.startswith("ANA_KEFF"):
                values = line.split("=", 1)[1].split("[", 1)[1].split()
                return float(values[0]), float(values[1])
    raise ValueError(path)


def load(pilot_dir, prefix):
    runs = {}
    for res in sorted(glob.glob(os.path.join(pilot_dir, f"{prefix}_sh*_h*_ms*.serpent_res.m"))):
        stem = res[: -len("_res.m")]
        i, h, j = NAME.search(os.path.basename(stem)[: -len(".serpent")]).groups()
        k, k_err = ana_keff(res)
        meshes = read_det_file(f"{stem}_det0.m", meshes_to_read=["subzone_3group_flux", "subzone_power"],
                               mesh_grids_rze=[(RADIUS_GRID_8, HEIGHT_GRID_20, ENERGY_GRID_3),
                                               (RADIUS_GRID_8, HEIGHT_GRID_20, [0.0])])
        power = np.asarray(meshes["subzone_power"]["data"], dtype=float)[..., 0]  # (radial, axial) W/cm3
        flux = np.asarray(meshes["subzone_3group_flux"]["data"], dtype=float)
        runs.setdefault(h, {})[(int(i), int(j))] = {"rho": 1e5 * (1 - 1 / k), "k_err": k_err * 1e5 / k,
                                                     "power": power, "flux": flux}
    return runs


def split(values):
    """values: array (5 shuffles, 2 seeds, ...) -> (sigma_MC, sigma_shuffle), elementwise."""
    mc2 = np.mean((values[:, 0] - values[:, 1]) ** 2 / 2, axis=0)
    sh2 = np.var(values.mean(axis=1), axis=0, ddof=1) - mc2 / 2
    return np.sqrt(mc2), np.sqrt(np.clip(sh2, 0, None))


def summaries(power):
    """Zone-level quantities from a power mesh (radial x axial cells)."""
    total = power.sum()
    return {"radial_fraction": power.sum(axis=1) / total, "axial_fraction": power.sum(axis=0) / total,
            "peaking": np.array([power.max() / power.mean()])}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("pilot_dir")
    parser.add_argument("--prefix", default="eq")
    args = parser.parse_args(argv)
    runs = load(args.pilot_dir, args.prefix)
    out, lines = {}, ["| histories | runs | sigma_MC rho (pcm) | sigma_shuffle rho (pcm) | Serpent-reported k err (pcm) |"
                      " power cell sigma_MC / sigma_sh (median rel.) | flux cell sigma_MC / sigma_sh (median rel.) |"
                      " peaking sigma_MC / sigma_sh |", "|---|---|---|---|---|---|---|---|"]
    for h, cases in sorted(runs.items()):
        keys = [(i, j) for i in range(5) for j in range(2)]
        if not all(key in cases for key in keys):
            print(f"{h}: {len(cases)} of 10 runs, skipped")
            continue

        def grid(field, cases=cases):
            return np.array([[cases[(i, j)][field] for j in range(2)] for i in range(5)])

        rho_mc, rho_sh = split(grid("rho"))
        power, flux = grid("power"), grid("flux")
        p_mean, f_mean = power.mean(axis=(0, 1)), flux.mean(axis=(0, 1))
        p_mc, p_sh = split(power / np.where(p_mean > 0, p_mean, np.nan))
        f_mc, f_sh = split(flux / np.where(f_mean > 0, f_mean, np.nan))
        zone = {name: np.array([[summaries(power[i, j])[name] for j in range(2)] for i in range(5)])
                for name in ("radial_fraction", "axial_fraction", "peaking")}
        zone_split = {name: [a.tolist() for a in split(values)] for name, values in zone.items()}
        reported = float(np.mean(grid("k_err")))
        res = {"runs": 10, "rho_mean_pcm": float(grid("rho").mean()), "sigma_mc_pcm": float(rho_mc),
               "sigma_shuffle_pcm": float(rho_sh), "serpent_reported_pcm": reported,
               "power_cell_rel": {"mc_median": float(np.nanmedian(p_mc)), "sh_median": float(np.nanmedian(p_sh)),
                                  "mc_p90": float(np.nanpercentile(p_mc, 90)), "sh_p90": float(np.nanpercentile(p_sh, 90))},
               "flux_cell_rel": {"mc_median": float(np.nanmedian(f_mc)), "sh_median": float(np.nanmedian(f_sh))},
               "zone": zone_split}
        mc2, sh2 = rho_mc ** 2, rho_sh ** 2
        res["equal_cost_pcm"] = {f"{m}x histories": float(np.sqrt(mc2 / m + sh2)) for m in (2, 4)}
        res["equal_cost_pcm"].update({f"{n} shuffles": float(np.sqrt((mc2 + sh2) / n)) for n in (2, 4)})
        out[h] = res
        lines.append(f"| {h} | 10 | {rho_mc:.0f} | {rho_sh:.0f} | {reported:.0f} | "
                     f"{res['power_cell_rel']['mc_median']:.4f} / {res['power_cell_rel']['sh_median']:.4f} | "
                     f"{res['flux_cell_rel']['mc_median']:.4f} / {res['flux_cell_rel']['sh_median']:.4f} | "
                     f"{zone_split['peaking'][0][0]:.4f} / {zone_split['peaking'][1][0]:.4f} |")
    lines += ["", "Equal cost (per step, from each history level's sigmas):", ""]
    for h, res in out.items():
        lines.append(f"- {h}: " + ", ".join(f"{k} {v:.0f} pcm" for k, v in res["equal_cost_pcm"].items()))
    with open(os.path.join(args.pilot_dir, "pilot_summary.json"), "w") as f:
        json.dump(out, f, indent=1)
    with open(os.path.join(args.pilot_dir, "pilot_summary.md"), "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
