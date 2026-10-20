"""Compare a PEARLSim pass-mode equilibrium run with the gFHR benchmark's KPACS equilibrium.

    python scripts/kpacs_check/compare_equilibrium.py RUN_DIR ECC_JSON OUT_DIR [--window 80] [--every 10]
        [--v1 V1_RUN_DIR]

RUN_DIR: a PEARLSim run directory (records, zone maps, core_materials, discard inventories), e.g.
runs/v2_kpacs_eq/gFHR_v2_kpacs_eq. The equilibrium window is the last --window Serpent steps
(80 = one 522-EFPD residence); compositions are sampled every --every steps in it.

Benchmark values (Satvat et al. 2021, accepted manuscript, docs/literature):
  - KPACS equilibrium k 1.01230 +/- 0.00068 (§4.1); Kairos's Serpent with the 40 zone-averaged
    compositions 1.01472 +/- 0.00019 (Table 10), with all 320 1.01443 +/- 0.00023 (footnote 1)
  - Table 4: average-per-pass mass (g) per pebble, top axial zone
  - average discharge burnup 17.6 %FIMA at 60% packing (Table 6)
ECC_JSON: Kairos's 40-zone equilibrium kernel compositions ({"Z<a>R<r>": {"ZZAAA": atom/b/cm}}), Z1 = bottom.

Writes OUT_DIR/summary.json, summary.md and plots.
"""

import argparse
import glob
import gzip
import json
import math
import os
import re

import numpy as np

KPACS_K, KPACS_K_UNC = 1.01230, 0.00068
KAIROS_SERPENT_40 = 1.01472
DISCHARGE_FIMA_BENCHMARK = 17.6
# Table 4 (g per pebble, passes 1-8, top axial zone)
TABLE4 = {
    "92235": [5.44e-01, 4.48e-01, 3.69e-01, 3.04e-01, 2.49e-01, 2.05e-01, 1.69e-01, 1.39e-01],
    "92236": [2.25e-02, 4.04e-02, 5.45e-02, 6.54e-02, 7.37e-02, 7.99e-02, 8.46e-02, 8.77e-02],
    "92238": [2.69e+00, 2.65e+00, 2.62e+00, 2.58e+00, 2.55e+00, 2.52e+00, 2.48e+00, 2.45e+00],
    "94238": [1.17e-05, 1.00e-04, 3.14e-04, 6.73e-04, 1.21e-03, 1.87e-03, 2.67e-03, 3.57e-03],
    "94239": [2.31e-02, 3.33e-02, 3.74e-02, 3.92e-02, 3.97e-02, 3.98e-02, 3.96e-02, 3.93e-02],
    "94240": [3.03e-03, 7.70e-03, 1.12e-02, 1.35e-02, 1.50e-02, 1.57e-02, 1.60e-02, 1.63e-02],
    "94241": [8.24e-04, 3.91e-03, 7.83e-03, 1.14e-02, 1.41e-02, 1.59e-02, 1.73e-02, 1.79e-02],
    "94242": [4.03e-05, 4.26e-04, 1.39e-03, 2.89e-03, 4.77e-03, 6.81e-03, 8.93e-03, 1.10e-02],
    "95241": [2.25e-06, 1.92e-05, 5.48e-05, 1.01e-04, 1.49e-04, 1.92e-04, 2.27e-04, 2.52e-04],
    "95342": [1.20e-08, 1.62e-07, 5.83e-07, 1.21e-06, 1.91e-06, 2.57e-06, 3.14e-06, 3.58e-06],  # Am-242m
    "55137": [3.62e-03, 7.00e-03, 1.01e-02, 1.29e-02, 1.54e-02, 1.77e-02, 1.97e-02, 2.16e-02],
    "54135": [4.32e-06, 4.01e-06, 3.67e-06, 3.35e-06, 3.05e-06, 2.78e-06, 2.58e-06, 2.42e-06],
    "53131": [2.10e-04, 2.00e-04, 1.85e-04, 1.71e-04, 1.56e-04, 1.44e-04, 1.33e-04, 1.24e-04],
    "38090": [2.12e-03, 3.94e-03, 5.50e-03, 6.82e-03, 7.96e-03, 8.92e-03, 9.74e-03, 1.04e-02],
}
NAMES = {"92235": "U-235", "92236": "U-236", "92238": "U-238", "94238": "Pu-238", "94239": "Pu-239",
         "94240": "Pu-240", "94241": "Pu-241", "94242": "Pu-242", "95241": "Am-241", "95342": "Am-242m",
         "55137": "Cs-137", "54135": "Xe-135", "53131": "I-131", "38090": "Sr-90", "62149": "Sm-149"}
ECC_NUCLIDES = ["92235", "92236", "92238", "94239", "94240", "94241", "94242", "55137", "54135", "62149"]
# kernel volume per pebble: 9,022 TRISOs, kernel radius 0.02125 cm (gFHR_triso.dat, Material.write_triso)
KERNEL_CM3_PER_PEBBLE = 9022 * 4.0 / 3.0 * math.pi * 0.02125 ** 3
AVOGADRO = 6.02214076e23
MATERIAL = re.compile(r"^fuel20_R(\d+)Z(\d+)G(\d+)$")


def za_mass(za):
    """Mass number as g/mol (within 0.1% for these nuclides); Serpent isomers add 100 to A (95342 = Am-242m)."""
    a = int(za[-3:])
    return float(a - 100 if a > 300 else a)


def grams(conc_per_b_cm, za):
    return conc_per_b_cm * 1e24 * KERNEL_CM3_PER_PEBBLE * za_mass(za) / AVOGADRO


def load_records(run):
    records = {}
    for path in glob.glob(os.path.join(run, "*_record_*.json")):
        with open(path) as f:
            record = json.load(f)
        records[record["step"]] = record
    return dict(sorted(records.items()))


def v1_keff(run):
    series = {}
    for path in glob.glob(os.path.join(run, "*.serpent_res.m")):
        step = int(re.search(r"_(\d+)\.serpent_res\.m$", path).group(1))
        with open(path) as f:
            values = [float(line.split("=", 1)[1].split("[", 1)[1].split()[0]) for line in f if line.startswith("ANA_KEFF")]
        if len(values) >= 2:
            series[step] = values[:2]
    return dict(sorted(series.items()))


def zone_compositions(run, step):
    """{(r, a, g): (count, {za: atom/b/cm})} for fuel at the start of `step`."""
    with open(os.path.join(run, f"zone_map{step}.json")) as f:
        zone_map = json.load(f)
    with gzip.open(os.path.join(run, f"core_materials_{step}.json.gz")) as f:
        materials = json.load(f)
    out = {}
    for zone in zone_map.values():
        for name, count in zone.items():
            match = MATERIAL.match(name)
            if match and count:
                conc = {k.split("<")[0]: v for k, v in materials[name].items()}
                out[tuple(int(x) for x in match.groups())] = (count, conc)
    return out


def average(groups, keys, nuclides):
    total = sum(groups[k][0] for k in keys)
    return {za: sum(groups[k][0] * groups[k][1].get(za, 0.0) for k in keys) / total for za in nuclides}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("run")
    parser.add_argument("ecc_json")
    parser.add_argument("out")
    parser.add_argument("--window", type=int, default=80)
    parser.add_argument("--every", type=int, default=10)
    parser.add_argument("--v1")
    args = parser.parse_args(argv)
    os.makedirs(args.out, exist_ok=True)
    with open(args.ecc_json) as f:
        ecc = json.load(f)

    records = load_records(args.run)
    steps = list(records)
    window = steps[-args.window:]
    bos = np.array([records[s]["keff"]["bos"]["mean"] for s in steps])
    eos = np.array([records[s]["keff"]["eos"]["mean"] for s in steps])
    mid = 0.5 * (bos[-len(window):] + eos[-len(window):])
    summary = {"run": os.path.abspath(args.run), "steps": [steps[0], steps[-1]], "window": [window[0], window[-1]],
               "complete_residences": len(steps) / 80}
    summary["keff"] = {"bos_mean": float(bos[-len(window):].mean()), "eos_mean": float(eos[-len(window):].mean()),
                       "mid_mean": float(mid.mean()), "mid_std": float(mid.std(ddof=1)) if len(mid) > 1 else None,
                       "kpacs": KPACS_K, "kpacs_unc": KPACS_K_UNC, "kairos_serpent_40": KAIROS_SERPENT_40}
    summary["keff"]["mid_minus_kpacs_pcm"] = 1e5 * (summary["keff"]["mid_mean"] - KPACS_K)

    # discharge burnup: count-weighted FIMA of the discarded (8th-pass) pebbles
    fima, counts = [], []
    for s in window:
        path = os.path.join(args.run, f"{records[s]['run_name']}_discard_inventory{s}.json")
        if os.path.exists(path):
            with open(path) as f:
                for entry in json.load(f).values():
                    fima.append(entry["FIMA"])
                    counts.append(entry["count"])
    if counts:
        summary["discharge_fima"] = {"mean": float(np.average(fima, weights=counts)),
                                     "pebbles_per_step": float(sum(counts) / len(window)),
                                     "benchmark": DISCHARGE_FIMA_BENCHMARK}

    # compositions, sampled over the window
    sampled = [s for s in window if (window[-1] - s) % args.every == 0
               and os.path.exists(os.path.join(args.run, f"core_materials_{s}.json.gz"))]
    zones, table4 = {}, {za: np.zeros(8) for za in TABLE4}
    for s in sampled:
        groups = zone_compositions(args.run, s)
        for key in ecc:
            a, r = (int(x) for x in re.match(r"Z(\d+)R(\d+)", key).groups())
            members = [k for k in groups if k[0] == r and k[1] == a]
            avg = average(groups, members, ECC_NUCLIDES)
            for za, v in avg.items():
                zones.setdefault(key, {}).setdefault(za, []).append(v)
        top = max(k[1] for k in groups)
        for g in range(1, 9):
            members = [k for k in groups if k[1] == top and k[2] == g]
            if members:
                avg = average(groups, members, TABLE4)
                for za in TABLE4:
                    table4[za][g - 1] += grams(avg[za], za) / len(sampled)
    summary["sampled_steps"] = sampled
    ratio = {za: {} for za in ECC_NUCLIDES}
    for key, by_za in zones.items():
        for za, values in by_za.items():
            if ecc[key].get(za):
                ratio[za][key] = float(np.mean(values) / ecc[key][za])
    summary["ecc_ratio"] = {
        NAMES.get(za, za): {"core_mean": float(np.mean(list(v.values()))), "min": float(min(v.values())),
                            "max": float(max(v.values()))} for za, v in ratio.items() if v}
    summary["table4"] = {NAMES[za]: {"pearlsim_g": [float(x) for x in table4[za]], "kpacs_g": TABLE4[za],
                                     "ratio": [float(x / y) for x, y in zip(table4[za], TABLE4[za], strict=True)]}
                         for za in TABLE4}
    with open(os.path.join(args.out, "summary.json"), "w") as f:
        json.dump(summary, f, indent=1)

    lines = [f"# PEARLSim vs KPACS equilibrium ({os.path.basename(os.path.normpath(args.run))})", "",
             f"Steps {steps[0]}-{steps[-1]} ({summary['complete_residences']:.2f} residences of 80 steps); "
             f"window {window[0]}-{window[-1]}; compositions sampled at {len(sampled)} steps.", "",
             "| quantity | PEARLSim | benchmark |", "|---|---|---|",
             f"| k (mid-step mean over window) | {summary['keff']['mid_mean']:.5f} (step-to-step sd "
             f"{summary['keff']['mid_std'] or 0:.5f}) | KPACS {KPACS_K:.5f} +/- {KPACS_K_UNC:.5f} |",
             f"| k BOS / EOS mean | {summary['keff']['bos_mean']:.5f} / {summary['keff']['eos_mean']:.5f} | |"]
    if "discharge_fima" in summary:
        lines.append(f"| discharge burnup (%FIMA) | {summary['discharge_fima']['mean']:.2f} | "
                     f"{DISCHARGE_FIMA_BENCHMARK} (60% packing) |")
    lines += ["", "## Zone compositions / ECC (40 zones)", "", "| nuclide | core mean | min | max |", "|---|---|---|---|"]
    lines += [f"| {n} | {v['core_mean']:.3f} | {v['min']:.3f} | {v['max']:.3f} |" for n, v in summary["ecc_ratio"].items()]
    lines += ["", "## Table 4: per-pass mass per pebble, top axial zone (PEARLSim / KPACS)", "",
              "| nuclide | " + " | ".join(f"pass {g}" for g in range(1, 9)) + " |", "|---" * 9 + "|"]
    lines += [f"| {n} | " + " | ".join(f"{x:.3f}" for x in v["ratio"]) + " |" for n, v in summary["table4"].items()]
    with open(os.path.join(args.out, "summary.md"), "w") as f:
        f.write("\n".join(lines) + "\n")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(steps, bos, lw=1, label="v2 BOS")
    ax.plot(steps, eos, lw=1, label="v2 EOS")
    if args.v1:
        v1 = v1_keff(args.v1)
        ax.plot(list(v1), [v[0] for v in v1.values()], lw=0.8, alpha=0.6, label="v1 BOS (equal-area zones, 25 ppm, 1.74)")
    ax.axhspan(KPACS_K - KPACS_K_UNC, KPACS_K + KPACS_K_UNC, color="k", alpha=0.2, label="KPACS equilibrium")
    ax.axvspan(window[0], window[-1], color="tab:green", alpha=0.08, label="equilibrium window")
    ax.set_xlabel("step (6.525 d)")
    ax.set_ylabel("k_eff")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(args.out, "keff.png"), dpi=150)
    print("\n".join(lines))


if __name__ == "__main__":
    main()
