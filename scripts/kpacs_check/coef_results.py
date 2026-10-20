"""Reactivity coefficients from the build_coef_decks.py runs, against the gFHR benchmark's Table 5.

    python scripts/kpacs_check/coef_results.py COEF_DIR

Reads <case>.serpent_res.m (ANA_KEFF) and writes COEF_DIR/coefficients.json and .md.
"""

import argparse
import json
import math
import os

# (low case, high case, perturbation span, unit, Table 5 value, Table 5 uncertainty)
COEFFICIENTS = {
    "fuel temperature": ("fuel_m100", "fuel_p100", 200.0, "pcm/K", -4.56, 0.12),
    "coolant temperature": ("coolant_m200", "coolant_p200", 400.0, "pcm/K", -1.20, 0.12),
    "moderator temperature": ("moderator_m300", "moderator_p300", 600.0, "pcm/K", -0.40, 0.12),
    "reflector temperature": ("reflector_m300", "reflector_p300", 600.0, "pcm/K", 0.92, 0.12),
    "void": ("void_m10", "void_p10", 20.0, "pcm/void%", -48.19, 9.05),
    # diagnostics (benchmark value: total CTC)
    "coolant temperature, +/-100 K": ("coolant_m100", "coolant_p100", 200.0, "pcm/K", -1.20, 0.12),
    "coolant temperature only (density fixed)": ("coolant_tonly_m200", "coolant_tonly_p200", 400.0, "pcm/K",
                                                 math.nan, math.nan),
    "coolant density only (temperature fixed)": ("coolant_donly_m200", "coolant_donly_p200", 400.0, "pcm/K",
                                                 math.nan, math.nan),
    "coolant temperature, +/-200 K, 7.5 M histories": ("coolant_m200_hs", "coolant_p200_hs", 400.0, "pcm/K",
                                                       -1.20, 0.12),
}


def ana_keff(path):
    with open(path) as f:
        for line in f:
            if line.startswith("ANA_KEFF"):
                values = line.split("=", 1)[1].split("[", 1)[1].split()
                return float(values[0]), float(values[1])
    raise ValueError(f"no ANA_KEFF in {path}")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("coef_dir")
    args = parser.parse_args(argv)
    results, lines = {}, ["| coefficient | PEARLSim v2 | benchmark (KPACS, Table 5) |", "|---|---|---|"]
    for name, (low, high, span, unit, ref, ref_unc) in COEFFICIENTS.items():
        paths = [os.path.join(args.coef_dir, f"{case}.serpent_res.m") for case in (low, high)]
        if not all(os.path.exists(p) for p in paths):
            continue
        (k_lo, e_lo), (k_hi, e_hi) = (ana_keff(p) for p in paths)
        # rho = 1 - 1/k in pcm; sigma(rho) = rel. error of k / k
        value = 1e5 * (1 / k_lo - 1 / k_hi) / span
        sigma = 1e5 * math.hypot(e_lo / k_lo, e_hi / k_hi) / span
        results[name] = {"value": value, "sigma": sigma, "unit": unit, "k": [k_lo, k_hi], "k_rel_err": [e_lo, e_hi],
                         "benchmark": ref, "benchmark_unc": ref_unc}
        lines.append(f"| {name} ({unit}) | {value:+.2f} +/- {sigma:.2f} | {ref:+.2f} +/- {ref_unc:.2f} |")
    with open(os.path.join(args.coef_dir, "coefficients.json"), "w") as f:
        json.dump(results, f, indent=1)
    with open(os.path.join(args.coef_dir, "coefficients.md"), "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
