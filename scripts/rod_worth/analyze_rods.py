"""Rod-worth curves from build_rod_decks.py runs, and the common-random-numbers check.

    python scripts/rod_worth/analyze_rods.py OUT_DIR STATE_DIR [STATE_DIR ...]

For each state: w(z) = rho(fully withdrawn) - rho(z) in pcm, total worth W_tot = w(60 cm), the normalised
shape f(z) = w(z) / W_tot, and the independent-run error of each w (Serpent's errors added in quadrature).
For a state run with two Serpent seeds, the seed-to-seed difference of w(z) measures the actual noise in
worth differences: sigma_w(observed) = std(w_ms0 - w_ms1) / sqrt(2), against the independent-run value.
Writes OUT_DIR/rod_worth.{json,md} and rod_worth.png.
"""

import argparse
import glob
import json
import math
import os
import re

import numpy as np

NAME = re.compile(r"rod_z([\d.]+)_ms(\d+)\.serpent_res\.m$")
WITHDRAWN = 369.47


def ana_keff(path):
    with open(path) as f:
        for line in f:
            if line.startswith("ANA_KEFF"):
                values = line.split("=", 1)[1].split("[", 1)[1].split()
                return float(values[0]), float(values[1])
    raise ValueError(path)


def read_state(state_dir):
    runs = {}
    for path in glob.glob(os.path.join(state_dir, "rod_z*_ms*.serpent_res.m")):
        z, j = NAME.search(path).groups()
        k, rel = ana_keff(path)
        runs.setdefault(int(j), {})[float(z)] = (1e5 * (1 - 1 / k), 1e5 * rel / k)
    info = {}
    if os.path.exists(os.path.join(state_dir, "state.txt")):
        with open(os.path.join(state_dir, "state.txt")) as f:
            info = dict(line.split(None, 1) for line in f.read().splitlines() if line.strip())
    return runs, {k: v.strip() for k, v in info.items()}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("out")
    parser.add_argument("states", nargs="+")
    args = parser.parse_args(argv)
    os.makedirs(args.out, exist_ok=True)
    summary, lines = {}, ["| state | fuel fraction | rho out (pcm) | W_tot (pcm) | f(z) at 280/240/190/150/110 cm |"
                          " sigma_w independent (pcm) | sigma_w observed, 2 seeds (pcm) |", "|---|---|---|---|---|---|---|"]
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    for state_dir in args.states:
        runs, info = read_state(state_dir)
        label = f"{os.path.basename(os.path.dirname(os.path.abspath(state_dir)))}/{os.path.basename(state_dir)}"
        curves = {}
        for j, by_z in runs.items():
            if WITHDRAWN not in by_z:
                continue
            rho0, e0 = by_z[WITHDRAWN]
            curves[j] = {z: (rho0 - rho, math.hypot(e0, e)) for z, (rho, e) in sorted(by_z.items())}
        if not curves:
            continue
        c0 = curves[min(curves)]
        zs = sorted(c0)
        w = np.array([np.mean([curves[j][z][0] for j in curves if z in curves[j]]) for z in zs])
        w_tot = w[zs.index(60.0)] if 60.0 in zs else math.nan
        indep = float(np.median([c0[z][1] for z in zs if z != WITHDRAWN]))
        observed = None
        if len(curves) > 1:
            j0, j1 = sorted(curves)[:2]
            d = [curves[j0][z][0] - curves[j1][z][0] for z in zs if z != WITHDRAWN and z in curves[j1]]
            observed = float(np.sqrt(np.mean(np.square(d)) / 2))
        shape = {z: wi / w_tot for z, wi in zip(zs, w, strict=True)} if w_tot else {}
        summary[label] = {"info": info, "z_cm": zs, "w_pcm": w.tolist(), "W_tot_pcm": float(w_tot),
                          "rho_withdrawn_pcm": float(np.mean([runs[j][WITHDRAWN][0] for j in curves])),
                          "sigma_w_independent_pcm": indep, "sigma_w_observed_pcm": observed, "shape": shape}
        f_txt = "/".join(f"{shape.get(z, math.nan):.2f}" for z in (280.0, 240.0, 190.0, 150.0, 110.0))
        lines.append(f"| {label} | {info.get('fuel_fraction', '?')} | {summary[label]['rho_withdrawn_pcm']:.0f} | "
                     f"{w_tot:.0f} | {f_txt} | {indep:.0f} | {observed if observed is None else f'{observed:.0f}'} |")
        ax1.plot(zs, w, marker="o", label=label)
        if w_tot:
            ax2.plot(zs, w / w_tot, marker="o", label=label)
    for ax, ylabel in ((ax1, "rod worth w(z) (pcm)"), (ax2, "w(z) / W_tot")):
        ax.set_xlabel("rod tip position z (cm); bed 60-369.47")
        ax.set_ylabel(ylabel)
        ax.invert_xaxis()
    ax1.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(args.out, "rod_worth.png"), dpi=150)
    with open(os.path.join(args.out, "rod_worth.json"), "w") as f:
        json.dump(summary, f, indent=1)
    with open(os.path.join(args.out, "rod_worth.md"), "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
