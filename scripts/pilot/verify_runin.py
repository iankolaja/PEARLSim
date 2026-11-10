"""Verification report for run-in pilots: approach to equilibrium, waves, noise, controller.

    python scripts/pilot/verify_runin.py RUN_DIR [RUN_DIR ...] --out OUT_DIR

For each run (a PEARLSim run directory with step records; its deck <dir>.inp next to it tells whether k is
controlled) the analysed segment is the steps with fixed settings: after the last power change and, for open-loop
runs, after the last change of the inserted fuel fraction (a controlled run varies it itself). Over it:

* residence time R (steps) = fuel pebbles in the core / fuel pebbles discarded per step, averaged;
* equilibrium: means of k_eos, discharge rate and discharged burnup over consecutive windows of R steps counted
  back from the last step, and the change between the last two windows (drift per residence);
* waves: each series minus its mean in each window, the standard deviation per window (damping if it falls from
  window to window), the dominant period of the detrended full-power series (periodogram) and its
  autocorrelation at that period;
* noise: the replica spread of k_eos (pcm), the implied 1-sigma of the replica mean, and Serpent's own estimate;
* controller (runs whose deck sets set_k_target): k_eos minus the target.

Context only, not pass/fail: the KPACS benchmark equilibrium (k 1.01230) and PEARLSim's own benchmark-mode
equilibrium (k 1.01443, 17.97% FIMA discharge; ENDF/B-VII.0, isothermal, pass mode, benchmark zoning), which
differ from the v2 pilots in data, temperatures, discharge rule and zoning.

Writes OUT_DIR/summary.json, summary.md, trajectories.png and fixed_settings.png.
"""

import argparse
import json
import os
import re
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from pearlsim import gather  # noqa: E402

CONTEXT = {"kpacs_equilibrium_k": 1.01230, "pearlsim_benchmark_mode_equilibrium_k": 1.01443,
           "pearlsim_benchmark_mode_discharge_fima": 17.97}
# categorical slots 1-2 of the reference palette (validated light-mode steps), ink and grid
COLORS = ["#2a78d6", "#eb6834"]
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
WAVE_SERIES = {"keff_eos_pcm": "k at end of step (pcm from its mean)", "num_discarded": "fuel pebbles discarded per step",
               "avg_FIMA": "mean burnup of discharged pebbles (% FIMA)"}


def discard_burnup(directory, run_name, steps):
    """Mean burnup (% FIMA) of the fuel pebbles discarded in each step (NaN without discards)."""
    out = []
    for step in steps:
        path = os.path.join(directory, f"{run_name}_discard_inventory{step}.json")
        if not os.path.isfile(path):
            out.append(np.nan)
            continue
        with open(path) as f:
            bins = json.load(f)
        pairs = [(b["FIMA"], b["count"]) for name, b in bins.items() if "fuel" in name and b["count"]]
        count = sum(n for _, n in pairs)
        out.append(sum(f * n for f, n in pairs) / count if count else np.nan)
    return np.array(out)


def dominant_period(values, shortest=4):
    """Period (steps) of the largest periodogram peak of a detrended series, and the autocorrelation there."""
    x = values - np.polyval(np.polyfit(np.arange(len(values)), values, 1), np.arange(len(values)))
    if len(x) < 3 * shortest or not np.any(x):
        return None, None
    power = np.abs(np.fft.rfft(x)) ** 2
    freq = np.fft.rfftfreq(len(x))
    ok = (freq > 0) & (freq <= 1.0 / shortest)
    if not ok.any():
        return None, None
    period = 1.0 / freq[ok][np.argmax(power[ok])]
    lag = int(round(period))
    acf = float(np.corrcoef(x[:-lag], x[lag:])[0, 1]) if 2 < lag < len(x) - 2 else None
    return float(period), acf


def k_target(directory):
    """The k target of the run's deck (<dir>.inp beside the run directory), or None for an open-loop run."""
    deck = os.path.normpath(directory) + ".inp"
    target = None
    if os.path.isfile(deck):
        with open(deck) as f:
            for line in f:
                tokens = line.split("#", 1)[0].split()
                if tokens[:1] == ["set_k_target"]:
                    target = None if tokens[1] == "off" else float(tokens[1])
    return target


def analyse(directory):
    run_name = gather.detect_run_name(directory)
    recorded = [int(m.group(1)) for m in map(re.compile(rf"^{re.escape(run_name)}_record_(\d+)\.json$").match,
                                               os.listdir(directory)) if m]
    # finished steps only: a running step may have some replicas' outputs but no record yet
    table, meta = gather.gather_run(directory, run_name=run_name, last=max(recorded), meshes=False)
    table = table.sort_values("step").reset_index(drop=True)
    name = os.path.basename(os.path.normpath(directory))
    steps = table["step"].to_numpy()
    k = table["keff_eos"].to_numpy(float)
    table["discard_fima"] = discard_burnup(directory, meta["run_name"], steps)
    table["fuel_inserted"] = 1.0 - table["graphite_insertion_fraction"].to_numpy(float)
    power = table["power"].to_numpy(float)
    target_k = k_target(directory)
    changed = np.abs(np.diff(power)) > 1e-3 * power.max()
    if target_k is None:
        changed |= np.abs(np.diff(table["fuel_inserted"].to_numpy(float))) > 1e-6
    first_full = int(np.flatnonzero(changed)[-1] + 1) if changed.any() else 0
    seg = table.iloc[first_full:].reset_index(drop=True)
    discards = seg["num_discarded"].to_numpy(float)
    fuel = seg["fuel_pebbles"].to_numpy(float)
    rate_ok = (discards > 0) & np.isfinite(fuel)
    residence = float(np.mean(fuel[rate_ok]) / np.mean(discards[rate_ok])) if rate_ok.any() else None
    no_inventory = table["fuel_pebbles"].isna()
    summary = {"run": name, "directory": os.path.abspath(directory), "steps": [int(steps[0]), int(steps[-1])],
               "days": float(table["days"].iloc[-1]), "controlled_k_target": target_k,
               "fixed_settings_from_step": int(steps[first_full]), "fixed_settings_steps": len(seg),
               "power_MW": float(power[first_full] / 1e6), "residence_steps": residence,
               "steps_without_inventory": [int(s) for s in table.loc[no_inventory, "step"]]}
    window = int(round(residence)) if residence else None
    windows = []
    if window and len(seg) >= window:
        for end in range(len(seg), window - 1, -window):
            part = seg.iloc[end - window:end]
            kk = part["keff_eos"].to_numpy(float)
            entry = {"steps": [int(part["step"].iloc[0]), int(part["step"].iloc[-1])],
                     "k_eos_mean": float(kk.mean()), "k_eos_std_pcm": float(1e5 * kk.std()),
                     "discarded_per_step_mean": float(part["num_discarded"].mean()),
                     "discarded_per_step_std": float(part["num_discarded"].std()),
                     "discharged_fima_mean": float(part["avg_FIMA"].mean()),
                     "discharged_fima_std": float(part["avg_FIMA"].std()),
                     "discard_fima_mean": float(np.nanmean(part["discard_fima"])),
                     "fuel_inserted_mean": float(part["fuel_inserted"].mean())}
            windows.append(entry)
        windows.reverse()
    summary["residence_windows"] = windows
    if len(windows) >= 2:
        last, before = windows[-1], windows[-2]
        summary["drift_last_residence"] = {
            "k_eos_pcm": 1e5 * (last["k_eos_mean"] - before["k_eos_mean"]),
            "discarded_per_step": last["discarded_per_step_mean"] - before["discarded_per_step_mean"],
            "discharged_fima": last["discharged_fima_mean"] - before["discharged_fima_mean"]}
    summary["waves"] = {}
    for column in ("keff_eos", "num_discarded", "avg_FIMA"):
        values = seg[column].to_numpy(float)
        period, acf = dominant_period(values)
        scale = 1e5 if column == "keff_eos" else 1.0
        in_window = [(seg["step"] >= first) & (seg["step"] <= last) for first, last in (w["steps"] for w in windows)]
        summary["waves"][column] = {"dominant_period_steps": period, "autocorrelation_at_period": acf,
                                    "std_per_residence_window": [scale * float(np.std(values[m.to_numpy()]))
                                                                 for m in in_window]}
    rep = table["keff_eos_rep_std"].to_numpy(float) / k ** 2 * 1e5
    n = table["replicas"].to_numpy(float)
    recent = slice(max(0, len(table) - 100), len(table))
    summary["noise_pcm"] = {"replica_std_median": float(np.nanmedian(rep[recent])),
                            "replica_mean_1sigma": float(np.nanmedian((rep / np.sqrt(n))[recent])),
                            "serpent_1sigma_of_mean": float(np.nanmedian(
                                (table["keff_eos_unc"].to_numpy(float) * 1e5 / k)[recent])),
                            "u235_max_rel_spread": float(np.nanmedian(table["replica_max_rel_std_92235"][recent])),
                            "pu239_max_rel_spread": float(np.nanmedian(table["replica_max_rel_std_94239"][recent]))}
    if target_k is not None:
        error = 1e5 * (seg["keff_eos"].to_numpy(float) - target_k)
        summary["controller"] = {"target_k": target_k, "k_eos_minus_target_mean_pcm": float(error.mean()),
                                 "k_eos_minus_target_std_pcm": float(error.std()),
                                 "fuel_inserted_range": [float(seg["fuel_inserted"].min()),
                                                         float(seg["fuel_inserted"].max())]}
    summary["temperatures_K_max"] = {c[2:]: float(table[c].max()) for c in table if c.startswith("T_")
                                     and c != "T_salt_density"}
    summary["peaking_mesh_full_power"] = [float(seg["peaking_mesh"].min()), float(seg["peaking_mesh"].max())]
    return summary, table, first_full


def _style(ax):
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=8)


def plot(runs, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    panels = [("keff_eos", "k at end of step"), ("fuel_inserted", "fuel fraction inserted"),
              ("power", "power (MW)"), ("num_discarded", "fuel pebbles discarded per step"),
              ("avg_FIMA", "mean burnup of discharged pebbles (% FIMA)"), ("peaking_mesh", "power peaking (mesh)")]
    fig, axes = plt.subplots(len(panels), 1, figsize=(9, 2.0 * len(panels)), sharex=True)
    for (column, label), ax in zip(panels, axes, strict=True):
        for (summary, table, _), color in zip(runs, COLORS, strict=False):
            y = table[column].to_numpy(float) / (1e6 if column == "power" else 1.0)
            ax.plot(table["step"], y, color=color, linewidth=1.5, label=summary["run"])
            if column == "keff_eos":
                sigma = table["keff_eos_rep_std"].to_numpy(float) / np.sqrt(table["replicas"].to_numpy(float))
                ax.fill_between(table["step"], y - sigma, y + sigma, color=color, alpha=0.18, linewidth=0)
        ax.set_title(label, loc="left", fontsize=9, color=INK)
        _style(ax)
    axes[0].legend(frameon=False, fontsize=8, loc="lower right")
    axes[-1].set_xlabel("step (6.525 d each)", color=MUTED, fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "trajectories.png"), dpi=130)
    plt.close(fig)

    fig, axes = plt.subplots(len(WAVE_SERIES), 1, figsize=(9, 2.2 * len(WAVE_SERIES)), sharex=True)
    for (column, label), ax in zip(WAVE_SERIES.items(), axes, strict=True):
        for (summary, table, first_full), color in zip(runs, COLORS, strict=False):
            seg = table.iloc[first_full:]
            source = "keff_eos" if column == "keff_eos_pcm" else column
            y = seg[source].to_numpy(float) * (1e5 if column == "keff_eos_pcm" else 1.0)
            if column == "keff_eos_pcm":
                y = y - y.mean()
            ax.plot(seg["step"], y, color=color, linewidth=1.5, label=summary["run"])
            for w in summary["residence_windows"]:
                ax.axvline(w["steps"][0], color=GRID, linewidth=0.8)
        ax.set_title(label + " with fixed settings", loc="left", fontsize=9, color=INK)
        _style(ax)
    axes[0].legend(frameon=False, fontsize=8, loc="lower right")
    axes[-1].set_xlabel("step (light vertical lines: residence windows, counted back from the last step)",
                        color=MUTED, fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "fixed_settings.png"), dpi=130)
    plt.close(fig)


def markdown(summaries):
    lines = ["# Run-in pilot verification", "",
             "Context (not pass/fail; different data, temperatures, discharge rule and zoning): KPACS equilibrium "
             f"k {CONTEXT['kpacs_equilibrium_k']}; PEARLSim benchmark-mode equilibrium k "
             f"{CONTEXT['pearlsim_benchmark_mode_equilibrium_k']}, discharge "
             f"{CONTEXT['pearlsim_benchmark_mode_discharge_fima']}% FIMA.", ""]
    for s in summaries:
        lines += [f"## {s['run']}", "",
                  f"- steps {s['steps'][0]}-{s['steps'][1]} ({s['days']:.0f} d); "
                  + (f"k controlled to {s['controlled_k_target']}" if s["controlled_k_target"] else "open loop")
                  + f"; fixed settings ({s['power_MW']:.0f} MW) from step {s['fixed_settings_from_step']} "
                  f"({s['fixed_settings_steps']} steps)",
                  f"- residence time ~{s['residence_steps']:.1f} steps" if s["residence_steps"] else "- no discards yet"]
        if s["steps_without_inventory"]:
            lines.append(f"- DATA GAP: no zone map or inventory for steps {s['steps_without_inventory']}")
        if s["residence_windows"]:
            lines += ["", "| window (steps) | k_eos mean | k_eos std (pcm) | discarded/step | discharged FIMA | "
                          "discard FIMA | fuel inserted |", "|---|---|---|---|---|---|---|"]
            for w in s["residence_windows"]:
                lines.append(f"| {w['steps'][0]}-{w['steps'][1]} | {w['k_eos_mean']:.5f} | {w['k_eos_std_pcm']:.0f} | "
                             f"{w['discarded_per_step_mean']:.0f} +- {w['discarded_per_step_std']:.0f} | "
                             f"{w['discharged_fima_mean']:.2f} +- {w['discharged_fima_std']:.2f} | "
                             f"{w['discard_fima_mean']:.2f} | {w['fuel_inserted_mean']:.3f} |")
        if "drift_last_residence" in s:
            d = s["drift_last_residence"]
            lines += ["", f"- drift over the last residence: k {d['k_eos_pcm']:+.0f} pcm, discarded/step "
                          f"{d['discarded_per_step']:+.0f}, discharged FIMA {d['discharged_fima']:+.2f}"]
        for column, w in s["waves"].items():
            stds = ", ".join(f"{v:.3g}" for v in w["std_per_residence_window"])
            period = f"{w['dominant_period_steps']:.1f}" if w["dominant_period_steps"] else "-"
            acf = f"{w['autocorrelation_at_period']:+.2f}" if w["autocorrelation_at_period"] is not None else "-"
            lines.append(f"- waves in {column}: dominant period {period} steps (autocorrelation {acf}); "
                         f"std per residence window: {stds or '-'}")
        n = s["noise_pcm"]
        lines.append(f"- noise (last 100 steps, median): replica std {n['replica_std_median']:.0f} pcm; replica mean "
                     f"1 sigma {n['replica_mean_1sigma']:.0f} pcm; Serpent 1 sigma {n['serpent_1sigma_of_mean']:.0f} pcm; "
                     f"max per-material replica spread U-235 {100 * n['u235_max_rel_spread']:.1f}%, Pu-239 "
                     f"{100 * n['pu239_max_rel_spread']:.1f}%")
        if "controller" in s:
            c = s["controller"]
            lines.append(f"- controller (target {c['target_k']}): k_eos - target {c['k_eos_minus_target_mean_pcm']:+.0f} "
                         f"+- {c['k_eos_minus_target_std_pcm']:.0f} pcm; fuel inserted {c['fuel_inserted_range'][0]:.2f}"
                         f"-{c['fuel_inserted_range'][1]:.2f}")
        t = s["temperatures_K_max"]
        lines += [f"- max temperatures (K): kernel {t.get('kernel', float('nan')):.0f}, salt {t.get('salt', float('nan')):.0f}; "
                  f"mesh peaking at full power {s['peaking_mesh_full_power'][0]:.2f}-{s['peaking_mesh_full_power'][1]:.2f}",
                  ""]
    return "\n".join(lines) + "\n"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("runs", nargs="+")
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    os.makedirs(args.out, exist_ok=True)
    runs = [analyse(d) for d in args.runs]
    summaries = [r[0] for r in runs]
    with open(os.path.join(args.out, "summary.json"), "w") as f:
        json.dump({"context": CONTEXT, "runs": summaries}, f, indent=1)
    with open(os.path.join(args.out, "summary.md"), "w") as f:
        f.write(markdown(summaries))
    plot(runs, args.out)
    print(markdown(summaries))


if __name__ == "__main__":
    main()
