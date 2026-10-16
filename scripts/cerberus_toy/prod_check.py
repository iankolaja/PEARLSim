"""Pebble parking on a real production deck: start-up cost and k at full scale.

    python scripts/cerberus_toy/prod_check.py build RUN_DIR STEP OUT   (no Serpent; login node is fine)
    python scripts/cerberus_toy/prod_check.py run OUT --exe SSS2 --omp N   (batch: run_prod.sub)

build copies the v1 deck of step STEP (e.g. gFHR_benchmark_1975, 100) into OUT and writes:
  ref/   the deck and its pebble file unchanged (trisos/ is a symlink to RUN_DIR/trisos, read only)
  live/  the same deck as a live deck: no "set inventory", "dep daytot" (both forced by the HxF
         build, see build_toy.py), pebble file pool.csv. Every in-core position carries one pebble of
         each universe of its zone; radius 2.0 for the universe the step-STEP file places there and 0
         (parked) for the others, so the pool is pebbles x universes-per-zone (~2.0 M for 250 k pebbles
         and 8 universes per zone).
  prod.json  pool size, universes, and the v1 run's own INIT_TIME and ANA_KEFF.
run (each part writes OUT/results.json when it finishes; a finished ref is reused):
  ref   file-based run of ref/ with --omp N: INIT_TIME, k at BOS
  live  live session on live/: start-up time, full xyzr send, rebuild + solve, k at BOS vs ref
Nothing outside OUT is written.
"""

import argparse
import json
import os
import re
import shutil
import sys
import time
import traceback

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from driver import Live, Results, compare, res_values  # noqa: E402


def zone_of(universe):
    match = re.search(r"_(R\d+Z\d+)G", universe)
    return match.group(1) if match else "other"


def build(run_dir, step, out):
    deck_name = f"gFHR_equilibrium_{step}.serpent"
    pebble_name = f"pebble_positions_{step}.csv"
    with open(os.path.join(run_dir, deck_name)) as f:
        deck = f.read()
    if f'pbed u_pb u_flibe "{pebble_name}"' not in deck:
        raise ValueError("pbed card does not name the expected pebble file")

    ref = os.path.join(out, "ref")
    os.makedirs(ref, exist_ok=True)
    shutil.copyfile(os.path.join(run_dir, deck_name), os.path.join(ref, deck_name))
    shutil.copyfile(os.path.join(run_dir, pebble_name), os.path.join(ref, pebble_name))
    link = os.path.join(ref, "trisos")
    if not os.path.lexists(link):
        os.symlink(os.path.abspath(os.path.join(run_dir, "trisos")), link)

    live_text, n = re.subn(r"^set inventory .*\n", "", deck, flags=re.M)
    if n != 1:
        raise ValueError(f"expected one 'set inventory' line, found {n}")
    live_text, n = re.subn(r"^dep daystep ", "dep daytot ", live_text, flags=re.M)
    if n != 1:
        raise ValueError(f"expected one 'dep daystep' line, found {n}")
    live_text = live_text.replace(f'"{pebble_name}"', '"pool.csv"')

    xyz, unis = [], []
    with open(os.path.join(run_dir, pebble_name)) as f:
        for line in f:
            x, y, z, _, uni = line.split()
            xyz.append((x, y, z))
            unis.append(uni)
    by_zone = {}
    for uni in unis:
        by_zone.setdefault(zone_of(uni), set()).add(uni)
    by_zone = {z: sorted(u) for z, u in by_zone.items()}

    live = os.path.join(out, "live")
    os.makedirs(live, exist_ok=True)
    with open(os.path.join(live, "live.serpent"), "w") as f:
        f.write(live_text)
    pool = 0
    with open(os.path.join(live, "pool.csv"), "w") as f:
        for (x, y, z), uni in zip(xyz, unis, strict=True):
            for u in by_zone[zone_of(uni)]:
                f.write(f"{x} {y} {z} {'2.0' if u == uni else '0.0'} {u}\n")
                pool += 1
    link = os.path.join(live, "trisos")
    if not os.path.lexists(link):
        os.symlink(os.path.abspath(os.path.join(run_dir, "trisos")), link)

    res = os.path.join(run_dir, deck_name + "_res.m")
    manifest = {"run_dir": os.path.abspath(run_dir), "step": step, "deck": deck_name, "pebbles": len(unis),
                "universes": len(set(unis)), "zones": len(by_zone), "pool": pool,
                "v1_init_time_min": res_values(res, "INIT_TIME")[0][0],
                "v1_ana_keff": res_values(res, "ANA_KEFF")[0][:2],
                "v1_omp_threads": res_values(res, "OMP_THREADS")[0][0]}
    with open(os.path.join(out, "prod.json"), "w") as f:
        json.dump(manifest, f, indent=1)
    print(json.dumps(manifest, indent=1))


def run(out, exe, omp, port):
    import subprocess

    out = os.path.abspath(out)
    os.chdir(out)  # Cerberus creates wrk_<name>/ in the current directory
    manifest = json.load(open(os.path.join(out, "prod.json")))
    results = Results(os.path.join(out, "results.json"))
    deck = manifest["deck"]

    ref = results.get("ref", {})
    if not (ref.get("returncode") == 0 and "k_bos" in ref):
        ref_dir = os.path.join(out, "ref")
        start = time.time()
        with open(os.path.join(ref_dir, "serpent.log"), "w") as log:
            code = subprocess.run([exe, deck, "-omp", str(omp)], cwd=ref_dir, stdout=log,
                                  stderr=subprocess.STDOUT).returncode
        ref = {"returncode": code, "wall_s": time.time() - start}
        res = os.path.join(ref_dir, deck + "_res.m")
        if code == 0 and os.path.exists(res):
            k = res_values(res, "ANA_KEFF")
            ref.update(k_bos=k[0][0], k_bos_rel_unc=k[0][1], init_time_min=res_values(res, "INIT_TIME")[0][0],
                       running_time_min=res_values(res, "RUNNING_TIME")[0][0])
        results["ref"] = ref
    print(f"ref: {ref}", flush=True)

    live_dir = os.path.join(out, "live")
    result = {"pool": manifest["pool"]}
    live = Live(exe, omp, [os.path.join(live_dir, "live.serpent"), os.path.join(live_dir, "pool.csv"),
                           os.path.join(live_dir, "trisos")], "serpent_prod", port)
    result["init_s"] = live.init_s
    try:
        array = np.loadtxt(os.path.join(live_dir, "pool.csv"), usecols=(0, 1, 2, 3)).ravel()
        live.solver.set_current_time(0.0)
        t = time.time()
        live.send("sss_iv_pbed_u_pb_xyzr", array)
        result["send_xyzr_s"] = time.time() - t
        result["send_mb"] = array.nbytes / 1e6
        t = time.time()
        live.solver.solve()
        result["rebuild_and_solve_s"] = time.time() - t
        if "k_bos" in ref:
            result["k_bos"] = compare(live.k(), ref["k_bos"], ref["k_bos_rel_unc"])
        else:
            result["k_bos"] = live.k()
    except Exception:  # noqa: BLE001 - record what happened
        result["error"] = traceback.format_exc()
    results["live"] = result
    live.close()
    print(json.dumps(results.data, indent=1, default=float), flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = parser.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("run_dir")
    b.add_argument("step", type=int)
    b.add_argument("out")
    r = sub.add_parser("run")
    r.add_argument("out")
    r.add_argument("--exe", required=True)
    r.add_argument("--omp", type=int, default=16)
    r.add_argument("--port", type=int, default=2221)
    args = parser.parse_args(argv)
    if args.cmd == "build":
        build(args.run_dir, args.step, args.out)
    else:
        run(args.out, args.exe, args.omp, args.port)


if __name__ == "__main__":
    sys.exit(main())
