"""Run the Cerberus pebble-parking toy test (batch job; see run_toy.sub and build_toy.py).

    python scripts/cerberus_toy/driver.py TOY_DIR --exe SSS2 --omp N [--parts refs,live,empty,scale]

Parts (each writes its results into TOY_DIR/results.json as soon as it finishes):
  refs   ordinary file-based runs of ref_A/B/C: k at the start (BOS) and end (EOS) of a
         6.525-day step, and the final compositions (.bumat1)
  live   one live Serpent session on the pooled deck:
           A: send layout A (rebuilds the pebble search mesh without parked pebbles), solve
           B: move pebbles to layout B, update fuel volumes (child and parent), solve
           back to A, then deplete 6.525 days, solve, read compositions
         k values are compared with the references (z = difference / combined sigma)
  empty  a second session: layout C with the emptied fuel slot at volume 0 (outcome recorded,
         including a crash)
  scale  a session on live_scale/ (~3 M pebbles): start-up time, full-array send and solve time

Large arrays are sent with one sendall (Cerberus's own send loop makes one socket call
per value and does not check partial sends). Needs Cerberus_HxF and KrakenTools_HxF on
PYTHONPATH. Nothing here edits files outside TOY_DIR.
"""

import argparse
import json
import os
import re
import struct
import subprocess
import sys
import time
import traceback

import numpy as np

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, REPO)

from pearlsim.replicas import read_bumat  # noqa: E402

TAKE_VARIABLE_DATA = 16
DAY = 86400.0


class Results:
    def __init__(self, path):
        self.path = path
        self.data = json.load(open(path)) if os.path.exists(path) else {}

    def __setitem__(self, key, value):
        self.data[key] = value
        with open(self.path, "w") as f:
            json.dump(self.data, f, indent=1, default=float)

    def get(self, key, default=None):
        return self.data.get(key, default)


def res_values(path, key):
    """All occurrences of a _res.m vector as lists of floats."""
    out = []
    with open(path) as f:
        for line in f:
            if line.startswith(key + " "):
                values = line.split("=", 1)[1].split(";")[0].replace("[", " ").replace("]", " ")
                out.append([float(v) for v in values.split()])
    return out


def run_reference(toy, name, exe, omp):
    ref = os.path.join(toy, f"ref_{name}")
    deck = f"ref_{name}.serpent"
    start = time.time()
    with open(os.path.join(ref, "serpent.log"), "w") as log:
        code = subprocess.run([exe, deck, "-omp", str(omp)], cwd=ref, stdout=log, stderr=subprocess.STDOUT).returncode
    wall = time.time() - start
    res = os.path.join(ref, deck + "_res.m")
    if code != 0 or not os.path.exists(res):
        return {"returncode": code, "wall_s": wall}
    imp = res_values(res, "ANA_KEFF")
    init = res_values(res, "INIT_TIME")
    bumat = read_bumat(os.path.join(ref, deck + ".bumat1")) if os.path.exists(os.path.join(ref, deck + ".bumat1")) else {}
    return {"returncode": code, "wall_s": wall,
            "k_bos": imp[0][0], "k_bos_rel_unc": imp[0][1], "k_eos": imp[-1][0], "k_eos_rel_unc": imp[-1][1],
            "init_time_min": init[0][0] if init else None,
            "u235_eos": {m: c.get("92235<lib>") for m, c in bumat.items()},
            "xe135_eos": {m: c.get("54135<lib>") for m, c in bumat.items()}}


class Live:
    """A live Serpent session through Cerberus, with a bulk sender for large arrays."""

    def __init__(self, exe, omp, workdir_inputs, name, port):
        from cerberus.solvers import CodeInput, Solver

        self.solver = Solver(name, exe, ["-omp", str(omp), "-port"])
        self.solver.input = CodeInput(workdir_inputs)
        start = time.time()
        self.solver.initialize(port_number=port)
        self.init_s = time.time() - start
        self.names = sorted(self.solver._transferrables)

    def send(self, name, values):
        if name not in self.names:
            raise KeyError(f"Serpent does not offer input variable {name}")
        values = np.ascontiguousarray(values, dtype="<f8")
        sock = self.solver._communicator._connection
        encoded = name.encode("ascii")
        sock.sendall(struct.pack("<q", TAKE_VARIABLE_DATA) + struct.pack("<q", len(encoded)) + encoded
                     + struct.pack("<q", values.size) + values.tobytes())

    def get(self, name):
        if name not in self.names:
            raise KeyError(f"Serpent does not offer output variable {name}")
        variable = self.solver.get_transferrable(name)
        variable.communicate()
        return np.array(variable.value_vec)

    def k(self):
        k = float(self.get("sss_ov_ANA_KEFF")[0])
        return k, float(self.get("sss_ov_ANA_KEFF_rel_unc")[0])

    def child(self, material):
        """Name of the auto-divided burnable child of a material (e.g. fuel20_R1Z1G1pp2r1)."""
        pattern = re.compile(rf"^sss_iv_composition_({re.escape(material)}p\w+)_adens$")
        found = [m.group(1) for m in map(pattern.match, self.names) if m]
        if len(found) != 1:
            raise KeyError(f"expected one divided child of {material}, found {found}")
        return found[0]

    def set_volumes(self, volumes):
        """Volume of each fuel material's divided child; resend compositions to flag the update.

        The parent's own volume is not settable live (sss_iv_material_div_<parent>_volume writes the
        children, receivevardata.c:787). It only weights SumDivCompositions' parent composition, which
        transport, depletion and this driver do not use.
        """
        for mat, vol in volumes.items():
            child = self.child(mat)
            self.send(f"sss_iv_material_{child}_volume", [vol])
            self.send(f"sss_iv_composition_{child}_adens", self.get(f"sss_ov_composition_{child}_adens"))

    def close(self):
        try:
            self.solver.send_signal(0)  # TERMINATE
        except Exception:  # noqa: BLE001 - best effort at the end of a test
            pass


def xyzr(manifest, layout, toy_dir, scale=False):
    pool = np.loadtxt(os.path.join(toy_dir, "live_scale" if scale else "live", "pool.csv"), usecols=(0, 1, 2))
    radii = np.zeros(len(pool))
    r = np.array(manifest["layouts"][layout]["pool_radius"])
    radii[: len(r)] = r
    return np.column_stack([pool, radii]).ravel()


def compare(live_k, ref_k, ref_rel):
    (k, rel), sigma = live_k, np.hypot(live_k[0] * live_k[1], ref_k * ref_rel)
    return {"live": k, "live_rel_unc": rel, "ref": ref_k, "diff_pcm": (k - ref_k) * 1e5,
            "z": (k - ref_k) / sigma if sigma else None}


def part_live(toy, manifest, results, exe, omp, port):
    refs = results.get("refs", {})
    out = {}
    live = Live(exe, omp, [os.path.join(toy, "live", "live.serpent"), os.path.join(toy, "live", "pool.csv"),
                           os.path.join(toy, "trisos")], "serpent_live", port)
    out["init_s"] = live.init_s
    with open(os.path.join(toy, "live_variables.txt"), "w") as f:
        f.write("\n".join(live.names) + "\n")
    out["children"] = {m: live.child(m) for m in manifest["fuel_materials"]}
    try:
        live.solver.set_current_time(0.0)
        t = time.time()
        live.send("sss_iv_pbed_u_pb_xyzr", xyzr(manifest, "A", toy))
        out["send_xyzr_s"] = time.time() - t
        t = time.time()
        live.solver.solve()
        k_a = live.k()
        out["A"] = {"solve_s": time.time() - t, **compare(k_a, refs["A"]["k_bos"], refs["A"]["k_bos_rel_unc"])}
        results["live"] = out

        live.send("sss_iv_pbed_u_pb_xyzr", xyzr(manifest, "B", toy))
        live.set_volumes(manifest["layouts"]["B"]["fuel_volumes"])
        t = time.time()
        live.solver.solve()
        out["B"] = {"solve_s": time.time() - t,
                    **compare(live.k(), refs["B"]["k_bos"], refs["B"]["k_bos_rel_unc"])}
        results["live"] = out

        live.send("sss_iv_pbed_u_pb_xyzr", xyzr(manifest, "A", toy))
        live.set_volumes(manifest["layouts"]["A"]["fuel_volumes"])
        live.solver.solve()
        t = time.time()
        live.solver.advance_to_time(manifest["burn_days"] * DAY)
        live.solver.solve()
        k_eos = live.k()
        deplete = {"advance_and_solve_s": time.time() - t,
                   **compare(k_eos, refs["A"]["k_eos"], refs["A"]["k_eos_rel_unc"]), "u235": {}}
        for mat, child in out["children"].items():
            zai = live.get(f"sss_ov_composition_{child}_zai") if f"sss_ov_composition_{child}_zai" in live.names else None
            adens = live.get(f"sss_ov_composition_{child}_adens")
            if zai is not None:
                idx = np.flatnonzero(np.asarray(zai, dtype=np.int64) == 922350)
                u235 = float(adens[idx[0]]) if idx.size else None
            else:
                u235 = None
            ref = refs["A"]["u235_eos"].get(mat)
            deplete["u235"][mat] = {"live": u235, "ref": ref,
                                    "rel_diff": (u235 - ref) / ref if u235 is not None and ref else None}
        out["deplete_A"] = deplete
    except Exception:  # noqa: BLE001 - record what happened and keep the partial results
        out["error"] = traceback.format_exc()
    results["live"] = out
    live.close()


def part_empty(toy, manifest, results, exe, omp, port):
    refs = results.get("refs", {})
    out = {}
    live = Live(exe, omp, [os.path.join(toy, "live", "live.serpent"), os.path.join(toy, "live", "pool.csv"),
                           os.path.join(toy, "trisos")], "serpent_empty", port)
    try:
        live.solver.set_current_time(0.0)
        live.send("sss_iv_pbed_u_pb_xyzr", xyzr(manifest, "C", toy))
        volumes = dict(manifest["layouts"]["C"]["fuel_volumes"])
        # Serpent rejects a burnable material with volume 0 ("Volume or mass ... must be given"), so the
        # emptied slot gets the same tiny volume as in ref_C.
        volumes.update({m: manifest["empty_volume"] for m in manifest["fuel_materials"] if m not in volumes})
        live.set_volumes(volumes)
        live.solver.solve()
        out["C_empty_slot"] = compare(live.k(), refs["C"]["k_bos"], refs["C"]["k_bos_rel_unc"])
    except Exception:  # noqa: BLE001
        out["error"] = traceback.format_exc()
    results["empty"] = out
    live.close()


def part_scale(toy, manifest, results, exe, omp, port):
    out = {"pebbles": manifest.get("scale_pebbles")}
    live = Live(exe, omp, [os.path.join(toy, "live_scale", "live.serpent"),
                           os.path.join(toy, "live_scale", "pool.csv"), os.path.join(toy, "trisos")],
                "serpent_scale", port)
    out["init_s"] = live.init_s
    try:
        live.solver.set_current_time(0.0)
        array = xyzr(manifest, "A", toy, scale=True)
        t = time.time()
        live.send("sss_iv_pbed_u_pb_xyzr", array)
        out["send_xyzr_s"] = time.time() - t
        out["send_mb"] = array.nbytes / 1e6
        t = time.time()
        live.solver.solve()
        out["k"] = live.k()
        out["rebuild_and_solve_s"] = time.time() - t
    except Exception:  # noqa: BLE001
        out["error"] = traceback.format_exc()
    results["scale"] = out
    live.close()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("toy")
    parser.add_argument("--exe", required=True)
    parser.add_argument("--omp", type=int, default=8)
    parser.add_argument("--port", type=int, default=2211)
    parser.add_argument("--parts", default="refs,live,empty,scale")
    args = parser.parse_args(argv)
    toy = os.path.abspath(args.toy)
    os.chdir(toy)  # Cerberus creates wrk_<name>/ in the current directory
    manifest = json.load(open(os.path.join(toy, "toy.json")))
    results = Results(os.path.join(toy, "results.json"))
    parts = args.parts.split(",")
    if "refs" in parts:
        refs = results.get("refs", {})
        for name in ("A", "B", "C"):
            if refs.get(name, {}).get("returncode") == 0 and "k_bos" in refs[name]:
                continue  # finished in an earlier job (or before a requeue)
            refs[name] = run_reference(toy, name, args.exe, args.omp)
            results["refs"] = refs
    for offset, (part, fn) in enumerate((("live", part_live), ("empty", part_empty), ("scale", part_scale))):
        if part in parts:
            start = time.time()
            fn(toy, manifest, results, args.exe, args.omp, args.port + offset)
            print(f"{part}: {time.time() - start:.0f} s", flush=True)
    print(json.dumps({k: v for k, v in results.data.items() if k != "refs"}, indent=1, default=float)[:4000])


if __name__ == "__main__":
    sys.exit(main())
