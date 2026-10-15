"""A stand-in for the Serpent executable, for testing PEARLSim without Serpent.

Usage (as PEARLSim invokes Serpent): ``fake_sss2.py <deck> [-omp N]``

Reads the deck, finds burnable materials (``mat ... burn 1``) and the number of
``dep daystep`` intervals, and writes:

* ``<deck>_res.m``, ``<deck>_det0.m``, ``<deck>_det1.m``: canned copies of real
  v1 outputs (tests/fixtures/equilibrium_step100);
* ``<deck>.bumat0`` .. ``<deck>.bumat<N>``: every burnable material, renamed the
  way Serpent renames materials inside particles (``<mat>pp<k>r1``), with a
  deterministic "depletion": U-235 scaled by (1 - 0.01 step (1 + 0.05 k)) for
  depletion step ``step`` and material index ``k`` (so materials differ in
  burnup), and Xe-135 added.

Environment:
  FAKE_SERPENT_MODE  ok (default) | fail (exit 1, no output) | no_output (exit 0,
                     no output) | partial (only bumat0)
  FAKE_SERPENT_LOG   path of a JSON-lines file recording each invocation
  FAKE_SERPENT_FAIL_DECKS  comma-separated substrings; a deck whose name contains
                     one fails like mode "fail" (one replica killed)
  FAKE_SERPENT_SEED_NOISE  if set, U-235 is also scaled by (1 + 1e-3 (seed mod 97)),
                     where seed is the deck's "set seed", so replicas differ
  FAKE_SERPENT_SLEEP seconds to wait before writing outputs (concurrency tests)
"""

import json
import os
import shutil
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
CANNED = os.path.join(HERE, "..", "fixtures", "equilibrium_step100")
CANNED_STEM = "gFHR_equilibrium_100.serpent"


def parse_deck(path):
    materials = {}
    order = []
    daysteps = 0
    seed = 0
    current = None
    with open(path) as f:
        for raw in f:
            line = raw.split("%")[0].rstrip("\n")
            tokens = line.split()
            if not tokens:
                current = None
                continue
            if tokens[:2] == ["set", "seed"] and len(tokens) > 2:
                seed = int(tokens[2])
                current = None
            elif tokens[0] == "dep" and len(tokens) > 1 and tokens[1] == "daystep":
                daysteps = len(tokens) - 2
                current = None
            elif tokens[0] == "mat":
                if "burn" in tokens:
                    current = tokens[1]
                    materials[current] = {}
                    order.append(current)
                else:
                    current = None
            elif current is not None and len(tokens) == 2 and line.startswith(" "):
                try:
                    materials[current][tokens[0]] = float(tokens[1])
                except ValueError:
                    current = None
            else:
                current = None
    return order, materials, daysteps, seed


def write_bumat(path, order, materials, step, noise=1.0):
    with open(path, "w") as f:
        f.write(f"% Fake Serpent material compositions, step {step}\n\n")
        for k, name in enumerate(order, start=1):
            f.write(f"mat  {name}pp{k}r1  0.0 fix 09c 9.59000E+02 vol 1.0\n")
            for nuclide, density in materials[name].items():
                if nuclide.startswith("92235"):
                    density = density * (1.0 - 0.01 * step * (1.0 + 0.05 * k))
                    if step > 0:
                        density *= noise
                f.write(f"            {nuclide}  {density:.14E}\n")
            if step > 0:
                f.write(f"               541350  {1.0e-9 * step:.14E}\n")
            f.write("\n")


def main(argv):
    deck = argv[1]
    mode = os.environ.get("FAKE_SERPENT_MODE", "ok")
    fail_decks = [d for d in os.environ.get("FAKE_SERPENT_FAIL_DECKS", "").split(",") if d]
    if any(d in os.path.basename(deck) for d in fail_decks):
        mode = "fail"
    order, materials, daysteps, seed = parse_deck(deck)
    noise = 1.0 + 1e-3 * (seed % 97) if os.environ.get("FAKE_SERPENT_SEED_NOISE") else 1.0
    start = time.time()
    time.sleep(float(os.environ.get("FAKE_SERPENT_SLEEP", "0")))
    log = os.environ.get("FAKE_SERPENT_LOG")
    if log:
        with open(log, "a") as f:
            f.write(json.dumps({"argv": argv[1:], "cwd": os.getcwd(), "mode": mode,
                                "burnable_materials": len(order), "daysteps": daysteps,
                                "start": start, "end": time.time()}) + "\n")
    if mode == "fail":
        return 1
    if mode == "no_output":
        return 0
    for suffix in ("_res.m", "_det0.m", "_det1.m"):
        shutil.copyfile(os.path.join(CANNED, CANNED_STEM + suffix), deck + suffix)
    last = 0 if mode == "partial" else daysteps
    for step in range(last + 1):
        write_bumat(f"{deck}.bumat{step}", order, materials, step, noise)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
