"""Pack the files of a PEARLSim run directory that the archive policy keeps (docs/ARCHIVE_POLICY.md).

    python scripts/pack_run.py RUN_DIR                     # summary of keep / drop by kind (no changes)
    python scripts/pack_run.py RUN_DIR --tar OUT.tar.gz    # write the kept files to a compressed tar
        [--checkpoint-every 10]

Nothing in RUN_DIR is modified or deleted. Files that match no rule are reported
as "unknown" and kept, so a new output type is never dropped silently.
"""

import argparse
import os
import re
import sys
import tarfile
from collections import defaultdict

# (kind, regex on the file name, action); the first match wins. "checkpoint" files
# are kept for steps that are multiples of --checkpoint-every (and the last step).
RULES = [
    ("record", r"_record_\d+\.json$", "keep"),
    ("operating", r"_operating_\d+\.json$", "keep"),
    ("zone map", r"^zone_map\d+\.json$", "keep"),
    ("discharge/discard/held", r"_(discharge|discard|held)_inventory\d+\.json$", "keep"),
    ("replica spread", r"_replica_spread_\d+\.json$", "keep"),
    ("serpent results", r"\.serpent_res\.m$", "keep"),
    ("serpent detectors", r"\.serpent_det\d+\.m$", "keep"),
    ("serpent warnings", r"\.serpent\.wrn$", "keep"),
    ("final compositions", r"\.serpent\.bumat(\d+)$", "bumat"),
    ("pebble positions", r"^pebble_positions_(\d+)(_r\d+)?\.csv$", "checkpoint"),
    ("core materials", r"^core_materials_(\d+)\.json\.gz$", "checkpoint"),
    ("deck", r"\.serpent$", "drop"),
    ("serpent output listing", r"\.serpent\.out$", "drop"),
    ("depletion output", r"\.serpent_dep\.m$", "drop"),
    ("binary restart", r"\.serpent\.dep$", "drop"),
    ("seed", r"\.serpent\.seed$", "drop"),
    ("replica log", r"\.serpent\.log$", "drop"),
    ("geometry plot", r"\.serpent_geom\d+\.png$", "drop"),
]


def step_of(name):
    match = re.search(r"_(\d+)(_r\d+)?\.serpent", name) or re.search(r"(\d+)(_r\d+)?\.(json|csv|json\.gz)$", name)
    return int(match.group(1)) if match else None


def classify(directory, checkpoint_every=10):
    """{file name: (kind, keep?)} for the regular files directly inside directory."""
    names = sorted(n for n in os.listdir(directory) if os.path.isfile(os.path.join(directory, n)))
    steps = [s for s in map(step_of, names) if s is not None]
    last_step = max(steps) if steps else None
    final_bumat = {}  # deck stem -> highest bumat index
    for name in names:
        match = re.search(r"^(.*)\.serpent\.bumat(\d+)$", name)
        if match:
            final_bumat[match.group(1)] = max(final_bumat.get(match.group(1), 0), int(match.group(2)))
    result = {}
    for name in names:
        for kind, pattern, action in RULES:
            match = re.search(pattern, name)
            if not match:
                continue
            if action == "bumat":
                stem = name.split(".serpent.bumat")[0]
                keep = int(match.group(1)) == final_bumat[stem]
            elif action == "checkpoint":
                step = int(match.group(1))
                keep = step % checkpoint_every == 0 or step == last_step
            else:
                keep = action == "keep"
            result[name] = (kind, keep)
            break
        else:
            result[name] = ("unknown", True)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("directory")
    parser.add_argument("--tar", default=None, help="write kept files to this .tar.gz")
    parser.add_argument("--checkpoint-every", type=int, default=10)
    args = parser.parse_args(argv)
    files = classify(args.directory, args.checkpoint_every)
    totals = defaultdict(lambda: [0, 0, 0, 0])  # kept count, kept bytes, dropped count, dropped bytes
    for name, (kind, keep) in files.items():
        size = os.path.getsize(os.path.join(args.directory, name))
        entry = totals[kind]
        entry[0 if keep else 2] += 1
        entry[1 if keep else 3] += size
    print(f"{'kind':28s} {'kept':>7s} {'kept GB':>9s} {'dropped':>8s} {'dropped GB':>11s}")
    for kind, (kc, kb, dc, db) in sorted(totals.items()):
        print(f"{kind:28s} {kc:7d} {kb / 1e9:9.3f} {dc:8d} {db / 1e9:11.3f}")
    kept = sum(t[1] for t in totals.values())
    dropped = sum(t[3] for t in totals.values())
    print(f"{'total':28s} {'':7s} {kept / 1e9:9.3f} {'':8s} {dropped / 1e9:11.3f}")
    if args.tar:
        base = os.path.basename(os.path.normpath(args.directory))
        with tarfile.open(args.tar, "w:gz") as tar:
            for name, (_, keep) in files.items():
                if keep:
                    tar.add(os.path.join(args.directory, name), arcname=f"{base}/{name}")
        print(f"wrote {args.tar}")


if __name__ == "__main__":
    sys.exit(main())
