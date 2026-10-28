"""Copy the cross-section files a run can use to node-local storage and write an index pointing at them.

    python scripts/stage_xsdata.py SRC_INDEX DEST_DIR [--suffixes .01c .02c] [--thermal grph30 c-sic si-sic]
        [--min-free-gb 40]

Prints the index to use (Serpent `set acelib`): DEST_DIR/<name of SRC_INDEX> when staged, else SRC_INDEX.
Entries of SRC_INDEX are copied when they are continuous-energy data (type 1) with one of ``--suffixes``, or
thermal-scattering tables (type 3) whose name starts with one of ``--thermal``; every other entry keeps its
original path, so a nuclide the selection missed falls back to the shared copy instead of failing. The data are
byte-identical copies; only their location changes. If DEST_DIR has less free space than the files plus
``--min-free-gb`` margin, nothing is copied and SRC_INDEX is printed.

Why: on 2026-10-28 three concurrent replicas spent ~20 min each initialising (reading these files from the
shared group directory) for ~2.7 min of transport. Savio allows a job's /tmp (per-job overlay of /local, an SSD
on savio4_htc nodes) for such I/O (Research IT "Staging Data for Computation").
"""

import argparse
import os
import shutil
import sys


def select(lines, suffixes, thermal):
    """Paths to copy, from xsdata index lines (alias ZAID type ZA isomer mass T binary path)."""
    paths = set()
    for line in lines:
        t = line.split()
        if len(t) < 9:
            continue
        alias, kind, path = t[0], t[2], t[-1]
        if (kind == "1" and alias.endswith(tuple(suffixes))) or (kind == "3" and alias.startswith(tuple(thermal))):
            paths.add(path)
    return paths


def stage(src_index, dest, suffixes, thermal, min_free_gb, log=print):
    with open(src_index) as f:
        lines = f.read().splitlines()
    paths = sorted(p for p in select(lines, suffixes, thermal) if os.path.exists(p))
    need = sum(os.path.getsize(p) for p in paths)
    os.makedirs(dest, exist_ok=True)
    free = shutil.disk_usage(dest).free
    if free < need + min_free_gb * 1e9:
        log(f"stage_xsdata: {free / 1e9:.1f} GB free in {dest}, need {need / 1e9:.1f} GB + {min_free_gb} GB; "
            f"using {src_index}")
        return src_index
    root = os.path.commonpath(paths) if paths else ""
    mapping = {}
    for p in paths:
        local = os.path.join(dest, "data", os.path.relpath(p, root))
        os.makedirs(os.path.dirname(local), exist_ok=True)
        if not (os.path.exists(local) and os.path.getsize(local) == os.path.getsize(p)):
            shutil.copyfile(p, local + ".part")
            os.replace(local + ".part", local)
        mapping[p] = local
    out = os.path.join(dest, os.path.basename(src_index))
    with open(out, "w") as f:
        for line in lines:
            t = line.split()
            if len(t) >= 9 and t[-1] in mapping:
                line = line[: line.rfind(t[-1])] + mapping[t[-1]]
            f.write(line + "\n")
    log(f"stage_xsdata: copied {len(mapping)} files ({need / 1e9:.1f} GB) to {dest}")
    return out


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("src_index")
    parser.add_argument("dest")
    parser.add_argument("--suffixes", nargs="+", default=[".01c", ".02c"])
    parser.add_argument("--thermal", nargs="+", default=["grph30", "c-sic", "si-sic"])
    parser.add_argument("--min-free-gb", type=float, default=40.0)
    args = parser.parse_args(argv)
    print(stage(args.src_index, args.dest, args.suffixes, args.thermal, args.min_free_gb,
                log=lambda m: print(m, file=sys.stderr)))


if __name__ == "__main__":
    main()
