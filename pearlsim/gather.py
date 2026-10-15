"""Gather per-step records of one or more runs into a table (replaces gather_sequence_results_*.py).

    python -m pearlsim.gather RUN_DIR [RUN_DIR ...] --out table.parquet [--run-name NAME]
                              [--first N] [--last M] [--no-meshes] [--rebuild] [--write-records]

Each run directory gives one sequence. A step's row comes from its record
(``<sim>_record_<i>.json``) when present; otherwise, as for v1 runs, the record is
built from the step's output files, so v1 data is re-exported with the v2 features.
Steps without Serpent results (bookkeeping-only steps, unfinished steps) are
skipped and listed in the metadata. ``days`` is the cumulative time at the end of
each step, from the run start when records carry it, otherwise summed from the
first gathered step (as v1 gathering did).

Output: Parquet (``.parquet``) or CSV (``.csv``/``.csv.gz``), plus ``<out>.meta.json``
listing the sources, steps, skipped steps and schema version.
"""

import argparse
import json
import logging
import os
import re
import sys

import pandas as pd

from pearlsim import records, replicas

logger = logging.getLogger(__name__)


def detect_run_name(directory):
    """Simulation name from the ``<sim>_operating_<i>.json`` files of a run directory."""
    names = {m.group(1) for m in map(re.compile(r"^(.+)_operating_(\d+)\.json$").match, os.listdir(directory)) if m}
    if len(names) != 1:
        raise ValueError(f"cannot infer the run name in {directory}: found {sorted(names) or 'none'}; use --run-name")
    return names.pop()


def available_steps(directory, run_name):
    """Steps with an operating file, and those of them that have Serpent results."""
    pattern = re.compile(rf"^{re.escape(run_name)}_operating_(\d+)\.json$")
    files = set(os.listdir(directory))
    steps = sorted(int(m.group(1)) for m in map(pattern.match, files) if m)
    with_results = [s for s in steps
                    if f"{replicas.stem(run_name, s)}_res.m" in files or f"{replicas.stem(run_name, s, 0)}_res.m" in files]
    return steps, with_results


def load_or_build(directory, run_name, step, rebuild=False, write=False):
    path = os.path.join(directory, records.record_file(run_name, step))
    if os.path.isfile(path) and not rebuild:
        with open(path) as f:
            return json.load(f)
    if write:
        return records.write_record(directory, run_name, step)
    return records.build_record(directory, run_name, step)


def gather_run(directory, run_name=None, first=None, last=None, meshes=True, rebuild=False, write=False):
    """Rows of one run, sorted by step; returns (DataFrame, metadata)."""
    run_name = run_name or detect_run_name(directory)
    steps, with_results = available_steps(directory, run_name)
    selected = [s for s in with_results if (first is None or s >= first) and (last is None or s <= last)]
    rows, days = [], 0.0
    for step in selected:
        record = load_or_build(directory, run_name, step, rebuild=rebuild, write=write)
        row = records.flatten(record, meshes=meshes)
        days += row["burnup_step"]
        row["days_since_first"] = days
        if row["days"] is None:
            row["days"] = days
        rows.append(row)
    table = pd.DataFrame(rows)
    sequence = os.path.basename(os.path.normpath(directory))
    if not table.empty:
        table.insert(0, "sequence", sequence)
    skipped = [s for s in steps if s not in with_results and (first is None or s >= first)
               and (last is None or s <= last)]
    return table, {"directory": os.path.abspath(directory), "sequence": sequence, "run_name": run_name,
                   "steps": selected, "skipped_without_results": skipped}


def gather(directories, out, **kwargs):
    tables, sources = [], []
    for directory in directories:
        table, meta = gather_run(directory, **kwargs)
        logger.info("%s: %d steps (%d skipped)", directory, len(meta["steps"]), len(meta["skipped_without_results"]))
        tables.append(table)
        sources.append(meta)
    table = pd.concat(tables, ignore_index=True) if tables else pd.DataFrame()
    if out.endswith(".parquet"):
        table.to_parquet(out, index=False)
    else:
        table.to_csv(out, index=False)
    with open(out + ".meta.json", "w") as f:
        json.dump({"record_schema_version": records.SCHEMA_VERSION, "rows": len(table),
                   "columns": len(table.columns), "sources": sources, "code": records.code_version()}, f, indent=1)
    return table


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("directories", nargs="+")
    parser.add_argument("--out", required=True, help=".parquet, .csv or .csv.gz")
    parser.add_argument("--run-name", default=None, help="simulation name (default: inferred per directory)")
    parser.add_argument("--first", type=int, default=None)
    parser.add_argument("--last", type=int, default=None)
    parser.add_argument("--no-meshes", action="store_true", help="leave out the flux and power mesh columns")
    parser.add_argument("--rebuild", action="store_true", help="rebuild records from the output files")
    parser.add_argument("--write-records", action="store_true", help="save records built from output files")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    gather(args.directories, args.out, run_name=args.run_name, first=args.first, last=args.last,
           meshes=not args.no_meshes, rebuild=args.rebuild, write=args.write_records)


if __name__ == "__main__":
    sys.exit(main())
