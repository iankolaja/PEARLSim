"""Build PEARLSim's verified ENDF/B-VIII.0 Serpent cross-section index (xsdata).

    python scripts/build_endf8_xsdata.py [OUTPUT]

The shared co_nuclear index (``serpent/xsdata/endf8/endf8.xsdata``) mixes in 551
entries that point to the ENDF/B-VII.0 library (``endfb7/acedata``, including
natural carbon 6000), and maps ``grph30.41t``-``grph30.49t`` (30 % porous
graphite) to the 10 % porous files. This script keeps only entries whose data
file lies inside the ENDF/B-VIII.0 directory, points every ``grph30`` entry at
its own file, and checks that every referenced file exists. Provenance (source
index checksum, counts, corrections) is written next to the output as JSON.
See docs/nuclear_data.md (workspace) section 7.
"""

import hashlib
import json
import os
import sys

ENDF8 = "/global/home/groups/co_nuclear/serpent/xsdata/endf8"
SOURCE = os.path.join(ENDF8, "endf8.xsdata")
DEFAULT_OUTPUT = "/global/scratch/users/ikolaja/PEARLSim_dev/xsdata/endf8_pearlsim.xsdata"


def build(source=SOURCE, endf8_dir=ENDF8):
    kept, dropped, corrected = [], 0, []
    with open(source) as f:
        for line in f:
            fields = line.split()
            if len(fields) < 9:
                continue
            path = fields[8]
            if not os.path.abspath(path).startswith(endf8_dir + os.sep):
                dropped += 1
                continue
            name = fields[0]
            if name.startswith("grph30."):
                own = os.path.join(endf8_dir, "ENDF80SaB2", name)
                if path != own:
                    corrected.append((name, path, own))
                    fields[8] = own
                    line = " ".join(fields) + "\n"
            if not os.path.exists(fields[8]):
                raise FileNotFoundError(f"{name}: data file {fields[8]} does not exist")
            kept.append(line if line.endswith("\n") else line + "\n")
    return kept, dropped, corrected


def main(output=DEFAULT_OUTPUT):
    kept, dropped, corrected = build()
    os.makedirs(os.path.dirname(output), exist_ok=True)
    with open(output, "w") as f:
        f.writelines(kept)
    with open(SOURCE, "rb") as f:
        digest = hashlib.sha256(f.read()).hexdigest()
    provenance = {"source": SOURCE, "source_sha256": digest, "entries_kept": len(kept),
                  "entries_dropped_outside_endf8": dropped,
                  "grph30_paths_corrected": [{"entry": n, "was": old, "now": new} for n, old, new in corrected]}
    with open(output + ".provenance.json", "w") as f:
        json.dump(provenance, f, indent=1)
    print(f"wrote {output}: {len(kept)} entries, {dropped} dropped, {len(corrected)} grph30 paths corrected")


if __name__ == "__main__":
    main(*sys.argv[1:])
