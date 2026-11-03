"""Step checkpoints: resume a run at its last finished step instead of replaying every finished step.

After each step of an ``insertion ... 2`` line (reuse finished Serpent results), the whole simulation state is
pickled with its position in the deck: a hash of the deck lines read so far and the number of steps of the
current insertion line that are done. A rerun of the same deck with the same pearlsim source restores that state
and carries on from the next step. The checkpoint is ignored, and the run replays its finished steps as before,
when any of these holds:

* the deck lines up to that position, the pearlsim source, or the Python / numpy / pandas versions differ from
  when it was written;
* a finished step's record or final .bumat is missing, or a .bumat is newer than the checkpoint (steps rerun);
* a line before that position runs steps in another way (transport, load_from_step, restart, or an insertion
  line whose Serpent flag is not 2), since replaying it would not merely reuse results.

Settings that belong to the job (cores, nodes, debug level, Serpent command) are kept from the new job, and the
cached run provenance (host, Slurm job) is recomputed.

Why: on savio4_htc at low priority the family-A run-ins were preempted ~17 times a day, and each restart replayed
every finished step (~5 s each, ~14 min at step 160) before the interrupted step started again (2026-11-02).
"""

import glob
import hashlib
import logging
import os
import pickle
import platform
import random
import time

import numpy as np
import pandas as pd

from pearlsim import records, replicas
from pearlsim.nuclear_data import PROFILES

logger = logging.getLogger(__name__)

FORMAT = 1
FILE_PREFIX = "pearlsim_checkpoint"
# Simulation attributes that come from the job (command line) or are caches, never from a checkpoint
JOB_ATTRIBUTES = ("cpu_cores", "num_nodes", "debug", "serpent_command", "_provenance", "_deck", "_resume_steps",
                  "resumed_from")
# deck keywords that run or load steps other than by reusing finished insertion steps
_OTHER_STEP_KEYWORDS = ("transport", "load_from_step", "restart")


def file_name(simulation_name):
    return f"{FILE_PREFIX}_{simulation_name}.pkl"


_FINGERPRINT = []


def code_fingerprint():
    """sha256 of the pearlsim package's top-level sources, read once per process: it labels the code this process
    runs (loaded at its start), not edits made on disk while it runs."""
    if not _FINGERPRINT:
        _FINGERPRINT.append(_source_hash())
    return _FINGERPRINT[0]


def _source_hash():
    digest = hashlib.sha256()
    package = os.path.dirname(os.path.abspath(__file__))
    for path in sorted(glob.glob(os.path.join(package, "*.py"))):
        digest.update(os.path.basename(path).encode())
        with open(path, "rb") as f:
            digest.update(f.read())
    return digest.hexdigest()


def library_versions():
    return {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__}


def deck_hash(lines):
    return hashlib.sha256("".join(lines).encode()).hexdigest()


def _tokens(line):
    return line.split("#", 1)[0].split()


def resumable_prefix(lines, index):
    """Whether lines[:index] only configure the run or reuse finished steps, and lines[index] is 'insertion N x 2'."""
    for line in lines[:index]:
        tokens = _tokens(line)
        if tokens and (tokens[0] in _OTHER_STEP_KEYWORDS or tokens[0] == "insertion" and tokens[3:4] != ["2"]):
            return False
    tokens = _tokens(lines[index]) if index < len(lines) else []
    return tokens[:1] == ["insertion"] and tokens[3:4] == ["2"]


def _final_bumats(simulation, iteration):
    last_step = len(simulation.core.substep_schema)
    return [f"{replicas.stem(simulation.simulation_name, iteration, r)}.bumat{last_step}"
            for r in replicas.replica_ids(simulation.replicas)]


def save(simulation, lines, index, steps_done):
    """Write the state after ``steps_done`` steps of the insertion line ``lines[index]`` (atomically)."""
    start = time.time()
    state = {key: value for key, value in simulation.__dict__.items() if key not in JOB_ATTRIBUTES}
    payload = {"format": FORMAT, "code": code_fingerprint(), "versions": library_versions(),
               "deck": deck_hash(lines[:index + 1]),
               "line": index, "steps_done": steps_done, "iteration": simulation.core.iteration,
               "random": random.getstate(), "np_random": np.random.get_state(), "state": state}
    name = file_name(simulation.simulation_name)
    with open(name + ".part", "wb") as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(name + ".part", name)
    logger.info("checkpoint after step %d (%.0f MB, %.1f s)", simulation.core.iteration - 1,
                os.path.getsize(name) / 1e6, time.time() - start)


def _reason_to_ignore(payload, simulation, lines, written):
    if payload.get("format") != FORMAT:
        return "format"
    if payload["code"] != code_fingerprint():
        return "pearlsim source changed"
    if payload["versions"] != library_versions():
        return f"library versions changed ({payload['versions']} -> {library_versions()})"
    index = payload["line"]
    if index >= len(lines) or payload["deck"] != deck_hash(lines[:index + 1]):
        return "deck changed"
    if not resumable_prefix(lines, index):
        return "deck runs steps other than by reusing them"
    state = payload["state"]
    probe = type(simulation).__new__(type(simulation))
    probe.__dict__.update(state)
    for iteration in range(1, payload["iteration"]):
        if not os.path.isfile(records.record_file(simulation.simulation_name, iteration)):
            return f"record of step {iteration} missing"
        for bumat in _final_bumats(probe, iteration):
            if not os.path.isfile(bumat):
                return f"{bumat} missing"
            if os.path.getmtime(bumat) > written:
                return f"{bumat} newer than the checkpoint"
    return None


def restore(simulation, lines):
    """Restore the checkpoint in the current directory if it is valid for ``lines``.

    Returns (line index, steps done of that line) to resume from, or None (no usable checkpoint: replay).
    """
    name = file_name(simulation.simulation_name)
    if not os.path.isfile(name):
        return None
    written = os.path.getmtime(name)
    try:
        with open(name, "rb") as f:
            payload = pickle.load(f)
    except Exception as error:  # a truncated or foreign file: replay instead
        logger.warning("checkpoint %s unreadable (%s); replaying finished steps", name, error)
        return None
    reason = _reason_to_ignore(payload, simulation, lines, written)
    if reason:
        logger.warning("checkpoint %s not used (%s); replaying finished steps", name, reason)
        return None
    simulation.__dict__.update(payload["state"])
    core = simulation.core
    # the profile is compared by identity (e.g. 'is ENDF7_V1'); unpickling made a copy
    if PROFILES.get(core.nuclear_data.name) == core.nuclear_data:
        core.nuclear_data = PROFILES[core.nuclear_data.name]
    random.setstate(payload["random"])
    np.random.set_state(payload["np_random"])
    logger.info("resuming from checkpoint after step %d (deck line %d, %d steps of it done)",
                payload["iteration"] - 1, payload["line"] + 1, payload["steps_done"])
    return payload["line"], payload["steps_done"]
