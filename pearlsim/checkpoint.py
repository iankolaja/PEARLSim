"""Step checkpoints: resume a run at its last finished step instead of replaying every finished step.

After each step of an ``insertion ... 2`` line (reuse finished Serpent results), the whole simulation state is
pickled with its position in the deck: a hash of the deck lines read so far and the number of steps of the
current insertion line that are done. A rerun of the same deck with the same pearlsim source restores that state
and carries on from the next step. The checkpoint is ignored, and the run replays its finished steps as before,
when any of these holds:

* the deck lines up to that position, the pearlsim source, or the Python / numpy / pandas versions differ from
  when it was written;
* a finished step's record, zone map, materials snapshot or final .bumat is missing (the replay writes the first
  three again), or a .bumat is newer than the checkpoint (steps rerun);
* a line before that position runs steps in another way (transport, load_from_step, restart, or an insertion
  line whose Serpent flag is not 2), since replaying it would not merely reuse results.

Settings that belong to the job (cores, nodes, debug level, Serpent command) are kept from the new job, and the
cached run provenance (host, Slurm job) is recomputed.

Every ``checkpoint_every`` steps (``set_checkpoint_archive``, default 10) the checkpoint is also kept as
``pearlsim_checkpoint_<run>_step<N>.pkl``. A branch run starts from one with ``fork_from <parent dir> <N>``
(decision 5a, docs/v2_dataset_decisions.md): it continues exactly as the parent would have after step N, under the
branch deck's own controls, and records its lineage (``Simulation.fork``) in every step record, so the data loader
can give a branch the parent's steps as context without counting them twice.

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


def archive_name(simulation_name, step):
    """The kept checkpoint of the state after ``step``."""
    return f"{FILE_PREFIX}_{simulation_name}_step{step}.pkl"


def _sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 24), b""):
            digest.update(block)
    return digest.hexdigest()


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
    """Whether lines[:index] only configure the run, start it from a kept checkpoint (fork_from) or reuse finished
    steps, and lines[index] is 'insertion N x 2'."""
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
    step = simulation.core.iteration - 1
    every = getattr(simulation, "checkpoint_every", None)
    if every and step % every == 0:
        kept = archive_name(simulation.simulation_name, step)
        if os.path.exists(kept):
            os.remove(kept)
        os.link(name, kept)  # the next save replaces 'name' by a new file, so this one keeps its content
    logger.info("checkpoint after step %d (%.0f MB, %.1f s)", step, os.path.getsize(name) / 1e6, time.time() - start)


def _first_own_step(state):
    fork = state.get("fork")
    return fork["step"] + 1 if fork else 1


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
    fork = state.get("fork")
    if fork:
        parent = os.path.join(fork["parent"], archive_name(simulation.simulation_name, fork["step"]))
        if not os.path.isfile(parent) or _sha256(parent) != fork["checkpoint_sha256"]:
            return f"the parent checkpoint {parent} is missing or changed"
    probe = type(simulation).__new__(type(simulation))
    probe.__dict__.update(state)
    for iteration in range(_first_own_step(state), payload["iteration"]):
        for name in (records.record_file(simulation.simulation_name, iteration), f"zone_map{iteration}.json",
                     f"core_materials_{iteration}.json.gz"):
            if not os.path.isfile(name):
                return f"{name} missing"
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
    _apply(simulation, payload)
    logger.info("resuming from checkpoint after step %d (deck line %d, %d steps of it done)",
                payload["iteration"] - 1, payload["line"] + 1, payload["steps_done"])
    return payload["line"], payload["steps_done"]


def _apply(simulation, payload):
    """Load a checkpoint's state into ``simulation``, keeping the job's own settings."""
    simulation.__dict__.update(payload["state"])
    core = simulation.core
    # the profile is compared by identity (e.g. 'is ENDF7_V1'); unpickling made a copy
    if PROFILES.get(core.nuclear_data.name) == core.nuclear_data:
        core.nuclear_data = PROFILES[core.nuclear_data.name]
    random.setstate(payload["random"])
    np.random.set_state(payload["np_random"])


def fork(simulation, parent_dir, step, allow_code_change=False):
    """Start ``simulation`` from the parent run's kept checkpoint after ``step`` (deck keyword fork_from).

    The parent's state must come from this pearlsim source and these library versions: a branch continues the
    parent exactly, which a different code version would not guarantee. Raises ValueError otherwise, unless
    ``allow_code_change`` (the user's explicit choice, e.g. after a bug fix that cannot affect the branch's steps):
    the branch then runs the current code and its lineage records both code versions. Library versions must
    still match.
    """
    parent_dir = os.path.abspath(parent_dir)
    path = os.path.join(parent_dir, archive_name(simulation.simulation_name, step))
    if not os.path.isfile(path):
        kept = sorted(glob.glob(os.path.join(parent_dir, archive_name(simulation.simulation_name, "*"))))
        raise ValueError(f"no kept checkpoint {path}; kept: {[os.path.basename(k) for k in kept] or 'none'}")
    with open(path, "rb") as f:
        payload = pickle.load(f)
    if payload.get("format") != FORMAT:
        raise ValueError(f"{path}: checkpoint format {payload.get('format')}, expected {FORMAT}")
    code_changed = payload["code"] != code_fingerprint()
    if code_changed and not allow_code_change:
        raise ValueError(f"{path} was written by a different pearlsim source; fork with the parent's code version "
                         "or, if the change cannot affect the branch, add allow_code_change")
    if payload["versions"] != library_versions():
        raise ValueError(f"{path}: library versions {payload['versions']} differ from {library_versions()}")
    if payload["iteration"] != step + 1:
        raise ValueError(f"{path} holds the state after step {payload['iteration'] - 1}, not {step}")
    if not os.path.isfile(os.path.join(parent_dir, records.record_file(simulation.simulation_name, step))):
        raise ValueError(f"the parent has no record of step {step} in {parent_dir}")
    _apply(simulation, payload)
    simulation.fork = {"parent": parent_dir, "step": step, "checkpoint_sha256": _sha256(path),
                       "parent_fork": payload["state"].get("fork"),
                       "code_change": {"parent_code": payload["code"], "branch_code": code_fingerprint()}
                       if code_changed else None}
    logger.info("forked from %s after step %d", parent_dir, step)
