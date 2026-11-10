"""Restarting a run must continue exactly as if it had not been interrupted.

A run is interrupted at a step (its Serpent replicas fail, as when a job is preempted) and rerun with the same
deck. Every output file of the resumed run must equal the uninterrupted run's, whether the restart replays the
finished steps or resumes from the step checkpoint.
"""

import gzip
import json
import os

import pytest
from helpers.synthetic import setup_lines, write_core

from pearlsim import checkpoint
from pearlsim.simulation import SerpentError, Simulation

pytestmark = pytest.mark.e2e

# what the family-A run-ins use: v2 data with thermal coupling, replicas, the mix controller with dither,
# the power cap, and several insertion blocks with a power change between them; reuse flag 2
DECK = ["set_dataset_version test", "set_seed 7", "set_averaging_mode burnup", *setup_lines(),
        "set_nuclear_data endf8_lib80x", "set_thermal_coupling on", "set pop 1000 10 5", "set printm 1 1.0E-41",
        "set_replicas 2 concurrent", "initialize 0 graphpeb 0.5 fuel20 0.5",
        "set_discharge_threshold 18.63", "set_threshold_rel_std 0.02288", "set_burnup_time_step 6.525",
        "set_rod_depth 300", "set_power_per_pebble_cap 2.0",
        "set_k_target 1.02 mode mix sensitivity 1.0 max_step 0.1 dither 0.02",
        "set_power 1.0E+06",
        "insertion 3 off 2 graphpeb 0.5 reinsert 1.0 fuel20 1.0",
        "set_power 2.0E+06",
        "insertion 4 off 2 graphpeb 0.5 reinsert 1.0 fuel20 1.0"]
STEPS = 7
# files whose content legitimately differs between runs
VOLATILE = (".log", ".seed", ".wrn", ".out")


def run(case, fake_serpent_command, monkeypatch, fail_step=None, deck=DECK):
    os.chdir(case)
    write_core(case)
    (case / "case.inp").write_text("\n".join(deck) + "\n")
    monkeypatch.setenv("FAKE_SERPENT_SEED_NOISE", "1")
    if fail_step is None:
        monkeypatch.delenv("FAKE_SERPENT_FAIL_DECKS", raising=False)
    else:
        monkeypatch.setenv("FAKE_SERPENT_FAIL_DECKS", f"synthetic_{fail_step}_")
    sim = Simulation("synthetic")
    sim.serpent_command = fake_serpent_command
    sim.cpu_cores = 2
    sim.read_input_file("case.inp", directory_name="run")
    return sim


def outputs(directory):
    files = {}
    for name in sorted(os.listdir(directory)):
        path = directory / name
        if path.is_dir() or name.endswith(VOLATILE) or name.startswith(checkpoint.FILE_PREFIX):
            continue
        data = path.read_bytes()
        files[name] = gzip.decompress(data) if name.endswith(".gz") else data  # gzip headers hold a time
    return files


def interrupted(case, fake_serpent_command, monkeypatch, at, before_resume=None):
    with pytest.raises(SerpentError):
        run(case, fake_serpent_command, monkeypatch, fail_step=at)
    if before_resume:
        before_resume(case / "run")
    return run(case, fake_serpent_command, monkeypatch)


@pytest.fixture
def reference(tmp_path, fake_serpent_command, monkeypatch):
    case = tmp_path / "reference"
    case.mkdir()
    sim = run(case, fake_serpent_command, monkeypatch)
    assert sim.core.iteration == STEPS + 1
    return sim, outputs(case / "run")


def replay_normalised(files):
    """Replaying rewrites the finished steps' records and inventories, with two known differences that change
    no number: a reused step's record lists the 'set' options read back from its deck (template ones such as
    bc and nfg included), and discharged graphite loses its zero-valued 5011 entry."""
    out = {}
    for name, data in files.items():
        if name.endswith(".json") and ("_record_" in name or "_discharge_inventory" in name):
            content = json.loads(data)
            content.get("provenance", {}).pop("serpent_settings", None)
            for value in content.values():
                if isinstance(value, dict) and "concentration" in value:
                    value["concentration"] = {k: v for k, v in value["concentration"].items() if v != 0}
            data = json.dumps(content, sort_keys=True).encode()
        elif name.startswith("core_materials_"):
            content = json.loads(data)
            data = json.dumps({m: {k: v for k, v in c.items() if v != 0} for m, c in content.items()},
                              sort_keys=True).encode()
        out[name] = data
    return out


def assert_same_run(sim, case, reference, normalise=lambda files: files):
    ref_sim, ref_files = reference
    files, ref_files = normalise(outputs(case / "run")), normalise(ref_files)
    assert sorted(files) == sorted(ref_files)
    different = [name for name in files if files[name] != ref_files[name]]
    assert different == []
    assert sim.core.iteration == ref_sim.core.iteration and sim.days == ref_sim.days
    assert sim.controlled_graphite == ref_sim.controlled_graphite and sim.power == ref_sim.power


@pytest.mark.parametrize("at", [2, 4, 5, 7])  # inside the first block, at a block boundary, inside the second
def test_resumed_run_equals_uninterrupted_run(tmp_path, fake_serpent_command, monkeypatch, reference, at):
    case = tmp_path / "resumed"
    case.mkdir()
    sim = interrupted(case, fake_serpent_command, monkeypatch, at)
    assert sim.resumed_from == (at - 1 if at > 1 else None)
    assert_same_run(sim, case, reference)


def test_replay_without_checkpoint_equals_uninterrupted_run(tmp_path, fake_serpent_command, monkeypatch, reference):
    case = tmp_path / "replayed"
    case.mkdir()
    remove = lambda run_dir: [p.unlink() for p in run_dir.glob(checkpoint.FILE_PREFIX + "*")]  # noqa: E731
    sim = interrupted(case, fake_serpent_command, monkeypatch, 5, before_resume=remove)
    assert sim.resumed_from is None
    assert_same_run(sim, case, reference, normalise=replay_normalised)


def test_checkpoint_is_ignored_when_the_deck_changed(tmp_path, fake_serpent_command, monkeypatch):
    """The checkpoint holds the state after the deck lines it was written at; any change replays instead."""
    case = tmp_path / "changed"
    case.mkdir()
    with pytest.raises(SerpentError):
        run(case, fake_serpent_command, monkeypatch, fail_step=4)
    deck = [line.replace("set pop 1000 10 5", "set pop 2000 10 5") for line in DECK]
    sim = run(case, fake_serpent_command, monkeypatch, deck=deck)
    assert sim.resumed_from is None and sim.core.iteration == STEPS + 1


def test_checkpoint_is_ignored_when_its_step_outputs_are_missing(tmp_path, fake_serpent_command, monkeypatch,
                                                                  reference):
    """Deleting finished steps to rerun them must not be skipped over by a newer checkpoint."""
    case = tmp_path / "deleted"
    case.mkdir()

    def delete_step_3(run_dir):
        for p in run_dir.glob("synthetic_3_r*.serpent.bumat*"):
            p.unlink()

    sim = interrupted(case, fake_serpent_command, monkeypatch, 5, before_resume=delete_step_3)
    assert sim.resumed_from is None
    assert_same_run(sim, case, reference, normalise=replay_normalised)


def test_step_interrupted_before_its_bookkeeping_is_repaired(tmp_path, fake_serpent_command, monkeypatch, reference):
    """Preempted after a step's last .bumat but before its zone map and materials were saved (A1 step 207): the
    restart must write them again, and a checkpoint must not skip over the gap."""
    case = tmp_path / "gap"
    case.mkdir()

    def lose_bookkeeping(run_dir):
        for name in ("zone_map3.json", "core_materials_3.json.gz", "synthetic_record_3.json"):
            (run_dir / name).unlink()

    sim = interrupted(case, fake_serpent_command, monkeypatch, 5, before_resume=lose_bookkeeping)
    assert sim.resumed_from is None
    assert_same_run(sim, case, reference, normalise=replay_normalised)
    assert json.loads((case / "run" / "synthetic_record_3.json").read_text())["inventory"]["fuel_pebbles"] > 0
    assert not list((case / "run").glob("part_*"))


def test_checkpoint_is_ignored_when_the_code_changed(tmp_path, fake_serpent_command, monkeypatch, reference):
    case = tmp_path / "code"
    case.mkdir()
    with pytest.raises(SerpentError):
        run(case, fake_serpent_command, monkeypatch, fail_step=5)
    monkeypatch.setattr(checkpoint, "code_fingerprint", lambda: "a different version")
    sim = run(case, fake_serpent_command, monkeypatch)
    assert sim.resumed_from is None
    assert_same_run(sim, case, reference, normalise=replay_normalised)


def test_command_line_settings_are_kept_on_resume(tmp_path, fake_serpent_command, monkeypatch):
    """Cores, nodes and debug level come from the job, not from the checkpoint."""
    case = tmp_path / "cores"
    case.mkdir()
    with pytest.raises(SerpentError):
        run(case, fake_serpent_command, monkeypatch, fail_step=3)
    os.chdir(case)
    monkeypatch.delenv("FAKE_SERPENT_FAIL_DECKS")
    sim = Simulation("synthetic")
    sim.serpent_command = fake_serpent_command
    sim.cpu_cores = 4
    sim.read_input_file("case.inp", directory_name="run")
    assert sim.resumed_from == 2 and sim.cpu_cores == 4
