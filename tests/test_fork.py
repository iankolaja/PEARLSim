"""Branch runs (fork_from, decision 5a): a branch continues exactly as its parent would have, records its lineage,
and the data loader gives it the parent's steps as context without counting them twice."""

import json
import os
import re
import sys

import numpy as np
import pandas as pd
import pytest
from conftest import FAKE_SERPENT
from test_checkpoint import DECK, outputs, run

from pearlsim import checkpoint, gather
from pearlsim.ml.data import Sequence, attach_parents, load_gathered_table
from pearlsim.ml.forecaster import TokenScaler, _chunks, make_tokens
from pearlsim.ml.windows import TargetSpec, make_windows
from pearlsim.simulation import DeckError, Simulation

PARENT_DECK = DECK[:DECK.index("set_power 1.0E+06")] + ["set_checkpoint_archive 2"] + DECK[DECK.index("set_power 1.0E+06"):]
# the parent: 3 steps at 1 MW, then 4 at 2 MW; after step 4 one step of the second insertion line is done
REST_OF_PARENT = ["insertion 3 off 2 graphpeb 0.5 reinsert 1.0 fuel20 1.0"]


def branch(case, fake_serpent_command, monkeypatch, lines, name="branch", fail_step=None):
    os.chdir(case)
    (case / f"{name}.inp").write_text("\n".join(lines) + "\n")
    monkeypatch.setenv("FAKE_SERPENT_SEED_NOISE", "1")
    if fail_step is None:
        monkeypatch.delenv("FAKE_SERPENT_FAIL_DECKS", raising=False)
    else:
        monkeypatch.setenv("FAKE_SERPENT_FAIL_DECKS", f"synthetic_{fail_step}_")
    sim = Simulation("synthetic")
    sim.serpent_command = fake_serpent_command
    sim.cpu_cores = 2
    sim.read_input_file(f"{name}.inp", directory_name=name)
    return sim


@pytest.fixture(scope="module")
def parent_case(tmp_path_factory):
    """One parent run shared by the tests below (the fake-Serpent run takes ~1 min)."""
    case = tmp_path_factory.mktemp("fork")
    with pytest.MonkeyPatch.context() as patch:
        sim = run(case, f"{sys.executable} {FAKE_SERPENT}", patch, deck=PARENT_DECK)
    assert sim.core.iteration == 8
    return case


STEP_FILE = re.compile(r"^(?:synthetic_(?:operating_|record_|held_inventory|discharge_inventory|discard_inventory|"
                       r"replica_spread_)?|zone_map|core_materials_|pebble_positions_)(\d+)(?:_r\d+)?[._]")


def own_steps(files, first, last):
    """Output files of steps first..last, without the record provenance (which names the run's lineage)."""
    picked = {}
    for name, data in files.items():
        match = STEP_FILE.match(name)
        if match is None or not first <= int(match.group(1)) <= last:
            continue
        if "_record_" in name:
            content = json.loads(data)
            content.pop("provenance")
            data = json.dumps(content, sort_keys=True).encode()
        picked[name] = data
    return picked


def test_archive_keeps_every_nth_checkpoint(parent_case):
    kept = sorted(p.name for p in (parent_case / "run").glob(checkpoint.archive_name("synthetic", "*")))
    assert kept == [checkpoint.archive_name("synthetic", s) for s in (2, 4, 6)]


def test_branch_with_the_parents_controls_reproduces_the_parent(parent_case, fake_serpent_command, monkeypatch):
    sim = branch(parent_case, fake_serpent_command, monkeypatch, ["fork_from run 4", *REST_OF_PARENT])
    assert sim.core.iteration == 8 and sim.fork["step"] == 4
    assert sim.fork["parent"] == str(parent_case / "run")
    parent, child = outputs(parent_case / "run"), outputs(parent_case / "branch")
    expected, got = own_steps(parent, 5, 7), own_steps(child, 5, 7)
    assert sorted(got) == sorted(expected) and len(got) > 20
    assert [name for name in got if got[name] != expected[name]] == []
    assert not any(own_steps(child, 1, 4))  # the branch writes no steps of its own before the fork
    record = json.loads((parent_case / "branch" / "synthetic_record_5.json").read_text())
    assert record["provenance"]["fork"]["step"] == 4
    assert record["provenance"]["fork"]["parent"] == str(parent_case / "run")


def test_branch_with_its_own_controls_diverges_and_resumes(parent_case, fake_serpent_command, monkeypatch):
    lines = ["fork_from run 4", "set_rod_depth 250", REST_OF_PARENT[0]]
    with pytest.raises(Exception, match="exited with status"):
        branch(parent_case, fake_serpent_command, monkeypatch, lines, name="hot", fail_step=6)
    sim = branch(parent_case, fake_serpent_command, monkeypatch, lines, name="hot")
    assert sim.resumed_from == 5 and sim.core.iteration == 8
    rods = {d: [json.loads((parent_case / d / f"synthetic_operating_{s}.json").read_text())["control_rod_position"]
                for s in (5, 6, 7)] for d in ("hot", "run")}
    assert rods == {"hot": [250.0] * 3, "run": [300.0] * 3}


def test_fork_from_must_come_first_and_name_a_kept_step(parent_case, fake_serpent_command, monkeypatch):
    with pytest.raises(DeckError, match="first keyword line"):
        branch(parent_case, fake_serpent_command, monkeypatch, ["set_power 1.0E+06", "fork_from run 4"], name="late")
    with pytest.raises(DeckError, match="no kept checkpoint .*step2"):
        branch(parent_case, fake_serpent_command, monkeypatch, ["fork_from run 5", *REST_OF_PARENT], name="odd")


def test_fork_refuses_a_parent_from_another_code_version(parent_case, fake_serpent_command, monkeypatch):
    monkeypatch.setattr(checkpoint, "code_fingerprint", lambda: "a different version")
    with pytest.raises(DeckError, match="different pearlsim source"):
        branch(parent_case, fake_serpent_command, monkeypatch, ["fork_from run 4", *REST_OF_PARENT], name="newer")


def test_gathered_branch_gets_the_parents_steps_as_context(parent_case, fake_serpent_command, monkeypatch):
    if not (parent_case / "branch" / "synthetic_record_7.json").exists():
        branch(parent_case, fake_serpent_command, monkeypatch, ["fork_from run 4", *REST_OF_PARENT])
    table = gather.gather([str(parent_case / "run"), str(parent_case / "branch")], str(parent_case / "t.csv"),
                          meshes=False)
    assert table.loc[table.sequence == "branch", "fork_parent"].unique().tolist() == ["run"]
    assert table.loc[table.sequence == "run", "fork_parent"].isna().all()
    sequences = {s.name: s for s in load_gathered_table(str(parent_case / "t.csv"), family="sequence")}
    child = sequences["branch"]
    assert child.first_own_row == 4 and child.parent == "run" and child.family == sequences["run"].family
    assert child.frame["step"].tolist() == list(range(1, 8))
    assert child.frame["sequence"].tolist() == ["run"] * 4 + ["branch"] * 3
    assert child.own_frame()["step"].tolist() == [5, 6, 7]


# ---------------------------------------------------------------- data side, synthetic sequences


def family_sequences():
    """A root run of 10 steps and a branch after its step 6 with 4 own steps (frames as gathered)."""
    root = pd.DataFrame({"sequence": "root", "step": range(1, 11), "d": np.arange(10.0), "u": 1.0, "k": np.arange(10.0),
                         "fork_parent": None, "fork_step": None})
    child = pd.DataFrame({"sequence": "child", "step": range(7, 11), "d": 100 + np.arange(4.0), "u": 2.0,
                          "k": 100 + np.arange(4.0), "fork_parent": "root", "fork_step": 6})
    return attach_parents([Sequence("root", "A", root), Sequence("child", "child", child)])


def test_attach_parents_checks_its_inputs():
    root, child = family_sequences()
    assert (child.first_own_row, child.family, len(child)) == (6, "A", 10)
    orphan = pd.DataFrame({"step": [3, 4], "d": [0.0, 1.0], "fork_parent": "nobody", "fork_step": 2})
    with pytest.raises(ValueError, match="not in the data"):
        attach_parents([Sequence("orphan", "o", orphan)])
    late = pd.DataFrame({"step": [20], "d": [0.0], "fork_parent": "root", "fork_step": 19})
    with pytest.raises(ValueError, match="does not have"):
        attach_parents([Sequence("root", "A", root.frame.iloc[:10]), Sequence("late", "l", late)])


@pytest.mark.parametrize("horizon", [0, 2])
def test_windows_count_each_target_once(horizon):
    root, child = family_sequences()
    specs = [TargetSpec("k"), TargetSpec("d", "next")]
    alone = make_windows([root], ["d", "u"], specs, window=3, horizon=horizon, control_columns=["u"])
    both = make_windows([root, child], ["d", "u"], specs, window=3, horizon=horizon, control_columns=["u"])
    mine = both.subset(both.sequence == "child")
    # every unmasked target of the branch is one of its own rows (k >= 100 or d >= 100)
    assert (mine.targets[mine.mask] >= 100).all()
    own_targets = 4 + 4  # k of rows 6..9 and next d of rows 6..9 (the windows at rows 5..8)
    if horizon == 0:
        assert mine.mask.sum() == own_targets
    assert mine.mask.any(axis=(1, 2)).all()  # no window without an own target
    assert both.mask.sum() - mine.mask.sum() == alone.mask.sum()


def test_tokens_mask_the_parents_rows_in_scaling_and_chunks():
    root, child = family_sequences()
    tokens = make_tokens([root, child], ["d"], ["u"], ["k"])
    assert tokens.first_own == [0, 6]
    assert tokens.obs_own(1).tolist() == [False] * 5 + [True] * 4    # targets d of rows 1..9
    assert tokens.step_own(1).tolist() == [False] * 6 + [True] * 3   # targets k of rows 0..8
    scaler = TokenScaler.fit(tokens)
    unique_z = np.concatenate([root.frame["k"].to_numpy()[:-1], child.frame["k"].to_numpy()[6:-1]])
    assert np.isclose(scaler.z_mean[0], unique_z.mean())
    rng = np.random.default_rng(0)
    for _ in range(20):
        x, y, z, valid, obs_mask, step_mask = _chunks(tokens.subset([1]), scaler, 3, rng)
        assert obs_mask.any() and (obs_mask <= valid).all() and (step_mask <= obs_mask).all()
