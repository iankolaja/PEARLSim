"""Tests of the test infrastructure itself: light imports and the fake Serpent executable."""

import json
import os
import subprocess
import sys

import pytest

from conftest import FAKE_SERPENT

DECK = """\
dep daystep 1.0 2.0
mat fuel20_R1Z1G1 sum fix 09c 959 rgb 1 2 3  tmp 959 burn 1
  92235.09c    0.004
  92238.09c    0.019

mat graphpeb_R1Z1G1 -1.74 rgb 1 2 3  tmp 959 moder  grph 6000
  6000.09c    1.0
"""


def test_import_pearlsim_does_not_load_ml_frameworks():
    code = ("import sys, pearlsim; "
            "print([m for m in ('torch', 'tensorflow', 'keras') if m in sys.modules])")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "[]"


def run_fake(tmp_path, mode="ok"):
    deck = tmp_path / "case_1.serpent"
    deck.write_text(DECK)
    env = dict(os.environ, FAKE_SERPENT_MODE=mode, FAKE_SERPENT_LOG=str(tmp_path / "log.jsonl"))
    return subprocess.run([sys.executable, FAKE_SERPENT, str(deck), "-omp", "2"], env=env, cwd=tmp_path)


def test_fake_serpent_writes_outputs_per_depletion_step(tmp_path):
    assert run_fake(tmp_path).returncode == 0
    for name in ("case_1.serpent_res.m", "case_1.serpent_det1.m", "case_1.serpent.bumat0",
                 "case_1.serpent.bumat1", "case_1.serpent.bumat2"):
        assert (tmp_path / name).exists(), name
    assert not (tmp_path / "case_1.serpent.bumat3").exists()
    bumat2 = (tmp_path / "case_1.serpent.bumat2").read_text()
    assert "mat  fuel20_R1Z1G1pp1r1" in bumat2
    assert "graphpeb" not in bumat2  # only burnable materials
    log = [json.loads(line) for line in (tmp_path / "log.jsonl").read_text().splitlines()]
    assert log[0]["burnable_materials"] == 1 and log[0]["daysteps"] == 2


@pytest.mark.parametrize("mode, code, bumats", [("fail", 1, 0), ("no_output", 0, 0), ("partial", 0, 1)])
def test_fake_serpent_failure_modes(tmp_path, mode, code, bumats):
    assert run_fake(tmp_path, mode).returncode == code
    assert len(list(tmp_path.glob("case_1.serpent.bumat*"))) == bumats
