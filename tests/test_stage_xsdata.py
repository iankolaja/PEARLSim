"""scripts/stage_xsdata.py: copy the selected cross-section files and rewrite the index."""

import os
import sys

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(REPO, "scripts"))

import stage_xsdata  # noqa: E402


def make_library(tmp_path):
    lib = tmp_path / "lib"
    (lib / "H").mkdir(parents=True)
    (lib / "SaB").mkdir()
    files = {"H/1001.801nc": b"h600", "H/1001.802nc": b"h900", "H/1001.803nc": b"h1200", "SaB/grph30.45t": b"g800",
             "SaB/lwtr.40t": b"w"}
    for rel, data in files.items():
        (lib / rel).write_bytes(data)
    lines = [f"   1001.01c   1001.01c  1   1001  0    1.007825  600  0  {lib}/H/1001.801nc",
             f"    H-1.01c   1001.01c  1   1001  0    1.007825  600  0  {lib}/H/1001.801nc",
             f"   1001.02c   1001.02c  1   1001  0    1.007825  900  0  {lib}/H/1001.802nc",
             f"   1001.03c   1001.03c  1   1001  0    1.007825 1200  0  {lib}/H/1001.803nc",
             f" grph30.45t grph30.45t  3      0  0   11.996708  800  0  {lib}/SaB/grph30.45t",
             f"   lwtr.40t   lwtr.40t  3      0  0    1.007825  300  0  {lib}/SaB/lwtr.40t"]
    index = tmp_path / "lib.xsdata"
    index.write_text("\n".join(lines) + "\n")
    return index


def test_stages_selected_entries_and_keeps_the_rest(tmp_path):
    index = make_library(tmp_path)
    out = stage_xsdata.stage(str(index), str(tmp_path / "local"), [".01c", ".02c"], ["grph30"], 0.0,
                             log=lambda m: None)
    assert out == str(tmp_path / "local" / "lib.xsdata")
    entries = {line.split()[0]: line.split()[-1] for line in open(out)}
    for alias in ("1001.01c", "H-1.01c", "1001.02c", "grph30.45t"):
        assert entries[alias].startswith(str(tmp_path / "local")), alias
        original = [line.split()[-1] for line in open(index) if line.split()[0] == alias][0]
        assert open(entries[alias], "rb").read() == open(original, "rb").read()
    assert entries["1001.03c"].startswith(str(tmp_path / "lib"))  # not selected: original path
    assert entries["lwtr.40t"].startswith(str(tmp_path / "lib"))
    assert len(open(out).read().splitlines()) == len(open(index).read().splitlines())


def test_falls_back_to_the_shared_index_without_space(tmp_path):
    index = make_library(tmp_path)
    out = stage_xsdata.stage(str(index), str(tmp_path / "local"), [".01c"], [], 1e9, log=lambda m: None)
    assert out == str(index) and not (tmp_path / "local" / "lib.xsdata").exists()
