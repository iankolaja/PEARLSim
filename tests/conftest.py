import os
import random
import sys
import types

import numpy as np
import pytest

os.environ.setdefault("MPLBACKEND", "Agg")

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
FIXTURES = os.path.join(REPO_ROOT, "tests", "fixtures")
GOLDEN = os.path.join(REPO_ROOT, "tests", "golden")
FAKE_SERPENT = os.path.join(REPO_ROOT, "tests", "fake_serpent", "fake_sss2.py")


def _stub_module(name, **attributes):
    """Register a placeholder module so legacy imports resolve without heavy dependencies."""
    if name in sys.modules:
        return
    module = types.ModuleType(name)
    module.__dict__.update(attributes)
    module.__pearlsim_test_stub__ = True
    sys.modules[name] = module


# v1 simulation.py and pebble_model.py import ml_model_wrappers, which was never
# committed. The stub goes away when the pebble-wise ML code is removed
# (refactor plan, Phase 0.5 commit 11).
_stub_module("pearlsim.ml_model_wrappers", SingleModelWrapper=object, XSModelWrapper=object)


def pytest_addoption(parser):
    parser.addoption("--update-golden", action="store_true", default=False,
                     help="rewrite golden files in tests/golden from the current behaviour")


@pytest.fixture(autouse=True)
def isolated_cwd(tmp_path, monkeypatch):
    """Run every test in its own temporary directory (v1 code chdirs and writes to cwd)."""
    monkeypatch.chdir(tmp_path)
    return tmp_path


@pytest.fixture
def repo_root():
    return REPO_ROOT


@pytest.fixture
def fixtures_dir():
    return FIXTURES


@pytest.fixture
def equilibrium_dir():
    return os.path.join(FIXTURES, "equilibrium_step100")


@pytest.fixture
def seed_all():
    def _seed(n):
        random.seed(n)
        np.random.seed(n)
    return _seed


@pytest.fixture
def fake_serpent_command():
    return f"{sys.executable} {FAKE_SERPENT}"


@pytest.fixture
def golden(request):
    """Compare text with tests/golden/<name>; with --update-golden, rewrite it instead."""
    update = request.config.getoption("--update-golden")

    def _check(name, text):
        path = os.path.join(GOLDEN, name)
        if update:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "w") as f:
                f.write(text)
            return
        assert os.path.exists(path), f"missing golden file {name}; run pytest --update-golden"
        with open(path) as f:
            expected = f.read()
        assert text == expected, f"output differs from golden file {name}"
    return _check
