"""Characterization tests for pearlsim.core.Core bookkeeping on the synthetic core."""

import json

import numpy as np
import pytest
from helpers.synthetic import build_core

from pearlsim.core import get_zone
from pearlsim.material import Material
from pearlsim.zone import Out_Of_Core_Bin

pytestmark = pytest.mark.characterization

INSERT = [("graphpeb", 0.5), ("reinsert", 1.0), ("fuel20", 1.0)]
THRESHOLD, REL_STD = 18.63, 0.02288


@pytest.fixture
def core(tmp_path):
    core = build_core(tmp_path / "case")
    for channel in core.zones:
        for zone in channel:
            zone.initialize({"graphpeb": 0.5, "fuel20": 0.5})
            core.initialize_materials(zone.inventory)
    return core


def zone_totals(core):
    return [[sum(z.inventory.values()) for z in channel] for channel in core.zones]


def count(core, predicate):
    return sum(n for channel in core.zones for z in channel for k, n in z.inventory.items() if predicate(k))


def test_get_zone_maps_radius_and_height_to_indices(core):
    assert get_zone(2.5, 5.0, core.radial_bound_points, core.axial_zone_bounds) == (0, 0)
    assert get_zone(7.5, 25.0, core.radial_bound_points, core.axial_zone_bounds) == (1, 2)


def test_synthetic_core_layout(core):
    assert core.num_r == 2
    assert [[z.num_pebbles for z in ch] for ch in core.zones] == [[18, 18, 18], [18, 18, 18]]
    assert core.num_top_zone_pebbles == 36
    assert len(core.pebble_locations) == 108


@pytest.mark.xfail(strict=True, reason="review #23: define_zones appends num_z twice")
def test_num_z_has_one_entry_per_channel(core):
    assert len(core.num_z) == core.num_r


def test_insert_conserves_zone_occupancy_over_several_steps(core):
    for step in range(1, 5):
        core.iteration = step
        core.insert(INSERT, THRESHOLD, REL_STD)
        assert zone_totals(core) == [[18, 18, 18], [18, 18, 18]]
        assert count(core, lambda k: True) == 108


def test_insert_writes_discharge_and_discard_records(core, tmp_path):
    core.insert(INSERT, THRESHOLD, REL_STD)
    discharge = json.loads((tmp_path / "synthetic_discharge_inventory1.json").read_text())
    discard = json.loads((tmp_path / "synthetic_discard_inventory1.json").read_text())
    assert sorted(discharge) == ["discharge_fuel20_R1Z3G1", "discharge_fuel20_R2Z3G1",
                                 "discharge_graphpeb_R1Z3G1", "discharge_graphpeb_R2Z3G1"]
    assert discharge["discharge_fuel20_R1Z3G1"]["count"] == 9
    assert discharge["discharge_fuel20_R1Z3G1"]["FIMA"] == 0.0
    assert discharge["discharge_graphpeb_R1Z3G1"]["FIMA"] == 100.0  # graphite: no actinides
    assert discard == {}


def test_fuel_moves_up_one_zone_per_insert(core):
    core.insert(INSERT, THRESHOLD, REL_STD)
    assert core.zones[0][1].inventory == {"graphpeb_R1Z2G1": 9, "fuel20_R1Z2G1": 9}
    # a discharge is reinserted on the *next* insert, so step 1 fills the bottom with fresh fuel
    assert core.zones[0][0].inventory == {"graphpeb_R1Z1G1": 9, "fuel20_R1Z1G1": 9}
    core.iteration = 2
    core.insert(INSERT, THRESHOLD, REL_STD)
    # step 2: the pass-averaged step-1 discharge comes back as group 2
    assert core.zones[0][0].inventory == {"graphpeb_R1Z1G1": 9, "fuel20_R1Z1G2": 9}


def _reinserted_after_graphite_increase(core):
    core.iteration = 1
    core.insert(INSERT, THRESHOLD, REL_STD)  # discharges 18 fresh fuel pebbles, none discarded
    core.iteration = 2
    core.insert([("graphpeb", 0.8), ("reinsert", 1.0), ("fuel20", 1.0)], THRESHOLD, REL_STD)
    return sum(core.zones[r][0].inventory.get(f"fuel20_R{r + 1}Z1G2", 0) for r in range(core.num_r))


def test_graphite_increase_truncates_reinsertion(core):
    """v1 behaviour (review #7): only 8 of 18 recirculating fuel pebbles are reinserted."""
    assert _reinserted_after_graphite_increase(core) == 8


@pytest.mark.xfail(strict=True, reason="review #7: reinsertion overflow is truncated, not conserved")
def test_graphite_increase_conserves_recirculating_fuel(core):
    assert _reinserted_after_graphite_increase(core) == 18


def test_burnup_averaging_with_only_graphite_yields_empty_inventory(core):
    """Review #24 retracted: with no fuel discharged, averaging returns an empty inventory."""
    bin_ = Out_Of_Core_Bin()
    bin_.inventory = {"discharge_graphpeb_R1Z3G1": 9}
    core.materials["discharge_graphpeb_R1Z3G1"] = Material("discharge_graphpeb_R1Z3G1", {"6000<lib>": 1.0})
    core.volume_average_by_burnup(bin_)
    assert bin_.inventory == {}


def test_volume_average_by_burnup_groups_by_fima(core):
    fresh = core.fresh_pebbles["fuel20"].concentrations
    bin_ = Out_Of_Core_Bin()
    for i, burned in enumerate(np.linspace(0.0, 0.2, 12)):
        name = f"discharge_fuel20_R1Z3G{i + 1}"
        conc = dict(fresh)
        conc["92235<lib>"] *= 1 - burned
        conc["92238<lib>"] *= 1 - burned
        core.materials[name] = Material(name, conc)
        bin_.inventory[name] = 10
    core.volume_average_by_burnup(bin_)
    assert sorted(bin_.inventory) == [f"avgdischarge_fuel20_G{g}" for g in sorted(range(2, 13), key=str)]
    assert sum(bin_.inventory.values()) == 120


def _discharge_with_identical_burnup(core):
    fresh = core.fresh_pebbles["fuel20"].concentrations
    bin_ = Out_Of_Core_Bin()
    for name in ("discharge_fuel20_R1Z3G1", "discharge_fuel20_R2Z3G1"):
        conc = dict(fresh)
        conc["92235<lib>"] *= 0.97
        core.materials[name] = Material(name, conc)
        bin_.inventory[name] = 9
    return bin_


def test_burnup_averaging_drops_fuel_with_identical_burnup(core):
    """v1 behaviour (review #36): a single occupied histogram bin maps to group 1, which is never kept."""
    bin_ = _discharge_with_identical_burnup(core)
    core.volume_average_by_burnup(bin_)
    assert bin_.inventory == {}


@pytest.mark.xfail(strict=True, reason="review #36: identical-burnup discharge is dropped by burnup averaging")
def test_burnup_averaging_keeps_fuel_with_identical_burnup(core):
    bin_ = _discharge_with_identical_burnup(core)
    core.volume_average_by_burnup(bin_)
    assert sum(bin_.inventory.values()) == 18


def test_material_json_gz_round_trip(core, tmp_path):
    core.save_all_materials("materials.json.gz")
    original = {k: dict(m.concentrations) for k, m in core.materials.items()}
    for m in core.materials.values():
        m.concentrations = {}
    core.load_all_materials("materials.json.gz")
    assert {k: m.concentrations for k, m in core.materials.items()} == original


@pytest.mark.xfail(strict=True, reason="review #34: non-gzip branch calls json.loads on a file object")
def test_material_json_round_trip(core):
    core.save_all_materials("materials.json")
    core.load_all_materials("materials.json")
