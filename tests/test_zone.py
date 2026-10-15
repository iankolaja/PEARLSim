"""Characterization tests for pearlsim.zone (Zone, Out_Of_Core_Bin) as of v1-as-run."""

import pytest

from pearlsim.material import Material
from pearlsim.zone import Out_Of_Core_Bin, Zone

pytestmark = pytest.mark.characterization


def make_zone(n=18, r=1, z=1):
    zone = Zone(1, r, z)
    zone.pebble_locations = [(float(i), 0.0, 0.0, float(i)) for i in range(n)]
    return zone


def test_initialize_splits_by_fraction_and_names_group_1():
    zone = make_zone(18)
    zone.initialize({"graphpeb": 0.5, "fuel20": 0.5})
    assert zone.num_pebbles == 18
    assert zone.inventory == {"graphpeb_R1Z1G1": 9, "fuel20_R1Z1G1": 9}


def test_initialize_fills_rounding_shortfall(seed_all):
    seed_all(0)
    zone = make_zone(10)
    zone.initialize({"graphpeb": 0.33, "fuel20": 0.33})
    assert sum(zone.inventory.values()) == 10


def test_propagate_renames_to_destination_zone_and_conserves_count():
    below = make_zone(18, z=1)
    below.initialize({"graphpeb": 0.5, "fuel20": 0.5})
    above = make_zone(18, z=2)
    above.num_pebbles = 18
    rename = above.propagate_from(below)
    assert rename == {"graphpeb_R1Z1G1": "graphpeb_R1Z2G1", "fuel20_R1Z1G1": "fuel20_R1Z2G1"}
    assert sum(above.inventory.values()) == 18


def test_extract_top_zone_prefixes_discharge():
    zone = make_zone(18, z=3)
    zone.initialize({"graphpeb": 0.5, "fuel20": 0.5})
    bin_ = Out_Of_Core_Bin()
    rename = bin_.extract_top_zone(zone)
    assert rename == {"graphpeb_R1Z3G1": "discharge_graphpeb_R1Z3G1", "fuel20_R1Z3G1": "discharge_fuel20_R1Z3G1"}
    assert bin_.num_pebbles == 18


@pytest.mark.parametrize("mode, expected", [("pass", "reinsert_fuel20_R1Z3G3"), ("burnup", "reinsert_fuel20_R1Z3G2")])
def test_to_reinsert_increments_group_only_in_pass_mode(mode, expected):
    bin_ = Out_Of_Core_Bin()
    bin_.inventory = {"discharge_fuel20_R1Z3G2": 7}
    reinsert, rename = bin_.to_reinsert(mode)
    assert reinsert.inventory == {expected: 7}
    assert rename == {"discharge_fuel20_R1Z3G2": expected}


def test_remove_fractions_zeroes_graphite_without_counting_it():
    bin_ = Out_Of_Core_Bin()
    bin_.inventory = {"discharge_graphpeb_R1Z3G1": 5, "discharge_fuel20_R1Z3G1": 7}
    bin_.num_pebbles = 12
    removed = bin_.remove_fractions({}, True)
    assert removed == 0
    assert bin_.inventory == {"discharge_graphpeb_R1Z3G1": 0, "discharge_fuel20_R1Z3G1": 7}


def _materials(fresh_conc, burned_fraction):
    burned = dict(fresh_conc)
    burned["92235<lib>"] = fresh_conc["92235<lib>"] * (1 - burned_fraction)
    burned["92238<lib>"] = fresh_conc["92238<lib>"] * (1 - burned_fraction)
    return {"discharge_fuel20_R1Z3G1": Material("a", dict(fresh_conc)),
            "discharge_fuel20_R1Z3G2": Material("b", burned)}


def test_remove_threshold_uses_normal_cdf_of_fima(repo_root):
    fresh = Material("fuel20", f"{repo_root}/materials/fuel20.dat").concentrations
    initial = 0.00466953 + 0.0189728
    materials = _materials(fresh, 0.20)  # 20 %FIMA
    bin_ = Out_Of_Core_Bin()
    bin_.inventory = {"discharge_fuel20_R1Z3G1": 100, "discharge_fuel20_R1Z3G2": 100}
    bin_.num_pebbles = 200
    removed, removed_inventory = bin_.remove_threshold(18.63, 0.02288, materials, initial)
    # fresh fuel kept entirely; 20 %FIMA vs 18.63 threshold with 2.288 % spread: ~99.9 % removed
    assert bin_.inventory["discharge_fuel20_R1Z3G1"] == 100
    assert removed == 100
    assert list(removed_inventory) == ["discharge_fuel20_R1Z3G2"]
    assert removed_inventory["discharge_fuel20_R1Z3G2"]["FIMA"] == pytest.approx(20.0)
