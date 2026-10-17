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


def test_num_z_has_one_entry_per_channel(core):
    """Review #23 (fixed)."""
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


def _recirculating_fuel(core):
    """Group >= 2 fuel pebbles at the bottom of every channel, and held out of the core."""
    placed = sum(n for r in range(core.num_r) for k, n in core.zones[r][0].inventory.items()
                 if k.startswith("fuel20") and not k.endswith("G1"))
    return placed, sum(core.held_inventory.values())


def test_graphite_increase_holds_out_overflow(core, tmp_path):
    """Review #7 (fixed): recirculating fuel that does not fit is held out, not lost."""
    core.iteration = 1
    core.insert(INSERT, THRESHOLD, REL_STD)  # discharges 18 fresh fuel pebbles, none discarded
    core.iteration = 2
    core.insert([("graphpeb", 0.8), ("reinsert", 1.0), ("fuel20", 1.0)], THRESHOLD, REL_STD)
    # 14 graphite per 18-pebble bottom zone leaves room for 4 + 4 of the 18 recirculating pebbles
    assert _recirculating_fuel(core) == (8, 10)
    assert json.loads((tmp_path / "synthetic_held_inventory2.json").read_text()) == core.held_inventory
    assert zone_totals(core) == [[18, 18, 18], [18, 18, 18]]


def test_held_pebbles_are_reinserted_first(core):
    fresh = core.fresh_pebbles["fuel20"]
    for key in ("held_avgreinsert_fuel20_G5", "avgreinsert_fuel20_G2"):
        core.materials[key] = Material(key, dict(fresh.concentrations))
    # 9 + 9 free places after graphite; the 10 held pebbles go in before the 18 new ones
    held = core.fill_bottom_zones([("graphpeb", 0.5), ("reinsert", 1.0), ("fuel20", 1.0)],
                                  [("held_avgreinsert_fuel20_G5", 10), ("avgreinsert_fuel20_G2", 18)])
    placed_g5 = sum(core.zones[r][0].inventory.get(f"fuel20_R{r + 1}Z1G5", 0) for r in range(2))
    placed_g2 = sum(core.zones[r][0].inventory.get(f"fuel20_R{r + 1}Z1G2", 0) for r in range(2))
    assert (placed_g5, placed_g2) == (10, 8)
    assert held == {"held_avgreinsert_fuel20_G2": 10}
    assert "held_avgreinsert_fuel20_G2" in core.materials


def test_overflow_is_conserved_over_steps(core):
    core.iteration = 1
    core.insert(INSERT, THRESHOLD, REL_STD)
    core.iteration = 2
    core.insert([("graphpeb", 0.8), ("reinsert", 1.0), ("fuel20", 1.0)], THRESHOLD, REL_STD)
    core.iteration = 3
    core.insert(INSERT, THRESHOLD, REL_STD)
    # 10 held + 18 newly discharged recirculating pebbles; 18 places free after graphite
    assert _recirculating_fuel(core) == (18, 10)


def test_reinsertion_split_is_exact(core):
    """Review #37 (fixed): pool counts are split between channels without rounding loss."""
    fresh = core.fresh_pebbles["fuel20"]
    pool = []
    for g, count in ((2, 5), (3, 7), (4, 1)):
        key = f"avgreinsert_fuel20_G{g}"
        core.materials[key] = Material(key, dict(fresh.concentrations))
        pool.append((key, count))
    held = core.fill_bottom_zones([("graphpeb", 0.5), ("reinsert", 1.0), ("fuel20", 1.0)], pool)
    assert held == {}
    placed = {g: sum(core.zones[r][0].inventory.get(f"fuel20_R{r + 1}Z1G{g}", 0) for r in range(2))
              for g in (2, 3, 4)}
    assert placed == {2: 5, 3: 7, 4: 1}
    assert zone_totals(core)[0][0] == zone_totals(core)[1][0] == 18


def test_same_group_sources_are_volume_averaged(core):
    fresh = core.fresh_pebbles["fuel20"].concentrations
    a = Material("held_avgreinsert_fuel20_G3", {k: v for k, v in fresh.items()})
    b = Material("avgreinsert_fuel20_G3", {k: v * 0.5 for k, v in fresh.items()})
    core.materials[a.name], core.materials[b.name] = a, b
    core.fill_bottom_zones([("graphpeb", 0.5), ("reinsert", 1.0), ("fuel20", 1.0)],
                           [(a.name, 6), (b.name, 2)])
    mixed = core.materials["fuel20_R1Z1G3"].concentrations["92235<lib>"]
    assert mixed == pytest.approx(fresh["92235<lib>"] * (6 + 2 * 0.5) / 8)


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


def test_burnup_averaging_keeps_fuel_with_identical_burnup(core):
    """Review #36 (fixed): v1 dropped all fuel when only one histogram bin was occupied."""
    bin_ = _discharge_with_identical_burnup(core)
    core.volume_average_by_burnup(bin_)
    assert bin_.inventory == {"avgdischarge_fuel20_G2": 18}


@pytest.mark.parametrize("seed", range(25))
def test_burnup_averaging_conserves_pebbles(core, seed):
    """Every discharged fuel pebble ends up in exactly one averaged group (review #36)."""
    rng = np.random.default_rng(seed)
    fresh = core.fresh_pebbles["fuel20"].concentrations
    bin_ = Out_Of_Core_Bin()
    n_materials = int(rng.integers(1, 40))
    # clustered burnups make the binning loop overshoot, which v1 truncated at fuel_groups
    centres = rng.uniform(0.0, 0.2, size=3)
    for i in range(n_materials):
        burned = float(np.clip(rng.choice(centres) + rng.normal(0, 0.01), 0.0, 0.25))
        name = f"discharge_fuel20_R1Z3G{i + 1}"
        conc = dict(fresh)
        conc["92235<lib>"] *= 1 - burned
        conc["92238<lib>"] *= 1 - burned
        core.materials[name] = Material(name, conc)
        bin_.inventory[name] = int(rng.integers(1, 50))
    total = sum(bin_.inventory.values())
    core.volume_average_by_burnup(bin_)
    assert sum(bin_.inventory.values()) == total
    groups = sorted(int(k.rsplit("G", 1)[1]) for k in bin_.inventory)
    assert groups == list(range(2, 2 + len(groups)))  # consecutive labels from 2


def test_material_json_gz_round_trip(core, tmp_path):
    core.save_all_materials("materials.json.gz")
    original = {k: dict(m.concentrations) for k, m in core.materials.items()}
    for m in core.materials.values():
        m.concentrations = {}
    core.load_all_materials("materials.json.gz")
    assert {k: m.concentrations for k, m in core.materials.items()} == original


def test_material_json_round_trip(core):
    """Review #34 (fixed)."""
    core.save_all_materials("materials.json")
    original = {k: dict(m.concentrations) for k, m in core.materials.items()}
    core.load_all_materials("materials.json")
    assert {k: m.concentrations for k, m in core.materials.items()} == original


def test_initial_heavy_metal_comes_from_fresh_fuel(core):
    fuel = core.fresh_pebbles["fuel20"]
    half = Material("fuel_half", {k: v / 2 for k, v in fuel.concentrations.items()})
    core.set_fresh_material("fuel_half", half)
    assert core.initial_actinides == pytest.approx((0.00466953 + 0.0189728) / 2)
    core.set_fresh_material("graphpeb", Material("graphpeb", {"6000<lib>": 1.0}))  # no heavy metal: unchanged
    assert core.initial_actinides == pytest.approx((0.00466953 + 0.0189728) / 2)


# ---- pass mode (averaging_mode "pass": discard after max_passes, average discharge per pass) ----

PASS_INSERT = [("reinsert", 1.0), ("fuel20", 1.0)]


def _burn_in_core(core, factor=0.98):
    """Stand-in for a depletion step: scale heavy metal in every in-core fuel material."""
    for channel in core.zones:
        for zone in channel:
            for key in zone.inventory:
                if key.startswith("fuel"):
                    conc = core.materials[key].concentrations
                    for nuc in ("92235<lib>", "92238<lib>"):
                        conc[nuc] *= factor


def test_pass_mode_discards_after_max_passes_and_conserves_pebbles(core, tmp_path):
    core.averaging_mode, core.max_passes = "pass", 3
    for channel in core.zones:  # all-fuel core
        for zone in channel:
            zone.inventory = {k.replace("graphpeb", "fuel20"): 0 for k in zone.inventory}
            zone.initialize({"fuel20": 1.0})
            core.initialize_materials(zone.inventory)
    discarded = []
    for step in range(1, 31):
        core.iteration = step
        core.insert(PASS_INSERT, THRESHOLD, REL_STD)
        _burn_in_core(core)
        groups = {int(k.rsplit("G", 1)[1]) for ch in core.zones for z in ch for k in z.inventory}
        assert groups <= {1, 2, 3}
        assert count(core, lambda k: True) == 108
        discard = json.loads((tmp_path / f"synthetic_discard_inventory{step}.json").read_text())
        assert all(k.endswith("G3") for k in discard)  # only pebbles finishing their 3rd pass
        discarded.append(sum(v["count"] for v in discard.values()))
        fresh = count(core, lambda k: k.endswith("G1") and "Z1G" in k)
        if step >= 3:  # a discharge returns one step later, so fresh fuel replaces the previous discard
            assert fresh == discarded[-2]
            assert not core.held_inventory
        fima = {k: v["FIMA"] for k, v in json.loads(
            (tmp_path / f"synthetic_discharge_inventory{step}.json").read_text()).items()}
        last, first = [v for k, v in fima.items() if k.endswith("G3")], [v for k, v in fima.items() if k.endswith("G1")]
        if last and first:  # later passes are more burned
            assert min(last) > max(first)
    assert sum(discarded) >= 108  # the initial core has been discarded (after its 3 passes)


def test_recirculation_rescaled_when_bottom_zones_are_smaller(core):
    """Review #39: with smaller bottom zones (as in gFHR_points.dat) fresh fuel keeps entering
    and nothing piles up as held inventory."""
    core.averaging_mode, core.max_passes = "pass", 3
    for channel in core.zones:
        bottom = channel[0]
        bottom.pebble_locations = bottom.pebble_locations.iloc[:12].copy()
        bottom.num_pebbles = 12
        for zone in channel:
            zone.inventory = {}
            zone.initialize({"fuel20": 1.0})
            core.initialize_materials(zone.inventory)
    fresh = []
    for step in range(1, 31):
        core.iteration = step
        core.insert(PASS_INSERT, THRESHOLD, REL_STD)
        _burn_in_core(core)
        assert not core.held_inventory  # unscaled: 36 discharged per step into 24 places
        fresh.append(count(core, lambda k: k.endswith("G1") and "Z1G" in k))
    # the initial core is discarded in waves; each bottom load is eventually replaced by fresh fuel
    assert sum(fresh[3:]) >= 2 * 24
    assert zone_totals(core) == [[12, 18, 18], [12, 18, 18]]


def test_pass_averaging_conserves_atoms(core):
    fresh = core.fresh_pebbles["fuel20"].concentrations
    bin_ = Out_Of_Core_Bin()
    expected = {}
    for r, g, n, burned in ((1, 1, 5, 0.03), (2, 1, 7, 0.05), (1, 2, 4, 0.10), (2, 2, 9, 0.12)):
        name = f"discharge_fuel20_R{r}Z3G{g}"
        conc = {k: v * (1 - burned) for k, v in fresh.items()}
        core.materials[name] = Material(name, conc)
        bin_.inventory[name] = n
        expected[g] = expected.get(g, 0.0) + n * conc["92235<lib>"]
    core.max_passes = 3
    core.volume_average_by_pass(bin_)
    assert bin_.inventory == {"avgdischarge_fuel20_G1": 12, "avgdischarge_fuel20_G2": 13}
    for g, atoms in expected.items():
        key = f"avgdischarge_fuel20_G{g}"
        assert bin_.inventory[key] * core.materials[key].concentrations["92235<lib>"] == pytest.approx(atoms)


@pytest.mark.parametrize("mode", ["pass", "burnup"])
def test_fuel_types_sharing_a_name_prefix_are_kept_apart(core, mode):
    """'fuel1' must not collect 'fuel15' pebbles (substring match in the averaging)."""
    fresh = core.fresh_pebbles["fuel20"].concentrations
    bin_ = Out_Of_Core_Bin()
    for fuel_type, scale in (("fuel1", 0.9), ("fuel15", 0.8)):
        name = f"discharge_{fuel_type}_R1Z3G1"
        core.materials[name] = Material(name, {k: v * scale for k, v in fresh.items()})
        bin_.inventory[name] = 9
    if mode == "pass":
        core.volume_average_by_pass(bin_)
    else:
        core.volume_average_by_burnup(bin_)
    by_type = {}
    for key, n in bin_.inventory.items():
        by_type[key.split("_")[1]] = by_type.get(key.split("_")[1], 0) + n
    assert by_type == {"fuel1": 9, "fuel15": 9}
    for key in bin_.inventory:
        scale = 0.9 if key.split("_")[1] == "fuel1" else 0.8
        assert core.materials[key].concentrations["92235<lib>"] == pytest.approx(fresh["92235<lib>"] * scale)
