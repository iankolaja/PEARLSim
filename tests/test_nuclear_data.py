"""Tests of nuclear-data profiles, the power-temperature model and v2 deck rendering."""

import re

import numpy as np
import pytest
from helpers.synthetic import build_core

from pearlsim import thermal
from pearlsim.nuclear_data import (
    ENDF7_V1,
    ENDF8_LIB80X,
    CoreConditions,
    ThermRegistry,
    render_template,
    validate_deck,
)

# ------------------------------------------------------------------ profiles


@pytest.mark.parametrize("profile, temperature, suffix", [
    (ENDF7_V1, 900, ".09c"), (ENDF7_V1, 959, ".09c"), (ENDF7_V1, 823.15, ".06c"),
    (ENDF8_LIB80X, 294, ".00c"), (ENDF8_LIB80X, 823.15, ".01c"), (ENDF8_LIB80X, 953.6, ".02c"),
    (ENDF8_LIB80X, 1200, ".03c"), (ENDF8_LIB80X, 1300, ".03c"),
])
def test_suffix_is_highest_base_temperature_not_above(profile, temperature, suffix):
    assert profile.suffix(temperature) == suffix


def test_suffix_below_lowest_library_raises():
    with pytest.raises(ValueError, match="294"):
        ENDF8_LIB80X.suffix(250)


def test_natural_carbon_expansion_conserves_atoms_and_mass():
    (c12, a12, m12), (c13, a13, m13) = ENDF8_LIB80X.map_za("6000")
    assert (c12, c13) == ("6012", "6013")
    assert a12 + a13 == pytest.approx(1.0) and m12 + m13 == pytest.approx(1.0)
    assert m13 > a13  # C-13 is heavier
    assert ENDF7_V1.map_za("6000") == (("6000", 1.0, 1.0),)
    assert ENDF8_LIB80X.map_za("95342") == (("95642", 1.0, 1.0),)


@pytest.mark.parametrize("temperature, low, high", [(872.5, "grph30.45t", "grph30.46t"),
                                                   (823.15, "grph30.45t", "grph30.46t"),
                                                   (700, "grph30.43t", "grph30.44t")])  # at a table temperature: lower pair
def test_graphite_sab_brackets_temperature(temperature, low, high):
    registry = ThermRegistry(ENDF8_LIB80X, CoreConditions(shared_graphite_sab=None))
    name = registry.graphite("salt", temperature)
    assert registry.cards() == f"therm {name} {temperature:g} {low} {high}"


def test_v1_uses_one_shared_graphite_table_at_900K():
    registry = ThermRegistry(ENDF7_V1, CoreConditions())
    assert registry.graphite("reflector", 900) == registry.graphite("dummy", 959) == "grph"
    assert registry.cards() == "therm grph 900 gre6.18t gre6.20t"


# ------------------------------------------------------------------ thermal model


def test_zero_power_is_inlet_temperature_everywhere():
    c = thermal.core_conditions(0.0, 250_190)
    for value in (c.kernel, c.coating, c.matrix, c.pebble_graphite, c.dummy, c.salt, c.reflector, c.structure):
        assert value == pytest.approx(thermal.T_INLET)


def test_full_power_reproduces_published_averages():
    c = thermal.core_conditions(280e6, 250_190)
    assert c.kernel == pytest.approx(953.6)
    assert c.matrix == pytest.approx(928.9)
    assert c.reflector == pytest.approx(858.9)
    assert c.salt == pytest.approx((823.15 + 921.94) / 2)
    assert c.dummy == c.salt and c.structure == thermal.T_INLET
    assert c.salt_density == pytest.approx(1.9825, abs=0.006)  # published average 1982.5 kg/m3
    assert c.per_component_sab and c.sic_sab == "sic"


def test_graphite_dilution_raises_fuel_temperature_at_fixed_power():
    full = thermal.core_conditions(140e6, 250_190)
    diluted = thermal.core_conditions(140e6, 125_095)
    assert diluted.salt == pytest.approx(full.salt)
    assert diluted.kernel - diluted.salt == pytest.approx(2 * (full.kernel - full.salt))


def test_power_without_fuel_raises():
    with pytest.raises(ValueError):
        thermal.core_conditions(1e6, 0)
    assert thermal.core_conditions(0.0, 0).kernel == pytest.approx(thermal.T_INLET)


# ------------------------------------------------------------------ rendering


def test_v1_template_rendering_is_unchanged_apart_from_therm_marker(repo_root):
    with open(f"{repo_root}/gFHR_template.serpent") as f:
        template = f.read()
    registry = ThermRegistry(ENDF7_V1, CoreConditions())
    rendered = render_template(template, ENDF7_V1, CoreConditions(), registry, "MARK")
    assert rendered.replace("MARK", registry.cards()) == template


def render_v2(tmp_path, power=140e6, coupled=True):
    core = build_core(tmp_path / "case")
    for channel in core.zones:
        for zone in channel:
            zone.initialize({"graphpeb": 0.5, "fuel20": 0.5})
            core.initialize_materials(zone.inventory)
    core.rng = np.random.default_rng(3)
    core.insert([("graphpeb", 0.5), ("reinsert", 1.0), ("fuel20", 1.0)], 18.63, 0.02288)
    core.nuclear_data = ENDF8_LIB80X
    fuel = sum(n for ch in core.zones for z in ch for k, n in z.inventory.items() if "fuel" in k)
    core.conditions = (thermal.core_conditions(power, fuel, nominal_pebbles=len(core.pebble_locations))
                       if coupled else thermal.uncoupled_conditions())
    with open(core.generate_input({"pop": "1000 10 5"}, 6.525, 0, cr_position=300.0)) as f:
        return core, f.read()


def test_v2_deck_is_consistent(tmp_path):
    core, deck = render_v2(tmp_path)
    assert validate_deck(deck, ENDF8_LIB80X) == []
    assert ".09c" not in deck
    assert not re.search(r"^\s+6000\.", deck, re.M)
    assert "gre6" not in deck


def test_v2_deck_component_temperatures(tmp_path):
    core, deck = render_v2(tmp_path)
    c = core.conditions
    lines = deck.splitlines()

    def tmp_of(prefix):
        line = next(l for l in lines if l.startswith(f"mat {prefix}"))
        return float(re.search(r"\btmp (\S+)", line).group(1))

    assert tmp_of("fuel20_R1Z1G1 ") == pytest.approx(c.kernel, abs=1e-3)
    assert tmp_of("fuel20_R1Z1G1_SiC") == pytest.approx(c.coating, abs=1e-3)
    assert tmp_of("fuel20_R1Z1G1_matrix") == pytest.approx(c.matrix, abs=1e-3)
    assert tmp_of("graphpeb_R1Z1G1") == pytest.approx(c.dummy, abs=1e-3)
    assert tmp_of("FLiBe ") == pytest.approx(c.salt, abs=1e-3)
    assert tmp_of("reflectorS") == pytest.approx(c.reflector, abs=1e-3)
    assert tmp_of("CBSS316H") == pytest.approx(c.structure, abs=1e-3)
    assert tmp_of("pebgraph") == pytest.approx(c.pebble_graphite, abs=1e-3)
    flibe = next(l for l in lines if l.startswith("mat FLiBe "))
    assert flibe.split()[2] == f"-{c.salt_density:.4f}"


def test_v2_sic_uses_sic_thermal_scattering(tmp_path):
    _, deck = render_v2(tmp_path)
    sic = next(l for l in deck.splitlines() if l.startswith("mat fuel20_R1Z1G1_SiC"))
    assert "moder csic_coating 6012" in sic and "moder csic_coating 6013" in sic
    assert "moder sisic_coating 14028" in sic and "grph" not in sic
    pyc = next(l for l in deck.splitlines() if l.startswith("mat fuel20_R1Z1G1_PyC"))
    assert "moder grph_coating 6012 moder grph_coating 6013" in pyc


def test_v2_uncoupled_keeps_fixed_temperatures(tmp_path):
    core, deck = render_v2(tmp_path, coupled=False)
    assert validate_deck(deck, ENDF8_LIB80X) == []
    assert "tmp 959" in deck and "tmp 900" in deck


def test_validator_flags_inconsistencies():
    deck = ("mat a -1.0 tmp 950 moder grph_x 6012\n  6012.01c   1.0\n  6000.02c   1.0\n\n"
            "therm grph_x 1300 grph30.45t grph30.46t\n")
    problems = validate_deck(deck, ENDF8_LIB80X)
    assert any("6012.01c" in p for p in problems)
    assert any("natural carbon" in p for p in problems)
    assert any("outside" in p for p in problems)
