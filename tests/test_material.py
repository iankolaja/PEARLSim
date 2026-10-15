"""Characterization tests for pearlsim.material and pearlsim.get_fima as of v1-as-run."""

import pytest

from pearlsim.get_fima import atomic_number, get_fima, heavy_metal_density
from pearlsim.material import Material, get_cross_section_string

pytestmark = pytest.mark.characterization

FUEL20_INITIAL_ACTINIDES = 0.00466953 + 0.0189728  # core.py: Core.initial_actinides


@pytest.mark.parametrize("temperature, expected", [
    (300, ".03c"), (599.9, ".03c"), (600, ".06c"), (900, ".09c"), (959, ".09c"),
    (1199, ".09c"), (1200, ".12c"), (1500, ".15c"), (1800, ".18c"), (2500, ".18c"),
])
def test_cross_section_suffix_uses_library_at_or_below_temperature(temperature, expected):
    assert get_cross_section_string(temperature) == expected


def test_cross_section_suffix_below_300K_raises():
    """Review #26 (fixed): no library exists below 300 K."""
    with pytest.raises(ValueError, match="300 K"):
        get_cross_section_string(250)


def test_fresh_fuel_file_is_burnable_with_sum_fix_density(repo_root):
    fuel = Material("fuel20", f"{repo_root}/materials/fuel20.dat")
    assert fuel.density == "sum fix"
    assert fuel.temperature == 959
    assert fuel.concentrations["92235<lib>"] == 4.66953e-03


def test_graphite_file_has_numeric_density(repo_root):
    graphite = Material("graphpeb", f"{repo_root}/materials/graphpeb.dat")
    assert graphite.density == -1.74


def test_add_and_multiply_are_elementwise_and_do_not_mutate():
    a = Material("a", {"1": 1.0, "2": 2.0})
    b = Material("b", {"2": 3.0, "3": 4.0})
    c = a * 0.5 + b * 2
    assert c.concentrations == {"1": 0.5, "2": 7.0, "3": 8.0}
    assert a.concentrations == {"1": 1.0, "2": 2.0}


def test_fix_b11_merges_both_keys():
    m = Material("m", {"50110": 1.0, "5011<lib>": 2.0, "5010<lib>": 5.0})
    m.fix_B11()
    assert m.concentrations == {"5010<lib>": 5.0, "5011<lib>": 3.0}


def test_merge_duplicates_folds_decay_only_id_into_library_id():
    m = Material("m", {"541350": 1.0, "54135<lib>": 2.0, "60130": 3.0})
    removed = m.merge_duplicates(debug=0)
    assert removed == ["541350"]
    assert m.concentrations == {"54135<lib>": 3.0, "60130": 3.0}


def test_fresh_fuel_fima_is_zero(repo_root):
    fuel = Material("fuel20", f"{repo_root}/materials/fuel20.dat")
    assert get_fima(fuel.concentrations, FUEL20_INITIAL_ACTINIDES) == 0.0


def test_fima_counts_heavy_metal_loss():
    burned = {"92235<lib>": 0.00466953 * 0.5, "92238<lib>": 0.0189728 * 0.9, "94239<lib>": 0.0}
    expected = 100 * (FUEL20_INITIAL_ACTINIDES - (0.00466953 * 0.5 + 0.0189728 * 0.9)) / FUEL20_INITIAL_ACTINIDES
    assert get_fima(burned, FUEL20_INITIAL_ACTINIDES) == pytest.approx(expected)


def test_fima_ignores_fluorine():
    """Review #25 (fixed): atomic number is parsed from the ZA/ZAI key."""
    conc = {"92235<lib>": 0.00466953, "92238<lib>": 0.0189728, "9019<lib>": 0.01}
    assert get_fima(conc, FUEL20_INITIAL_ACTINIDES) == pytest.approx(0.0)


@pytest.mark.parametrize("key, z", [("92235<lib>", 92), ("6000<lib>", 6), ("9019<lib>", 9), ("1001<lib>", 1),
                                    ("922350", 92), ("952421", 95), ("20060", 2), ("541350", 54)])
def test_atomic_number_from_za_and_zai_keys(key, z):
    assert atomic_number(key) == z


def test_heavy_metal_density_of_fresh_fuel(repo_root):
    fuel = Material("fuel20", f"{repo_root}/materials/fuel20.dat")
    assert heavy_metal_density(fuel.concentrations) == FUEL20_INITIAL_ACTINIDES
