"""Golden tests of the Serpent deck that Core.generate_input renders (v1 behaviour).

Regenerate deliberately with ``pytest --update-golden`` and review the diff.
"""

import pytest
from helpers.synthetic import build_core

pytestmark = pytest.mark.characterization

SETTINGS = {"pop": " 24000 30 10", "power": " 280e6"}


def render(case_dir, seed_all, cr_position=300.0):
    core = build_core(case_dir)
    for channel in core.zones:
        for zone in channel:
            zone.initialize({"graphpeb": 0.5, "fuel20": 0.5})
            core.initialize_materials(zone.inventory)
    seed_all(11)
    core.insert([("graphpeb", 0.5), ("reinsert", 1.0), ("fuel20", 1.0)], 18.63, 0.02288, None)
    seed_all(12)
    name = core.generate_input(SETTINGS, 0, 6.525, 0, cr_position=cr_position)
    with open(name) as f:
        deck = f.read()
    with open(f"pebble_positions_{core.iteration}.csv") as f:
        positions = f.read()
    return name, deck, positions


def test_deck_matches_golden(tmp_path, seed_all, golden):
    name, deck, positions = render(tmp_path / "case", seed_all)
    assert name == "synthetic_1.serpent"
    golden("synthetic_1.serpent", deck)
    golden("pebble_positions_1.csv", positions)


def test_deck_rendering_is_deterministic_given_seeds(tmp_path, seed_all):
    first = render(tmp_path / "a", seed_all)
    second = render(tmp_path / "b", seed_all)
    assert first == second


def test_deck_structure(tmp_path, seed_all):
    _, deck, positions = render(tmp_path / "case", seed_all, cr_position=123.4)
    assert "<insertion_position>" not in deck
    assert "surf rod_insertion_limit pz 123.4" in deck
    assert "dep daystep 6.525\n" in deck
    assert "set pop  24000 30 10\n" in deck
    assert 'pbed u_pb u_flibe "pebble_positions_1.csv"' in deck
    assert len(positions.splitlines()) == 108
    # every burnable fuel material has its TRISO particle definition
    assert deck.count(" burn 1") == deck.count("\nparticle p")
