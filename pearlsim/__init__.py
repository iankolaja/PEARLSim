"""PEARLSim: zone-based pebble-bed reactor depletion with Serpent 2.

Core bookkeeping modules are imported eagerly. Modules with heavy or optional
dependencies are loaded on first attribute access (PEP 562), so that
``import pearlsim`` does not require TensorFlow or other ML libraries.
"""

import importlib

from .core import Core
from .get_fima import get_fima
from .group_definitions import (
    AXIAL_COARSE_ZONE_BOUNDS,
    AXIAL_SUBZONE_BOUNDS,
    AXIAL_ZONE_BOUNDS,
    BURNUP_BINS_9,
    ENERGY_CENTERS_3,
    ENERGY_CENTERS_7,
    ENERGY_CENTERS_14,
    ENERGY_CENTERS_56,
    ENERGY_GRID_3,
    ENERGY_GRID_7,
    ENERGY_GRID_14,
    ENERGY_GRID_56,
    HEIGHT_GRID_8,
    HEIGHT_GRID_10,
    HEIGHT_GRID_20,
    RADIAL_COARSE_ZONE_BOUNDS,
    RADIAL_SUBZONE_BOUNDS,
    RADIAL_ZONE_BOUNDS,
    RADIUS_GRID_3,
    RADIUS_GRID_4,
    RADIUS_GRID_8,
)
from .material import Material

# name -> (module, attribute) for lazily imported objects
_LAZY = {
    "Simulation": (".simulation", "Simulation"),
    "Pebble_Model": (".pebble_model", "Pebble_Model"),
    "standardize": (".ml_utilities", "standardize"),
    "unstandardize": (".ml_utilities", "unstandardize"),
    "extract_from_bumat": (".ml_utilities", "extract_from_bumat"),
    "read_core_flux": (".results_processing", "read_core_flux"),
    "read_res_file": (".results_processing", "read_res_file"),
    "LSTM_predictor": (".lstm", "LSTM_predictor"),
}


def __getattr__(name):
    if name in _LAZY:
        module_name, attribute = _LAZY[name]
        value = getattr(importlib.import_module(module_name, __name__), attribute)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "Core", "Material", "Pebble_Model", "Simulation", "standardize", "unstandardize", "read_core_flux",
    "read_res_file",
    "extract_from_bumat", "get_fima", "ENERGY_GRID_14", "ENERGY_CENTERS_14",
    "ENERGY_GRID_56", "ENERGY_CENTERS_56", "ENERGY_GRID_7", "ENERGY_CENTERS_7",
    "RADIUS_GRID_4", "HEIGHT_GRID_10", "RADIUS_GRID_8", "HEIGHT_GRID_20",
    "RADIUS_GRID_3", "HEIGHT_GRID_8", "RADIAL_ZONE_BOUNDS", "AXIAL_ZONE_BOUNDS",
    "RADIAL_SUBZONE_BOUNDS", "AXIAL_SUBZONE_BOUNDS", "RADIAL_COARSE_ZONE_BOUNDS",
    "AXIAL_COARSE_ZONE_BOUNDS", "ENERGY_GRID_3", "ENERGY_CENTERS_3", "LSTM_predictor",
    "BURNUP_BINS_9",
]
