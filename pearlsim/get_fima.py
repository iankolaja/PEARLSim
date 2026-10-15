FIRST_ACTINIDE_Z = 89  # actinium


def atomic_number(nuclide_key):
    """Atomic number from a PEARLSim nuclide key.

    Keys are either ZA with a library placeholder (``"92235<lib>"``, Z*1000 + A) or a bare
    ZAI as written by Serpent for decay-only nuclides (``"922350"``, Z*10000 + A*10 + I).
    """
    if nuclide_key.endswith("<lib>"):
        return int(nuclide_key[:-len("<lib>")]) // 1000
    return int(nuclide_key) // 10000


def heavy_metal_density(conc_dict):
    """Total atom density of actinides (Z >= 89) in a concentration dictionary."""
    return sum(value for key, value in conc_dict.items() if atomic_number(key) >= FIRST_ACTINIDE_Z)


def get_fima(conc_dict, initial_atoms):
    """Percent fissions per initial metal atom: heavy-metal depletion relative to ``initial_atoms``."""
    return 100 * (initial_atoms - heavy_metal_density(conc_dict)) / initial_atoms
