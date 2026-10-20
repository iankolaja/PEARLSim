"""Core-wide, per-component temperatures as a function of power (quasi-static, uniform).

Calibrated to the INL Virtual Test Bed gFHR model, which tabulates the Kairos gFHR
benchmark (https://github.com/idaholab/virtual_test_bed, pbfhr/gFHR; reactor
description and coupled Griffin-Pronghorn results):

* 280 MWth, inlet 823.15 K, mass flow 1173 kg/s, outlet 921.94 K;
* equilibrium-core averages: fuel 953.6 K, moderator (pebble graphite) 928.9 K,
  reflector 858.9 K, FLiBe density 1982.5 kg/m3.

Model (docs/dataset_v2_design.md section 1a): coolant flow is fixed, so the mean
salt temperature rises linearly with power. Pebble temperatures rise above the
salt in proportion to the power per *fuel* pebble (conduction plus convection at
fixed flow is linear in heat rate). Dummy graphite pebbles carry no heat source
and sit at the salt temperature. Steel structures and control rods sit at the
inlet temperature. At zero power everything is at the inlet temperature; at full
power with an all-fuel core the published averages are reproduced.
"""

from pearlsim.nuclear_data import CoreConditions

NOMINAL_POWER = 280e6           # W
NOMINAL_PEBBLES = 250_190       # VTB / Kairos pebble count
T_INLET = 823.15                # K
SALT_RISE_NOMINAL = 921.94 - 823.15   # outlet - inlet at nominal power, fixed flow
T_FUEL_NOMINAL = 953.6
T_MOD_NOMINAL = 928.9
T_REFLECTOR_NOMINAL = 858.9


FLIBE_LI6_PPM = 50.0  # 99.995 at.% Li-7 (Satvat et al. 2021, Table 2); v1 decks had 25 ppm
PEBBLE_CORE_DENSITY = 1.41  # g/cm3, benchmark buoyancy value (Satvat 2021 §2; INL Table 2); v1 decks had 1.74


def salt_density(temperature):
    """FLiBe density in g/cm3 (linear correlation; 1.9872 at the nominal mean salt temperature)."""
    return 2.413 - 4.88e-4 * temperature


def core_conditions(power, fuel_pebbles, nominal_power=NOMINAL_POWER, nominal_pebbles=NOMINAL_PEBBLES,
                    li6_ppm=FLIBE_LI6_PPM, pebble_core_density=PEBBLE_CORE_DENSITY):
    """Per-component temperatures for total power ``power`` (W) with ``fuel_pebbles`` fuel pebbles in core."""
    if power < 0:
        raise ValueError("power must be non-negative")
    x = power / nominal_power
    t_salt = T_INLET + 0.5 * SALT_RISE_NOMINAL * x
    t_salt_nominal = T_INLET + 0.5 * SALT_RISE_NOMINAL
    if fuel_pebbles > 0:
        q_rel = (power / fuel_pebbles) / (nominal_power / nominal_pebbles)
    elif power == 0:
        q_rel = 0.0
    else:
        raise ValueError("positive power with no fuel pebbles in core")
    t_fuel = t_salt + (T_FUEL_NOMINAL - t_salt_nominal) * q_rel
    t_mod = t_salt + (T_MOD_NOMINAL - t_salt_nominal) * q_rel
    return CoreConditions(
        kernel=t_fuel, coating=t_fuel, matrix=t_mod, pebble_graphite=t_mod,
        dummy=t_salt, salt=t_salt,
        reflector=T_INLET + (T_REFLECTOR_NOMINAL - T_INLET) * x,
        structure=T_INLET,
        salt_density=salt_density(t_salt), salt_li6_ppm=li6_ppm, pebble_core_density=pebble_core_density,
        shared_graphite_sab=None, sic_sab="sic", shared_fuel_components=True,
    )


def uncoupled_conditions(temperature=900.0, fuel_temperature=959.0, li6_ppm=FLIBE_LI6_PPM,
                         pebble_core_density=PEBBLE_CORE_DENSITY):
    """v2 data without thermal feedback: fixed temperatures, but per-component S(a,b) and SiC S(a,b)."""
    return CoreConditions(kernel=fuel_temperature, coating=fuel_temperature, matrix=fuel_temperature,
                          pebble_graphite=temperature, dummy=temperature, salt=temperature,
                          reflector=temperature, structure=temperature, salt_density=None,
                          salt_li6_ppm=li6_ppm, pebble_core_density=pebble_core_density,
                          shared_graphite_sab=None, sic_sab="sic", shared_fuel_components=True)
