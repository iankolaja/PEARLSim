"""Nuclear-data profiles and temperature-dependent rendering of Serpent materials.

A ``NuclearDataProfile`` describes one Serpent library: the library-ID suffix
processed at each temperature, nuclide renames needed by that library, and the
thermal-scattering (S(a,b)) tables available for graphite and SiC.
``CoreConditions`` holds the temperature of each material component.

The defaults (``ENDF7_V1`` and ``CoreConditions()``) reproduce the v1 decks
exactly. Dataset v2 uses ``ENDF8_LIB80X`` with power-coupled temperatures
(``pearlsim.thermal``). See docs/nuclear_data.md in the workspace for provenance;
changing profiles changes k_eff and defines a new dataset version.

Serpent syntax used here (docs/serpent cards_reference.md, set_options.md):
``mat NAME DENS ... tmp T moder THNAME ZA`` (``moder`` may be repeated, one
nuclide per entry) and ``therm THNAME T LIB1 LIB2`` (interpolation between two
tables at temperature T).
"""

import re
from dataclasses import dataclass, field

LIB = "<lib>"

# natural carbon (atom fractions; mass fractions follow from the isotope masses)
_C12_ATOM, _C13_ATOM = 0.9893, 0.0107
_C12_MASS = _C12_ATOM * 12.0 / (_C12_ATOM * 12.0 + _C13_ATOM * 13.003355)
_NATURAL_CARBON = (("6012", _C12_ATOM, _C12_MASS), ("6013", _C13_ATOM, 1.0 - _C12_MASS))


def format_temperature(value):
    """Temperatures as v1 wrote them: integral values without a decimal point."""
    value = round(float(value), 3)
    return str(int(value)) if value == int(value) else f"{value:g}"


@dataclass(frozen=True)
class NuclearDataProfile:
    name: str
    # (base temperature K, library suffix), ascending
    suffixes: tuple
    # ZA -> ((ZA, atom fraction, mass fraction), ...) expansions, e.g. natural carbon
    expansions: dict = field(default_factory=dict)
    # ZA renames, e.g. isomer naming conventions
    renames: dict = field(default_factory=dict)
    # concentration-key renames applied when writing (e.g. decay-only ZAI that gains data)
    key_renames: dict = field(default_factory=dict)
    # S(a,b) table families: (temperature K, table name), ascending
    graphite_tables: tuple = ()
    sic_carbon_tables: tuple = ()
    sic_silicon_tables: tuple = ()
    # S(a,b) tables with continuous secondary-energy distributions cannot be pre-interpolated with
    # 'therm NAME T LIB1 LIB2'; 'thermstoch' mixes the two tables stochastically (docs/serpent cards_reference.md)
    therm_card: str = "therm"

    def suffix(self, temperature):
        """Suffix of the library processed at the highest temperature not above ``temperature``.

        Serpent's Doppler preprocessor (``tmp``) needs the base temperature at or below the target.
        """
        chosen = None
        for base, suffix in self.suffixes:
            if base <= temperature + 1e-9:
                chosen = suffix
        if chosen is None:
            raise ValueError(f"{self.name}: no library at or below {temperature} K "
                             f"(lowest is {self.suffixes[0][0]} K)")
        return chosen

    def map_za(self, za):
        """Nuclides (ZA, atom fraction, mass fraction) to write for one ZA."""
        if za in self.expansions:
            return self.expansions[za]
        return ((self.renames.get(za, za), 1.0, 1.0),)

    @property
    def expands_carbon(self):
        return "6000" in self.expansions


ENDF7_V1 = NuclearDataProfile(
    name="endf7_v1",
    suffixes=((300, ".03c"), (600, ".06c"), (900, ".09c"), (1200, ".12c"), (1500, ".15c"), (1800, ".18c")),
    graphite_tables=((296, "gre6.00t"), (400, "gre6.04t"), (500, "gre6.08t"), (600, "gre6.12t"),
                     (700, "gre6.16t"), (800, "gre6.18t"), (1000, "gre6.20t"), (1200, "gre6.22t"),
                     (1600, "gre6.24t"), (1999, "gre6.26t")),
)

ENDF8_LIB80X = NuclearDataProfile(
    name="endf8_lib80x",
    suffixes=((294, ".00c"), (600, ".01c"), (900, ".02c"), (1200, ".03c"), (2500, ".04c")),
    expansions={"6000": _NATURAL_CARBON},
    # isomers, Serpent-library to Lib80x (MCNP) IDs; Ag-110m and Pm-148m appear only after depletion
    renames={"95342": "95642", "95344": "95644", "47310": "47510", "61348": "61548"},
    key_renames={"60130": "6013" + LIB},
    graphite_tables=tuple(zip((296, 400, 500, 600, 700, 800, 1000, 1200, 1600, 1999),
                              (f"grph30.4{i}t" for i in range(10)))),
    sic_carbon_tables=tuple(zip((300, 400, 500, 600, 700, 800, 1000, 1200),
                                (f"c-sic.4{i}t" for i in range(8)))),
    sic_silicon_tables=tuple(zip((300, 400, 500, 600, 700, 800, 1000, 1200),
                                 (f"si-sic.4{i}t" for i in range(8)))),
    therm_card="thermstoch",  # Lib80x tables: "Unable to interpolate continuous-energy S(a,b) data" with therm
)

PROFILES = {p.name: p for p in (ENDF7_V1, ENDF8_LIB80X)}


@dataclass
class CoreConditions:
    """Temperature (K) of each material component, and the FLiBe density.

    Defaults are the v1 values: fuel pebble materials at 959 K (including the
    fuel-zone matrix and dummy graphite pebbles), everything in the core template
    at 900 K, one graphite S(a,b) interpolation at 900 K shared by every
    material, and graphite S(a,b) on SiC.
    """
    kernel: float = 959.0          # UCO kernel
    coating: float = 959.0         # buffer, PyC, SiC layers
    matrix: float = 959.0          # fuel-zone matrix graphite
    pebble_graphite: float = 900.0  # pebble core and shell (template pebgraph/pebshell)
    dummy: float = 959.0           # dummy graphite pebbles
    salt: float = 900.0
    reflector: float = 900.0
    structure: float = 900.0       # steel barrel/vessel and B4C rods
    salt_density: float = None     # g/cm3; None keeps the template value
    # Li-6 share of FLiBe lithium in atom ppm; None keeps the template's 25 ppm (v1: 0.00005 Li-6 per
    # 2 Li). The gFHR benchmark specifies 99.995 at.% Li-7, i.e. 50 ppm (Satvat et al. 2021, Table 2).
    salt_li6_ppm: float = None
    # pebble central graphite core density in g/cm3; None keeps the template's 1.74 (v1). The gFHR
    # benchmark uses 1.41 for buoyancy (Satvat et al. 2021 §2; INL/RPT-23-72727 Table 2).
    pebble_core_density: float = None
    shared_graphite_sab: float = 900.0  # v1: one 'grph' therm at this T; None = per component
    sic_sab: str = "graphite"      # v1: 'graphite'; v2: 'sic'

    @property
    def per_component_sab(self):
        return self.shared_graphite_sab is None


class ThermRegistry:
    """Collects the S(a,b) names used in a deck and renders their ``therm`` cards."""

    def __init__(self, profile, conditions):
        self.profile = profile
        self.conditions = conditions
        self._cards = {}

    def _bracket(self, tables, temperature, family):
        for (t_lo, lib_lo), (t_hi, lib_hi) in zip(tables, tables[1:]):
            if t_lo <= temperature <= t_hi:
                return lib_lo, lib_hi
        raise ValueError(f"{self.profile.name}: {family} S(a,b) tables do not bracket {temperature} K "
                         f"({tables[0][0]}-{tables[-1][0]} K)")

    def _name(self, family, tables, component, temperature, za):
        # Serpent links a therm name to one nuclide only ("S(a,b) library ... is already assigned to
        # ZA = 6012"), so isotopic entries get one name each; natural elements (Z000) keep the v1 names.
        isotope = "" if za.endswith("000") else f"_{za}"
        if family == "grph" and not self.conditions.per_component_sab:
            name, temperature = f"grph{isotope}", self.conditions.shared_graphite_sab
        else:
            name = f"{family}_{component}{isotope}"
        if name not in self._cards:
            lo, hi = self._bracket(tables, temperature, family)
            self._cards[name] = f"{self.profile.therm_card} {name} {format_temperature(temperature)} {lo} {hi}"
        return name

    def graphite(self, component, temperature, za="6000"):
        return self._name("grph", self.profile.graphite_tables, component, temperature, za)

    def sic_carbon(self, component, temperature, za):
        return self._name("csic", self.profile.sic_carbon_tables, component, temperature, za)

    def sic_silicon(self, component, temperature, za):
        return self._name("sisic", self.profile.sic_silicon_tables, component, temperature, za)

    def moder_text(self, component, temperature, nuclides, sic=False, separator="  "):
        """``moder`` options for a carbon-bearing material (graphite or SiC).

        ``nuclides``: the ZA values present (after profile mapping). v1 writes the
        single option ``moder  grph 6000``.
        """
        parts = []
        if self.conditions.per_component_sab:
            separator = " "  # v1 decks keep their original double space
        if sic and self.conditions.sic_sab == "sic":
            for za in nuclides:
                if za.startswith("6"):
                    parts.append(f"moder {self.sic_carbon(component, temperature, za)} {za}")
                elif za.startswith("14"):
                    parts.append(f"moder {self.sic_silicon(component, temperature, za)} {za}")
        else:
            for za in nuclides:
                if za.startswith("60"):
                    parts.append(f"moder{separator}{self.graphite(component, temperature, za)} {za}")
        return " ".join(parts)

    def cards(self):
        return "\n".join(self._cards.values())


def render_concentrations(concentrations, profile, temperature):
    """Nuclide lines ``  ZA.xxc    value`` for a concentration dictionary (keys ``ZA<lib>`` or ZAI)."""
    suffix = profile.suffix(temperature)
    lines = []
    for key, value in concentrations.items():
        key = profile.key_renames.get(key, key)
        if key.endswith(LIB):
            for za, atom_frac, mass_frac in profile.map_za(key[:-len(LIB)]):
                frac = mass_frac if value < 0 else atom_frac
                written = value if frac == 1.0 else value * frac
                lines.append(f"  {za}{suffix}    {written}\n")
        else:
            lines.append(f"  {key}    {value}\n")
    return "".join(lines)


def mapped_carbon_silicon(concentrations, profile):
    """ZA values of C and Si present after profile mapping (for ``moder`` options)."""
    nuclides = []
    for key in concentrations:
        key = profile.key_renames.get(key, key)
        if key.endswith(LIB):
            for za, _, _ in profile.map_za(key[:-len(LIB)]):
                if za.startswith("6") or za.startswith("14"):
                    nuclides.append(za)
    return nuclides


# Template materials and the component whose temperature they take.
TEMPLATE_COMPONENTS = {
    "CBSS316H": "structure", "RVSS316H": "structure", "B4C": "structure",
    "pebgraph": "pebble_graphite", "pebshell": "pebble_graphite",
    "FLiBe": "salt", "FLiBeInC": "salt",
    "reflector": "reflector", "reflectorT": "reflector", "reflectorB": "reflector", "reflectorS": "reflector",
}

_NUCLIDE_LINE = re.compile(r"^(\s*)(\d+)\.09c(\s+)(\S+)\s*$")
_MODER = re.compile(r"moder(\s+)grph(\s+)6000")
_THERM_LINE = re.compile(r"^therm grph 900 gre6\.18t gre6\.20t\s*$")


def render_template(text, profile, conditions, registry, therm_marker="%%PEARLSIM_THERM_CARDS%%"):
    """Rewrite the core template's materials for a profile and component temperatures.

    The template is written for ENDF/B-VII.0 at 900 K (``.09c``, ``tmp 900``,
    ``moder grph 6000``, one ``therm grph`` card). The ``therm`` line is replaced by
    ``therm_marker``; the caller substitutes all cards once fuel materials are rendered.
    """
    out = []
    component = None
    for line in text.split("\n"):
        if component == "salt" and conditions.salt_li6_ppm is not None:
            line = _set_lithium(line, conditions.salt_li6_ppm)
        if line.startswith("mat "):
            name = line.split()[1]
            component = TEMPLATE_COMPONENTS.get(name)
            if component is None:
                raise ValueError(f"template material {name!r} has no temperature component")
            temperature = getattr(conditions, component)
            line = re.sub(r"tmp 900\b", f"tmp {format_temperature(temperature)}", line)
            if component == "salt" and conditions.salt_density is not None:
                line = re.sub(rf"^mat {name} -\S+", f"mat {name} -{conditions.salt_density:.4f}", line)
            if name == "pebgraph" and conditions.pebble_core_density is not None:
                line = re.sub(r"^mat pebgraph(\s+)-\S+", lambda m: f"mat pebgraph{m.group(1)}"
                              f"-{conditions.pebble_core_density:.4f}", line)
            if _MODER.search(line):
                carbon = [za for za, _, _ in profile.map_za("6000")]
                sep = _MODER.search(line).group(1)
                moder = registry.moder_text(component, temperature, carbon, separator=sep)
                line = _MODER.sub(lambda m: moder, line)
            out.append(line)
            continue
        match = _NUCLIDE_LINE.match(line)
        if match and component is not None:
            indent, za, space, value = match.groups()
            suffix = profile.suffix(getattr(conditions, component))
            mapped = profile.map_za(za)
            if len(mapped) == 1 and mapped[0][0] == za:
                out.append(f"{indent}{za}{suffix}{space}{value}")
            else:
                number = float(value)
                for new_za, atom_frac, mass_frac in mapped:
                    frac = mass_frac if number < 0 else atom_frac
                    out.append(f"{indent}{new_za}{suffix}{space}{number * frac:.6E}")
            continue
        if _THERM_LINE.match(line):
            out.append(therm_marker)
            component = None
            continue
        if not line.strip() or line.startswith("%"):
            component = None if not line.strip() else component
        out.append(line)
    return "\n".join(out)


V1_FLIBE_LI6_PPM = 25.0  # core template: 0.00005 Li-6 per 2 Li
V1_PEBBLE_CORE_DENSITY = 1.74  # core template pebgraph, g/cm3
FLIBE_LI_PER_FORMULA = 2.0  # Li2BeF4: the template lists Li-7 + Li-6 = 2, Be 1, F 4 (atom ratios)


def _set_lithium(line, li6_ppm):
    """Rewrite a FLiBe Li-7 or Li-6 atom-ratio line for a Li-6 share of ``li6_ppm`` (atom ppm of Li)."""
    match = _NUCLIDE_LINE.match(line)
    if not match or match.group(2) not in ("3006", "3007"):
        return line
    indent, za, space, _ = match.groups()
    li6 = FLIBE_LI_PER_FORMULA * li6_ppm * 1e-6
    value = li6 if za == "3006" else FLIBE_LI_PER_FORMULA - li6
    return f"{indent}{za}.09c{space}{value:.8f}"


_MAT_HEADER = re.compile(r"^mat\s+(\S+)\s.*?\btmp\s+(\S+)")
_DECK_NUCLIDE = re.compile(r"^\s+(\d+)(\.\d\dc)\s+\S+\s*$")


def validate_deck(text, profile):
    """Consistency checks on a rendered deck; returns a list of problems (empty if consistent).

    * every ``moder`` name has a ``therm`` card, and every card's tables bracket its temperature;
    * every nuclide in a ``mat`` block uses the library suffix the profile assigns to that
      block's ``tmp``;
    * with carbon expansion, no natural carbon (6000) remains.
    """
    problems = []
    therm = {}
    for line in text.splitlines():
        tokens = line.split()
        if tokens[:1] in (["therm"], ["thermstoch"]) and len(tokens) == 5:
            therm[tokens[1]] = (float(tokens[2]), tokens[3], tokens[4])
    families = {}
    for tables in (profile.graphite_tables, profile.sic_carbon_tables, profile.sic_silicon_tables):
        for t, lib in tables:
            families[lib] = t
    for name, (t, lo, hi) in therm.items():
        if lo not in families or hi not in families:
            problems.append(f"therm {name}: tables {lo}, {hi} not in profile {profile.name}")
        elif not families[lo] <= t <= families[hi]:
            problems.append(f"therm {name}: {t} K outside [{families[lo]}, {families[hi]}]")
    links = {}
    for name, za in set(re.findall(r"\bmoder\s+(\S+)\s+(\d+)", text)):
        links.setdefault(name, set()).add(za)
    for name, zas in links.items():
        if name not in therm:
            problems.append(f"moder {name} has no therm card")
        if len(zas) > 1:
            problems.append(f"therm {name} linked to several nuclides {sorted(zas)} (Serpent allows one)")
    current, expected = None, None
    for number, line in enumerate(text.splitlines(), start=1):
        header = _MAT_HEADER.match(line)
        if header:
            current = header.group(1)
            expected = profile.suffix(float(header.group(2)))
            continue
        if line.startswith("mat "):  # material without tmp: library ID not checked
            current, expected = line.split()[1], None
            continue
        nuclide = _DECK_NUCLIDE.match(line)
        if nuclide and current is not None:
            za, suffix = nuclide.groups()
            if expected is not None and suffix != expected:
                problems.append(f"line {number}: {za}{suffix} in {current} (tmp expects {expected})")
            if profile.expands_carbon and za == "6000":
                problems.append(f"line {number}: natural carbon 6000 in {current}")
        elif not line.strip():
            current, expected = None, None
    return problems


def missing_library_entries(deck_text, xsdata_path):
    """Nuclide IDs (``ZA.xxc``) and S(a,b) tables used by a deck but absent from an xsdata index."""
    with open(xsdata_path) as f:
        available = {line.split()[0] for line in f if line.strip()}
    used = set(re.findall(r"^\s+(\d+\.\d\dc)\s", deck_text, re.M))
    for line in deck_text.splitlines():
        tokens = line.split()
        if tokens[:1] in (["therm"], ["thermstoch"]) and len(tokens) == 5:
            used.update(tokens[3:5])
    return sorted(used - available)
