"""A tiny synthetic core in the file formats PEARLSim reads.

Geometry: 2 radial channels (r < 5, 5 <= r < 10) x 3 axial zones (z < 10,
10 <= z < 20, z >= 20), with pebble points on a deterministic polar grid.
The real gFHR template, material files and (trimmed) TRISO file are copied in,
because v1 resolves every input path as ``"../" + path`` relative to the run
directory.
"""

import math
import os
import shutil

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

RADIAL_BOUNDS = (5.0, 10.0)
AXIAL_DIVISIONS = (10.0, 20.0)
AXIAL_CENTERS = (5.0, 15.0, 25.0)
TRISO_LINES = 50


def _points():
    """Deterministic pebble points: 108 total, all with r < 10."""
    points = []
    for z in AXIAL_CENTERS:
        for dz in (-2.5, 0.0, 2.5):
            # inner channel: r = 2.5 (6 points per layer)
            for k in range(6):
                a = 2 * math.pi * k / 6
                points.append((2.5 * math.cos(a), 2.5 * math.sin(a), z + dz))
            # outer channel: r = 7.5 (6 points per layer)
            for k in range(6):
                a = 2 * math.pi * (k + 0.5) / 6
                points.append((7.5 * math.cos(a), 7.5 * math.sin(a), z + dz))
    return points


def write_core(case_dir):
    """Write the synthetic core inputs to ``case_dir`` and return the file names."""
    os.makedirs(case_dir, exist_ok=True)
    with open(os.path.join(case_dir, "zones.inp"), "w") as f:
        for r in RADIAL_BOUNDS:
            f.write("radial_channel\n")
            f.write("z_divisions " + " ".join(str(z) for z in AXIAL_DIVISIONS) + "\n")
            f.write(f"point 0 {r}\n")
            f.write(f"point 30 {r}\n")
            f.write("end\n\n")
    with open(os.path.join(case_dir, "points.dat"), "w") as f:
        for x, y, z in _points():
            f.write(f"{x:.4f} {y:.4f} {z:.4f} 2.0 point\n")

    shutil.copyfile(os.path.join(REPO, "gFHR_template.serpent"), os.path.join(case_dir, "template.serpent"))
    materials = os.path.join(case_dir, "materials")
    os.makedirs(materials, exist_ok=True)
    for name in ("fuel20.dat", "graphpeb.dat", "SiC_gFHR.dat", "PyC_gFHR.dat", "buffer_gFHR.dat",
                 "matrix_gFHR.dat"):
        shutil.copyfile(os.path.join(REPO, "materials", name), os.path.join(materials, name))
    with open(os.path.join(REPO, "gFHR_triso.dat")) as fin, open(os.path.join(case_dir, "triso.dat"), "w") as fout:
        for i, line in enumerate(fin):
            if i >= TRISO_LINES:
                break
            fout.write(line)
    return {"zones": "zones.inp", "points": "points.dat", "template": "template.serpent", "triso": "triso.dat"}


def setup_lines(num_trisos=4):
    """Input-deck lines that set up the synthetic core (paths relative to the case directory)."""
    return [
        'core_geometry "template.serpent"',
        f"triso_file {num_trisos} triso.dat",
        "fresh fuel20 materials/fuel20.dat graphpeb materials/graphpeb.dat",
        "static_fuel_materials SiC materials/SiC_gFHR.dat PyC materials/PyC_gFHR.dat "
        "buffer materials/buffer_gFHR.dat matrix materials/matrix_gFHR.dat",
        "define_zones zones.inp points.dat",
    ]


def build_core(case_dir, simulation_name="synthetic"):
    """Return a ``Core`` with zones defined and fresh materials loaded (no Simulation)."""
    from pearlsim.core import Core
    from pearlsim.material import Material

    write_core(case_dir)
    core = Core(simulation_name)
    core.define_zones(os.path.join(case_dir, "zones.inp"), os.path.join(case_dir, "points.dat"))
    for key, path in (("fuel20", "materials/fuel20.dat"), ("graphpeb", "materials/graphpeb.dat")):
        core.fresh_pebbles[key] = Material(key, os.path.join(case_dir, path))
    for key, path in (("SiC", "materials/SiC_gFHR.dat"), ("PyC", "materials/PyC_gFHR.dat"),
                      ("buffer", "materials/buffer_gFHR.dat"), ("matrix", "materials/matrix_gFHR.dat")):
        core.static_fuel_materials[key] = Material(key, os.path.join(case_dir, path))
    with open(os.path.join(case_dir, "template.serpent")) as f:
        core.core_geometry = f.read()
    return core
