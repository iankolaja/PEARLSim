"""Build the inputs of the Cerberus pebble-parking toy test (no Serpent is run here).

    python scripts/cerberus_toy/build_toy.py OUT_DIR [--scale-pebbles N]

The toy is a small cluster of real gFHR pebble positions (r < 30 cm, 200 < z < 240 cm,
~2,000 pebbles) in the real gFHR template, split into 2 x 2 zones, each holding fresh
fuel and graphite. PEARLSim's own deck writer produces the deck, so materials,
TRISO particles and pebble universes are exactly as in production decks.

Layouts (pebble -> material at the same positions):
  A  as written by PEARLSim (half fuel, half graphite per zone);
  B  zone R1Z1: half of its fuel pebbles become graphite (fuel volume changes);
  C  zone R2Z2: all fuel becomes graphite (an empty fuel slot).

Outputs in OUT_DIR:
  trisos/              TRISO position files referenced by the decks
  ref_A, ref_B, ref_C  ordinary decks with that layout's pebble file (file-based runs)
  live/                the deck for the live session; its pebble file is the pool:
                       every position carries one pebble of each material of its zone,
                       radius 2.0 where layout A places that material and 0 (parked) elsewhere
  live_scale/          (with --scale-pebbles) the same, with extra parked pebbles up to N in total
  toy.json             materials, pool order, per-layout radii, counts and fuel volumes
See docs/serpent/cerberus_hxf.md (workspace) for the mechanism being tested.
"""

import argparse
import json
import os
import re
import shutil
import sys

import numpy as np

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, REPO)

from pearlsim.core import Core  # noqa: E402
from pearlsim.material import Material  # noqa: E402

KERNEL_VOLUME_PER_PEBBLE = 0.36263376  # Core.get_volumes default
CLUSTER = {"r_max": 30.0, "z_min": 200.0, "z_max": 240.0}
RADIAL_BOUNDS = (20.0, 30.0)
AXIAL_SPLIT = 220.0
EMPTY_VOLUME = 1e-6  # reference-deck volume for the emptied slot (a burnable material needs one)
SETTINGS = {
    "pop": "20000 60 20",
    "seed": "20261015",
    "acelib": '"/global/home/groups/co_nuclear/serpent/xsdata/endfb7/sss_endfb7u.xsdata"',
    "declib": '"/global/home/groups/co_nuclear/serpent/xsdata/endfb7/sss_endfb7.dec"',
    "nfylib": '"/global/home/groups/co_nuclear/serpent/xsdata/endfb7/sss_endfb7.nfy"',
    "opti": "2", "gcu": "-1", "memfrac": "0.99", "repro": "0", "shbuf": "0 0", "outp": "100000",
    "nbuf": "1000", "depout": "3 1", "ures": "1", "inventory": "all", "printm": "1 1.0E-41",
    "power": "1.0E+06",
}


def write_geometry_inputs(case):
    with open(os.path.join(case, "zones.inp"), "w") as f:
        for r in RADIAL_BOUNDS:
            f.write(f"radial_channel\nz_divisions {AXIAL_SPLIT}\npoint 0 {r}\npoint 429 {r}\nend\n\n")
    count = 0
    with open(os.path.join(REPO, "gFHR_points.dat")) as fin, open(os.path.join(case, "points.dat"), "w") as fout:
        for line in fin:
            x, y, z = (float(v) for v in line.split()[:3])
            if np.hypot(x, y) < CLUSTER["r_max"] and CLUSTER["z_min"] < z < CLUSTER["z_max"]:
                fout.write(line)
                count += 1
    return count


def build_deck(case, out):
    core = Core("toy")
    core.define_zones(os.path.join(case, "zones.inp"), os.path.join(case, "points.dat"))
    for key in ("fuel20", "graphpeb"):
        core.fresh_pebbles[key] = Material(key, os.path.join(REPO, "materials", f"{key}.dat"))
    for key, name in (("SiC", "SiC_gFHR"), ("PyC", "PyC_gFHR"), ("buffer", "buffer_gFHR"), ("matrix", "matrix_gFHR")):
        core.static_fuel_materials[key] = Material(key, os.path.join(REPO, "materials", f"{name}.dat"))
    with open(os.path.join(REPO, "gFHR_template.serpent")) as f:
        core.core_geometry = f.read()
    rng = np.random.default_rng(int(SETTINGS["seed"]))
    for channel in core.zones:
        for zone in channel:
            zone.initialize({"graphpeb": 0.5, "fuel20": 0.5}, rng=rng)
            core.initialize_materials(zone.inventory)
    core.rng = rng
    cwd = os.getcwd()
    os.chdir(out)
    try:
        deck_name = core.generate_input(SETTINGS, 6.525, 0, cr_position=429.469)  # rods withdrawn
    finally:
        os.chdir(cwd)
    return core, os.path.join(out, deck_name), os.path.join(out, f"pebble_positions_{core.iteration}.csv")


def read_pebbles(path):
    xyz, mats = [], []
    with open(path) as f:
        for line in f:
            x, y, z, _, uni = line.split()
            xyz.append((float(x), float(y), float(z)))
            mats.append(uni[1:])  # "u<material>"
    return np.array(xyz), mats


def zone_of(material):
    return re.search(r"_(R\d+Z\d+)G", material).group(1)


def write_pebble_file(path, xyz, materials, radii):
    with open(path, "w") as f:
        for (x, y, z), mat, r in zip(xyz, materials, radii, strict=True):
            f.write(f"{x} {y} {z} {r} u{mat}\n")


def fuel_volumes(assignment):
    counts = {}
    for mat in assignment:
        if mat.startswith("fuel"):
            counts[mat] = counts.get(mat, 0) + 1
    return counts, {m: c * KERNEL_VOLUME_PER_PEBBLE for m, c in counts.items()}


def deck_for(deck_text, pebble_file, volumes, all_fuel):
    text = re.sub(r'pbed u_pb u_flibe "[^"]+"', f'pbed u_pb u_flibe "{pebble_file}"', deck_text)
    for mat in all_fuel:
        vol = volumes.get(mat, EMPTY_VOLUME)
        vol_str = np.format_float_scientific(vol, precision=8).replace("e", "E")
        text, n = re.subn(rf"^  {re.escape(mat)} 1 \S+$", f"  {mat} 1 {vol_str}", text, flags=re.M)
        if n != 1:
            raise ValueError(f"mvol line for {mat} not found exactly once")
    # Initial source guess confined to the cluster: the default fission-source sampling over the
    # whole core rarely hits the toy's fuel and Serpent aborts ("Unable to sample fission source").
    # src with srad/sz limits: docs/serpent/cards_reference.md §4 src (syntax/src.rst).
    r_max, z_min, z_max = CLUSTER["r_max"], CLUSTER["z_min"], CLUSTER["z_max"]
    return text + f"\n% initial source guess over the pebble cluster\nsrc toysrc srad 0.0 {r_max} sz {z_min} {z_max}\n"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("out")
    parser.add_argument("--scale-pebbles", type=int, default=0,
                        help="also build live_scale/ with this many pebbles in total (e.g. 3000000)")
    args = parser.parse_args(argv)
    out = os.path.abspath(args.out)
    case = os.path.join(out, "case")
    os.makedirs(case, exist_ok=True)
    n_points = write_geometry_inputs(case)
    core, deck_path, pebble_path = build_deck(case, case)
    with open(deck_path) as f:
        deck_text = f.read()
    triso_ids = sorted(set(re.findall(r'"trisos/triso(\d+)"', deck_text)), key=int)
    os.makedirs(os.path.join(out, "trisos"), exist_ok=True)
    with open(os.path.join(REPO, "gFHR_triso.dat")) as f:
        triso_lines = [line.split()[:4] for line in f if line.strip()]
    for i in triso_ids:
        with open(os.path.join(out, "trisos", f"triso{i}"), "w") as f:
            f.writelines(f"{x} {y} {z} {r} p{i}\n" for x, y, z, r in triso_lines)

    xyz, layout_a = read_pebbles(pebble_path)
    layout_b = list(layout_a)
    r1z1 = [i for i, m in enumerate(layout_a) if m == "fuel20_R1Z1G1"]
    for i in r1z1[: len(r1z1) // 2]:
        layout_b[i] = "graphpeb_R1Z1G1"
    layout_c = ["graphpeb_R2Z2G1" if m == "fuel20_R2Z2G1" else m for m in layout_a]
    layouts = {"A": layout_a, "B": layout_b, "C": layout_c}

    materials = sorted(set(layout_a) | set(layout_b) | set(layout_c))
    all_fuel = [m for m in materials if m.startswith("fuel")]
    by_zone = {}
    for m in materials:
        by_zone.setdefault(zone_of(m), []).append(m)
    pool_index, pool_xyz, pool_mat = [], [], []
    for p, (pos, mat) in enumerate(zip(xyz, layout_a, strict=True)):
        for m in by_zone[zone_of(mat)]:
            pool_index.append(p)
            pool_xyz.append(pos)
            pool_mat.append(m)
    pool_index, pool_xyz = np.array(pool_index), np.array(pool_xyz)

    manifest = {"points": n_points, "pebbles": len(xyz), "materials": materials, "fuel_materials": all_fuel,
                "pool_size": len(pool_mat), "pool_material": pool_mat, "pool_position": pool_index.tolist(),
                "layouts": {}, "settings": SETTINGS, "empty_volume": EMPTY_VOLUME,
                "burn_days": 6.525}
    for name, layout in layouts.items():
        counts, volumes = fuel_volumes(layout)
        radii = [2.0 if layout[p] == m else 0.0 for p, m in zip(pool_index, pool_mat, strict=True)]
        manifest["layouts"][name] = {"fuel_counts": counts, "fuel_volumes": volumes, "pool_radius": radii}
        ref = os.path.join(out, f"ref_{name}")
        os.makedirs(ref, exist_ok=True)
        write_pebble_file(os.path.join(ref, "layout.csv"), xyz, layout, [2.0] * len(layout))
        with open(os.path.join(ref, f"ref_{name}.serpent"), "w") as f:
            f.write(deck_for(deck_text, "layout.csv", volumes, all_fuel))
        link = os.path.join(ref, "trisos")
        if not os.path.lexists(link):
            os.symlink("../trisos", link)

    live = os.path.join(out, "live")
    os.makedirs(live, exist_ok=True)
    write_pebble_file(os.path.join(live, "pool.csv"), pool_xyz, pool_mat, manifest["layouts"]["A"]["pool_radius"])
    # No "set inventory" in live decks: the HxF build reads each inventory entry's name at the
    # nuclide-name offset while listing output variables (sendoutputvardefsmpilocal.c:106 vs
    # locations.h INVENTORY_PTR_NAME) and aborts with "GetText: Invalid pointer". The inventory only
    # selects _dep.m output; live compositions come from sss_ov_composition_*.
    live_text, n = re.subn(r"^set inventory .*\n", "", deck_text, flags=re.M)
    if n != 1:
        raise ValueError(f"expected one 'set inventory' line, found {n}")
    # Cumulative steps: the HxF MoveToTime (advance_to_time) supports only daytot/dectot/acttot
    # (butot is converted) and dies with "Unsupported step type 3" on daystep (movetotime.c:89-127).
    # One step of daytot T equals daystep T. dep modes: docs/serpent/cards_reference.md (dep).
    live_text, n = re.subn(r"^dep daystep ", "dep daytot ", live_text, flags=re.M)
    if n != 1:
        raise ValueError(f"expected one 'dep daystep' line, found {n}")
    with open(os.path.join(live, "live.serpent"), "w") as f:
        f.write(deck_for(live_text, "pool.csv", manifest["layouts"]["A"]["fuel_volumes"], all_fuel))

    if args.scale_pebbles:
        extra = args.scale_pebbles - len(pool_mat)
        if extra < 0:
            raise ValueError(f"--scale-pebbles must exceed the pool size {len(pool_mat)}")
        scale = os.path.join(out, "live_scale")
        os.makedirs(scale, exist_ok=True)
        # extra parked pebbles (r = 0) cycle through the pool entries, at in-core positions
        reps = np.arange(extra) % len(pool_mat)
        xyz_s = np.concatenate([pool_xyz, pool_xyz[reps]])
        mat_s = pool_mat + [pool_mat[i] for i in reps]
        radii_s = manifest["layouts"]["A"]["pool_radius"] + [0.0] * extra
        write_pebble_file(os.path.join(scale, "pool.csv"), xyz_s, mat_s, radii_s)
        shutil.copyfile(os.path.join(live, "live.serpent"), os.path.join(scale, "live.serpent"))
        manifest["scale_pebbles"] = len(mat_s)

    with open(os.path.join(out, "toy.json"), "w") as f:
        json.dump(manifest, f)
    counts = {k: v["fuel_counts"] for k, v in manifest["layouts"].items()}
    print(f"{len(xyz)} in-core pebbles, {len(materials)} materials, pool {len(pool_mat)}; fuel counts {counts}")


if __name__ == "__main__":
    sys.exit(main())
