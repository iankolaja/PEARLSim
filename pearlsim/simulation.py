"""Keyword-deck driver for PEARLSim simulations.

An input deck is a sequence of keyword lines (``#`` starts a comment). Each
keyword maps to a ``_kw_<name>`` handler; unknown keywords are logged and
ignored. Paths in a deck are relative to the directory containing the deck:
``read_input_file`` changes into a run directory one level below it, so the
handlers resolve input paths as ``"../" + path``.
"""

import logging
import os
import pickle
import random
import shlex
import subprocess

import numpy as np
import pandas as pd

from pearlsim.core import Core
from pearlsim.material import Material

logger = logging.getLogger(__name__)


class SerpentError(RuntimeError):
    """Serpent exited with an error or did not produce the expected outputs."""


class DeckError(ValueError):
    """An input-deck line could not be interpreted."""


class Simulation:
    def __init__(self, simulation_name):
        self.simulation_name = simulation_name
        self.core = Core(simulation_name)
        self.cpu_cores = 20
        self.num_nodes = 1
        self.days = 0
        self.power = 280e6
        self.serpent_command = "sss2_HxF_dev"
        self.rod_insertion = 429.469  # practically fully withdrawn
        self.burnup_time_step = 6.525  # days
        self.fima_discharge_threshold = 17
        self.fima_discharge_rel_std = 0.02576
        self.debug = 0
        self.serpent_settings = {"pop": "10000 50 25"}

    # ------------------------------------------------------------------ deck parsing

    def get_zone_model_materials(self):
        return self.core.materials

    def get_zone_model_pebble_locations(self):
        return self.core.pebble_locations

    def read_input_file(self, input_file, directory_name=None):
        if directory_name is None:
            directory_name = os.path.splitext(os.path.basename(input_file))[0]
        with open(input_file) as f:
            lines = f.readlines()
        os.makedirs(directory_name, exist_ok=True)
        os.chdir(directory_name)
        for line in lines:
            self.read_input_line(line)

    def read_input_line(self, raw_line):
        tokens = raw_line.split("#", 1)[0].split()
        if not tokens:
            return
        keyword, args = tokens[0], tokens[1:]
        handler = getattr(self, f"_kw_{keyword}", None)
        if handler is None:
            logger.warning("Unknown keyword %r ignored", keyword)
            return
        try:
            handler(args)
        except (IndexError, ValueError) as error:
            raise DeckError(f"Invalid deck line {raw_line.strip()!r}: {error}") from error

    # ------------------------------------------------------------------ keyword handlers

    def _kw_restart(self, args):
        with open("../" + args[0], "rb") as f:
            self.core = pickle.load(f)
        logger.info("Restarted core from %s", args[0])

    def _kw_fresh(self, args):
        for key, path in _pairs(args):
            self.core.fresh_pebbles[key] = Material(key, "../" + path)

    def _kw_triso_file(self, args):
        num_copies, path = int(args[0]), args[1]
        triso_data = pd.read_csv("../" + path, sep=" ", names=["x", "y", "z", "r", "id"])
        if not os.path.exists("trisos"):
            os.mkdir("trisos")
            for i in range(1, 1 + num_copies):
                triso_data["id"] = f"p{i}"
                triso_data.to_csv(f"trisos/triso{i}", sep=" ", header=False, index=False)

    def _kw_static_fuel_materials(self, args):
        for key, path in _pairs(args):
            self.core.static_fuel_materials[key] = Material(key, "../" + path)

    def _kw_set_seed(self, args):
        seed = int(args[0])
        random.seed(seed)
        np.random.seed(seed)
        logger.info("Set random seed to %d", seed)

    def _kw_set_substep_schema(self, args):
        self.core.substep_schema = [float(v) for v in args]

    def _kw_debug_level(self, args):
        self.debug = int(args[0])

    def _kw_define_zones(self, args):
        self.core.define_zones("../" + args[0], "../" + args[1], debug=self.debug)

    def _kw_core_geometry(self, args):
        path = args[0].replace('"', "")
        with open("../" + path) as f:
            self.core.core_geometry = f.read()

    def _kw_initialize(self, args):
        # args[0] was the number of individually modelled pebbles (removed pebble model)
        insertion_ratios = {key: float(value) for key, value in _pairs(args[1:])}
        for channel in self.core.zones:
            for zone in channel:
                zone.initialize(insertion_ratios, debug=self.debug)
                self.core.initialize_materials(zone.inventory)

    def _kw_insertion(self, args):
        num_steps = int(args[0])
        get_boolean(args[1])  # thermal-coupling flag; not implemented
        serpent_flag = int(args[2])
        insertion_ratios, insertion_dict = get_insertion_ratio_pairs(args[3:])
        for _ in range(num_steps):
            self.core.insert(insertion_ratios, self.fima_discharge_threshold,
                             self.fima_discharge_rel_std, debug=self.debug)
            self.run_step(serpent_flag, insertion_dict.get("graphpeb", 0.0))

    def _kw_transport(self, args):
        serpent_flag = int(args[0])
        last_step = len(self.core.substep_schema)
        if serpent_flag == 1:
            self.run_step(1, "static")
        elif serpent_flag == 2:
            self.core.update_from_bumat(self.debug, step=last_step)
            self.days += self.burnup_time_step
            self.core.iteration += 1
        elif serpent_flag == 3:
            self.core.update_from_bumat(self.debug, step=last_step)
            self.days += self.burnup_time_step
            self.core.generate_input(self.serpent_settings, self.burnup_time_step, self.debug,
                                     cr_position=self.rod_insertion)
            self.core.save_zone_maps(f"zone_map{self.core.iteration}.json")
            self.core.iteration += 1
        else:
            raise ValueError(f"unknown transport flag {serpent_flag}")

    def _kw_set_averaging_mode(self, args):
        if args[0] not in ("pass", "burnup"):
            raise ValueError(f"averaging mode must be 'pass' or 'burnup', not {args[0]!r}")
        self.core.averaging_mode = args[0]

    def _kw_set_rod_depth(self, args):
        self.rod_insertion = float(args[0])

    def _kw_set_discharge_threshold(self, args):
        self.fima_discharge_threshold = float(args[0])

    def _kw_set_threshold_rel_std(self, args):
        self.fima_discharge_rel_std = float(args[0])

    def _kw_set_num_fuel_groups(self, args):
        self.core.fuel_groups = int(args[0])

    def _kw_set_burnup_time_step(self, args):
        self.burnup_time_step = float(args[0])

    def _kw_set_power(self, args):
        self.power = float(args[0])
        self.serpent_settings["power"] = args[0]

    _kw_power = _kw_set_power

    def _kw_set_max_passes(self, args):
        self.core.max_passes = int(args[0])

    def _kw_load_from_step(self, args):
        self.load_from_step(int(args[0]), int(args[1]) if len(args) > 1 else 0)

    def _kw_save(self, args):
        with open(f"{self.simulation_name}_core.pkl", "wb") as f:
            pickle.dump(self.core, f)
        with open(f"{self.simulation_name}_simulation.pkl", "wb") as f:
            pickle.dump(self, f)

    def _kw_set(self, args):
        setting, value = args[0], " ".join(args[1:])
        if value == "clear":
            self.serpent_settings.pop(setting, None)
            return
        self.serpent_settings[setting] = value
        if setting == "power":
            self.power = float(value)

    # ------------------------------------------------------------------ steps

    def operating_parameters(self, graphite_insertion_fraction):
        return {"power": float(self.power),
                "burnup_step": self.burnup_time_step,
                "graphite_insertion_fraction": graphite_insertion_fraction,
                "control_rod_position": self.rod_insertion,
                "fima_discharge_threshold": self.fima_discharge_threshold,
                "fima_discharge_rel_std": self.fima_discharge_rel_std}

    def run_step(self, serpent_flag, graphite_insertion_fraction):
        """Advance one depletion step after the core inventory has been updated.

        serpent_flag: 0 = bookkeeping only (no Serpent); 1 = run Serpent;
        2 = reuse existing Serpent results for this step if present, otherwise run.
        """
        iteration = self.core.iteration
        last_step = len(self.core.substep_schema)
        bumat_name = f"{self.simulation_name}_{iteration}.serpent.bumat{last_step}"
        zone_file_name = f"zone_map{iteration}.json"
        operating_file = f"{self.core.simulation_name}_operating_{iteration}.json"

        if serpent_flag == 0:
            if os.path.isfile(zone_file_name):
                logger.info("%s exists; skipping step %d", zone_file_name, iteration)
            else:
                self.core.save_zone_maps(zone_file_name)
                self.core.save_all_materials(f"core_materials_{iteration}.json.gz")
            self.core.iteration += 1
            return
        if serpent_flag == 2 and os.path.isfile(bumat_name):
            logger.info("%s exists; reusing Serpent results for step %d", bumat_name, iteration)
            self.core.save_operating_parameters(operating_file,
                                                self.operating_parameters(graphite_insertion_fraction))
        elif serpent_flag in (1, 2):
            input_name = self.core.generate_input(self.serpent_settings, self.burnup_time_step, self.debug,
                                                  cr_position=self.rod_insertion)
            self.core.save_operating_parameters(operating_file,
                                                self.operating_parameters(graphite_insertion_fraction))
            self.run_serpent(input_name)
            if not os.path.isfile(bumat_name):
                raise SerpentError(f"Serpent finished without writing {bumat_name}")
            self.core.save_zone_maps(zone_file_name)
            self.core.save_all_materials(f"core_materials_{iteration}.json.gz")
        else:
            raise ValueError(f"unknown Serpent flag {serpent_flag}")
        self.core.update_from_bumat(self.debug, step=last_step)
        self.days += self.burnup_time_step
        self.core.iteration += 1

    def load_from_step(self, load_iteration, burnup_step, debug=0):
        for channel in self.core.zones:
            for zone in channel:
                zone.initialize({"graphpeb": 1}, debug=self.debug)
        self.core.iteration = load_iteration
        self.core.load_zone_maps(f"zone_map{load_iteration}.json")
        self.core.load_pebble_locations(f"pebble_positions_{load_iteration}.csv", debug=self.debug)
        self.core.update_from_bumat(self.debug, iteration=load_iteration, step=burnup_step)

    def run_serpent(self, input_name):
        command = shlex.split(self.serpent_command)
        if self.num_nodes > 1:
            command = (["mpirun", "-np", str(self.num_nodes), "--map-by", f"ppr:1:node:pe={self.cpu_cores}"]
                       + command + ["-omp", str(self.cpu_cores), input_name])
        else:
            command = command + [input_name, "-omp", str(self.cpu_cores)]
        logger.info("Running %s", " ".join(command))
        result = subprocess.run(command)
        if result.returncode != 0:
            raise SerpentError(f"{' '.join(command)} exited with status {result.returncode}")


def _pairs(tokens):
    if len(tokens) % 2:
        raise ValueError(f"expected name/value pairs, got {tokens}")
    return list(zip(tokens[0::2], tokens[1::2]))


def get_boolean(input_argument):
    if input_argument in ("1", "true", "True", "yes", "Yes", "on"):
        return True
    if input_argument in ("0", "false", "False", "no", "No", "off"):
        return False
    raise ValueError(f"invalid flag value {input_argument!r}")


def get_insertion_ratio_pairs(list_of_definitions):
    pairs = [(key, float(value)) for key, value in _pairs(list_of_definitions)]
    return pairs, dict(pairs)
