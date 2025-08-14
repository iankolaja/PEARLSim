import numpy as np
from copy import deepcopy

TRANSPORT_NUCLIDES = []

def get_cross_section_string(temperature):
    xs_str = "."
    if 300 <= temperature < 600:
        xs_str += "03"
    if 600 <= temperature < 900:
        xs_str += "06"
    if 900 <= temperature < 1200:
        xs_str += "09"
    elif 1200 <= temperature < 1500:
        xs_str += "12"
    elif 1500 <= temperature < 1800:
        xs_str += "15"
    elif 1800 <= temperature:
        xs_str += "18"
    xs_str += "c"
    return xs_str


class Material():
    def __init__(self, name, source, temperature = 959, density=None,
                 rgb = (None, None, None)):
        self.concentrations = {}
        if type(source) == str:
            with open(source, 'r') as f:
                line = f.readline().replace("\n","").split(",")

                # Material is burnable and thus a sum of concentrations
                if line[0] == "burnable":
                    self.density = "sum fix"
                    self.density_function = None
                    self.is_burned_fuel = False

                # Density is provided as a float
                elif len(line) == 1:
                    self.density = float(line[0])
                    self.density_function = None
                    self.is_burned_fuel = False

                # Density function for material is provided
                else:
                    d0 = line[0]
                    b = line[1]
                    t_r = line[2]
                    self.density_function = lambda t : d0*(1+b*(t-t_r))
                    self.density = self.density_function(temperature)
                    self.is_burned_fuel = True

                # Read each concentration
                for line in f:
                    line = line.split(',')
                    self.concentrations[line[0]] = float(line[1])
        if type(source) == dict:
            self.concentrations = source
            self.density = "sum fix"
        if type(source) == Material:
            self.concentrations = deepcopy(source.concentrations)
            self.density = source.density
        self.name = name
        self.rgb = rgb
        self.fima = 0.0
        self.fima_last_pass = 0.0
        self.temperature = temperature
        self.cross_section_str = get_cross_section_string(temperature)
        if density is not None:
            self.density = density




    def __add__(self, other_material):
        new_material = deepcopy(self)
        for key in other_material.concentrations.keys():
            if key in new_material.concentrations.keys():
                new_material.concentrations[key] += other_material.concentrations[key]
            else:
                new_material.concentrations[key] = other_material.concentrations[key]
        return new_material

    def __mul__(self, coefficient):
        new_material = deepcopy(self)
        for key in self.concentrations.keys():
            new_material.concentrations[key] *= coefficient
        return new_material
    
    def fix_B11(self):
        conc_tot = 0.0
        if "50110" in self.concentrations.keys():
            conc_tot += self.concentrations.pop("50110")
        if "5011<lib>" in self.concentrations.keys():
            conc_tot += self.concentrations.pop("5011<lib>")
        self.concentrations["5011<lib>"] = conc_tot

    def merge_duplicates(self, debug):
        keys_to_remove = []
        for key in self.concentrations.keys():
            if key[-1] == '0':
                new_key = key[:-1] + "<lib>"
                if new_key in self.concentrations.keys() and new_key != key:
                    self.concentrations[new_key] += self.concentrations[key]
                    keys_to_remove += [key]
                    if debug > 1:
                        print(f"Merging {key} into {new_key} in {self.name}")
        for key in keys_to_remove:
            self.concentrations.pop(key)
        return keys_to_remove

    def write_input(self, triso_counter, static_fuel_mats, debug, volume=None, never_burn=False):
        self.cross_section_str = get_cross_section_string(self.temperature)
        self.fix_B11()
        self.merge_duplicates(debug)
        if self.density == "sum fix":
            density_s = f"sum fix {self.cross_section_str[1:]} {round(self.temperature,3)}"
        else:
            density_s = self.density
        if debug >= 3:
            print(f"Writing material {self.name}")
        if None in self.rgb:
            r = int(self.name.split("R")[1].split("Z")[0])*32+100
            g = -int(self.name.split("Z")[1].split("G")[0])*15+250
            b = int(self.name.split("G")[1])*75
            rgb = (r%255, g%255, b%255)
        else:
            rgb = self.rgb
        input_s = f"mat {self.name} {density_s} rgb {rgb[0]} {rgb[1]} {rgb[2]}  tmp {round(self.temperature,3)}"
        if "graph" in self.name:
            input_s += " moder  grph 6000\n"
        elif volume is not None and not never_burn:
            input_s += f" burn 1 vol {volume}\n"
        elif volume is not None:
            input_s += f" vol {volume}\n"
        else:
            input_s += " burn 1 \n"
        for key in self.concentrations.keys():
            #if self.concentrations[key] > 0:
            conc_str = f"  {key}    {self.concentrations[key]}\n".replace("<lib>", self.cross_section_str)
            input_s += conc_str
        input_s += "\n"

        if "fuel" in self.name:
            for name in static_fuel_mats.keys():
                other_mat = static_fuel_mats[name]
                input_s += f"mat {self.name}_{name} {other_mat.density} rgb {rgb[0]} " \
                           f"{rgb[1]} " \
                           f"{rgb[2]} " \
                           f"tmp {round(self.temperature,3)} moder  grph 6000\n"
                for key in other_mat.concentrations.keys():
                    conc_str = f"  {key}    {other_mat.concentrations[key]}\n".replace("<lib>", self.cross_section_str)
                    input_s += conc_str
                input_s += "\n"

        # Write geometry for solid graphite pebble
        if "graph" in self.name:
            input_s += f"cell {self.name}_c u{self.name} {self.name} -pebble_outer\n"

        # Write geometry for fuel pebble + associated definitions
        elif "fuel" in self.name:
            input_s += f"particle p{triso_counter}\n"
            input_s += f"{self.name}	0.02125\n"
            input_s += f"{self.name}_buffer    0.031250\n"
            input_s += f"{self.name}_PyC    0.035250\n"
            input_s += f"{self.name}_SiC    0.038750\n"
            input_s += f"{self.name}_PyC    0.042750\n"
            input_s += f"{self.name}_matrix\n\n"

            input_s += f"cell inf_{self.name} infU_{self.name} {self.name}_matrix -infinite\n"
            input_s += f"pbed trisobed{triso_counter} infU_{self.name} \"trisos/triso{triso_counter}\"\n"
            input_s += f"cell {self.name}_c1 u{self.name} pebgraph -pebble_core_outer\n"
            input_s += f"cell {self.name}_c2 u{self.name} fill trisobed{triso_counter} pebble_core_outer -matrix_outer\n"
            input_s += f"cell {self.name}_c3 u{self.name} pebshell  matrix_outer -pebble_outer\n"
        input_s += "\n\n"
        return input_s