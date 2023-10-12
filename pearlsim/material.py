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
    def __init__(self, name, source, temperature = 959, pass_num = None, density=None,
                 rgb = (np.random.randint(50,150), np.random.randint(50,150), np.random.randint(50,150))):
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
        self.pass_num = pass_num
        self.rgb = rgb
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


    def write_input(self, triso_counter, static_fuel_mats, debug, volume=None, never_burn=False):
        self.cross_section_str = get_cross_section_string(self.temperature)
        if self.density == "sum fix":
            density_s = f"sum fix {self.cross_section_str[1:]} {round(self.temperature,3)}"
        else:
            density_s = self.density
        if debug >= 3:
            print(f"Writing material {self.name}")
        input_s = f"mat {self.name} {density_s} rgb {self.rgb[0]} {self.rgb[1]} {self.rgb[2]}  tmp {round(self.temperature,3)}"
        if "graph" in self.name:
            input_s += " moder  grph 6000\n"
        elif volume is not None and not never_burn:
            input_s += f" burn 1 vol {volume}\n"
        elif volume is not None:
            input_s += f" vol {volume}\n"
        else:
            input_s += " burn 1 \n"
        for key in self.concentrations.keys():
            if self.concentrations[key] > 0:
                conc_str = f"  {key}    {self.concentrations[key]}\n".replace("<lib>", self.cross_section_str)
                input_s += conc_str
        input_s += "\n"

        if "fuel" in self.name:
            rgb_modifier = 0
            for name in static_fuel_mats.keys():
                other_mat = static_fuel_mats[name]
                rgb_modifier += 1
                input_s += f"mat {self.name}_{name} {other_mat.density} rgb {other_mat.rgb[0]+rgb_modifier*10} " \
                           f"{other_mat.rgb[1]+rgb_modifier*10} " \
                           f"{other_mat.rgb[2]+rgb_modifier*10} " \
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