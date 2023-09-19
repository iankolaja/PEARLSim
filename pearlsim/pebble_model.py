from .material import Material
import random
from .core import get_zone


class Pebble():
    def __init__(self, r, z, material_source, id=random.randint(1,2147483647)):
        self.r = r
        self.z = z
        self.id = id
        self.pass_num = 1
        self.intial_fuel = material_source
        self.material = Material(f"pebble{id}", material_source)



class Pebble_Model():
    def __init__(self, model_type, load_file_path):
        self.pebbles = []
        self.model_type = model_type
        self.discarded_pebbles = []
        #self.model = pickle.loads(load_file_path)

    def distribute_pebbles(self, num_pebbles, distribution, core_fresh_materials, material_fractions, debug=0):
        if "function" in str(type(distribution)):
            r_vals, z_vals = distribution(num_pebbles)
        else:
            indices = random.choices(range(len(distribution)), k=num_pebbles)
            r_vals = distribution.iloc[indices]['r'].values
            z_vals = distribution.iloc[indices]['z'].values
        if debug > 0:
            print(f"Generating {num_pebbles} pebbles for individual pebble model.")
        for i in range(num_pebbles):
            r = r_vals[i]
            z = z_vals[i]
            mat_name = random.choices(list(material_fractions.keys()), weights=material_fractions.values(), k=1)[0]
            material = core_fresh_materials[mat_name]
            self.spawn_pebble(r, z, material, debug=debug)


    def spawn_pebble(self, r, z, material_source, debug=0):
        if debug > 1:
            print(f"Generating {material_source.name} pebble at r = {r}, z = {z}")
        self.pebbles += [Pebble(r, z, material_source)]

    def remove_spent_pebbles(self, radial_bound_points, axial_zone_bounds, threshold, graphite_removal_flag):
        """
        Check all invividual pebbles in top zones to see what fraction of them should be removed.
        :param radial_bound_points:
        :param axial_zone_bounds:
        :param threshold:
        :return:
        """
        pebbles_to_remove_by_zone = {} # How many pebbles were past threshold by zone/pass
        pebbles_modeled_by_zone = {} # How many pebbles exist in model by zone/pass
        self.discarded_pebbles = {}
        removal_fractions = {} # Resulting fraction of pebbles to remove
        indices_to_remove = [] # Resulting indices of pebbles to delete
        for i in range(len(self.pebbles)):
            pebble = self.pebbles[i]
            rad_zone, axial_zone = get_zone(pebble.r, pebble.z, radial_bound_points, axial_zone_bounds)

            # Check if the pebble is in a top-most zone
            if axial_zone == len(axial_zone_bounds[rad_zone-1])-1:
                zone_material_name = f"{pebble.intial_fuel}_R{rad_zone}Z{axial_zone}P{pebble.pass_num}"
                if zone_material_name in pebbles_modeled_by_zone.keys():
                    pebbles_modeled_by_zone[zone_material_name] += 1
                else:
                    pebbles_modeled_by_zone[zone_material_name] = 1

                # Check if pebble's cesium concentration is above threshold
                if "55137" in pebble.material.concentrations.keys():
                    if pebble.material.concentrations["55137"] > threshold:
                        if zone_material_name in pebbles_to_remove_by_zone.keys():
                            pebbles_to_remove_by_zone[zone_material_name] += 1
                        else:
                            pebbles_to_remove_by_zone[zone_material_name] = 1
                        indices_to_remove += [i]

                if graphite_removal_flag:
                    if "pebgraph" in pebble.material.name:
                        if zone_material_name in pebbles_to_remove_by_zone.keys():
                            pebbles_to_remove_by_zone[zone_material_name] += 1
                        else:
                            pebbles_to_remove_by_zone[zone_material_name] = 1
                        indices_to_remove += [i]

        # Delete pebbles with thresholds that are too high
        for i in sorted(indices_to_remove, reverse=True):
            self.discarded_pebbles += self.pebbles[i]
            del self.pebbles[i]

        # Calculate fractions of pebbles to remove from the zone model
        for key in pebbles_modeled_by_zone.keys():
            if key in pebbles_to_remove_by_zone.keys():
                removal_fractions[key] = pebbles_to_remove_by_zone[key]/pebbles_modeled_by_zone[key]
            else:
                removal_fractions[key] = 0
        return (removal_fractions, len(indices_to_remove))

