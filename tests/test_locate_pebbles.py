from pearlsim import *
import unittest
import os

class TestDatabaseParsing(unittest.TestCase):
    def test_database_parsing(self):
        working_directory = "gFHR_equilibrium"
        test_steps = [1,2,3]
        
        zone_def_file = "gFHR_zones.inp"
        # TODO: Read zone file
        
        os.chdir(f"{working_directory}")
        for i in test_steps:
            file_name = f"pebble_positions_{i}.csv"
            with open(file_name, 'r') as f:
                for line in f:
                    print(line)
        # TODO: Load zone_map for iteration i
        # TODO: Load create zone structure using the pebble positions above
        # TODO: Assert that recreated and loaded zone structures are the same

if __name__ == '__main__':
    unittest.main()
