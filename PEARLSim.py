from pearlsim.simulation import Simulation
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-f', '--file')
parser.add_argument('-c', '--cores')
parser.add_argument('-n', '--nodes')
parser.add_argument('-d', '--debug')
parser.add_argument('-v', dest='verbose', action='store_true')
args = parser.parse_args()
simulation_name = "gFHR_equilibrium"
simulation = Simulation(simulation_name)


args = parser.parse_args()
 
if args.file:
    print("Reading input file:", args.file)
    input_file = args.file
if args.cores:
    print("Setting number of cores", args.cores)
    simulation.num_cores = args.cores
if args.nodes:
    print("Setting number of nodes", args.nodes)
    simulation.num_nodes = args.nodes
if args.debug:
    print("Setting debug level", args.debug)
    simulation.debug = args.debug

simulation.read_input_file(input_file)
