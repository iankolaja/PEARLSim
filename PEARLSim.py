from pearlsim.simulation import Simulation
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-f', '--file')
parser.add_argument('-w', '--workingdir')
parser.add_argument('-c', '--cores')
parser.add_argument('-n', '--nodes')
parser.add_argument('-d', '--debug')
args = parser.parse_args()
simulation_name = "gFHR_equilibrium"
simulation = Simulation(simulation_name)
directory = None

args = parser.parse_args()
 
if args.file:
    print("Reading input file:", args.file)
    input_file = args.file
if args.workingdir:
    print("Working in directory", args.workingdir)
    directory = args.workingdir
if args.cores:
    print("Setting number of cores", args.cores)
    simulation.cpu_cores = int(args.cores)
if args.nodes:
    print("Setting number of nodes", args.nodes)
    simulation.num_nodes = int(args.nodes)
if args.debug:
    print("Setting debug level", args.debug)
    simulation.debug = int(args.debug)

simulation.read_input_file(input_file, directory_name=directory)
