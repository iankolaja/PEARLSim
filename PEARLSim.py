from pearlsim.simulation import Simulation
import getopt, sys

# Remove 1st argument from the
# list of command line arguments
argumentList = sys.argv[1:]

# Options
options = "fcnd:"

# Long options
long_options = ["file", "cores", "nodes", "debug"]

simulation_name = "gFHR_equilibrium"
simulation = Simulation(simulation_name)

num_cores = 20

try:
    # Parsing argument
    arguments, values = getopt.getopt(argumentList, options, long_options)

    # checking each argument
    for currentArgument, currentValue in arguments:

        if currentArgument in ("-f", "--file"):
            print("Reading input file:", sys.argv[1])
            input_file = sys.argv[1]

        elif currentArgument in ("-c", "--cores"):
            print("Number of cores:", sys.argv[1])
            cores = int(sys.argv[1])
            simulation.cpu_cores = cores

        elif currentArgument in ("-n", "--nodes"):
            print("Number of nodes:", sys.argv[1])
            nodes = int(sys.argv[1])
            simulation.num_nodes = nodes

        elif currentArgument in ("-d", "--debug"):
            print("Setting debug level:", sys.argv[1])
            simulation.debug = int(sys.argv[1])

except getopt.error as err:
    # output error, and return with an error code
    print(str(err))

simulation.read_input_file(input_file)