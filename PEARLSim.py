from pearlsim.simulation import Simulation
import getopt, sys

# Remove 1st argument from the
# list of command line arguments
argumentList = sys.argv[1:]

# Options
options = "hmo:"

# Long options
long_options = ["Help", "My_file", "Output="]

simulation = Simulation()

input_file = "gFHR_equilibrium.inp"
num_cores = 20

try:
    # Parsing argument
    arguments, values = getopt.getopt(argumentList, options, long_options)

    # checking each argument
    for currentArgument, currentValue in arguments:

        if currentArgument in ("-f", "--file"):
            print("Reading input file:", sys.argv[0])
            input_file = sys.argv[0]

        elif currentArgument in ("-c", "--cores"):
            print("Number of cores:", sys.argv[0])
            cores = int(sys.argv[0])
            simulation.cpu_cores = cores

        elif currentArgument in ("-d", "--debug"):
            print("Setting debug level:", sys.argv[0])
            simulation.debug = int(sys.argv[0])

except getopt.error as err:
    # output error, and return with an error code
    print(str(err))

simulation.read_input_file(input_file)