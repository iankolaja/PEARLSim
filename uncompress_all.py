import os



import os
from multiprocessing import Pool

def run_command(file_name):
    """Function to execute a single shell command."""
    print(f"Executing: Unpigz {file_name}")
    os.system(f"unpigz {file_name}.tar.gz")
    print(f"Finished: Unpigz {file_name}")
    print(f"Executing: tar {file_name}")
    os.system(f"tar -xvf {file_name}.tar")
    print(f"Finished: tar {file_name}")

if __name__ == "__main__":
    file_names = ["gFHR_sequence7",
        "gFHR_sequence8",
        "gFHR_sequence9",
        "gFHR_sequence10",
        "gFHR_sequence11",
        "gFHR_sequence12",
        "gFHR_sequence13",
        "gFHR_sequence14",
        "gFHR_forecast_base",
        "gFHR_random_1",
        "gFHR_random_2",
        "gFHR_random_3",
        "gFHR_random_4",
        "gFHR_random_5",
        "gFHR_random_6",
        "gFHR_random_7",
        "gFHR_random_8",
        "gFHR_random_9",
        "gFHR_random_10",
        "gFHR_random_11",
        "gFHR_random_12",
        "gFHR_random_13",
        "gFHR_random_14",
        "gFHR_random_15",
        "gFHR_random_16",
        "gFHR_random_18",
        "gFHR_random_19",
        "gFHR_random_20"]

    file_names = ['gFHR_forecast_power_up', 'gFHR_forecast_power_down', 'gFHR_forecast_rods_in', 'gFHR_forecast_threshold_down', 'gFHR_forecast_circulation_slower', 'gFHR_forecast_graphite_up']

    # Create a Pool of worker processes
    # The number of processes in the pool can be adjusted (e.g., os.cpu_count())
    with Pool(processes=12) as pool:
        # Map the run_command function to each command in the list
        # This will execute the commands in parallel
        pool.map(run_command, file_names)

    print("All commands have been executed.")