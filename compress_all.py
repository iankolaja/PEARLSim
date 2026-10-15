import os
import glob

def run_command(file_name):
    """Function to execute a single shell command."""
    print(f"Compressing: {file_name}")
    #if not os.path.exists(f"{file_name}.tar.gz"):
    print(f"tar cf - {file_name} | pigz > {file_name}.tar.gz")
    os.system(f"tar cf - {file_name} | pigz > {file_name}.tar.gz")
    print(f"Finished: Compressing {file_name}. Moving...")
    os.system(f"mv {file_name} raw_pearlsim_data")
    print("Moved.")
    #else:
    #    print(f"Skipping: {file_name}")

automated_files = glob.glob("*automated*")
sequence_files = glob.glob("*sequence*")
random_files = glob.glob("*random*")
ascension_files = glob.glob("*ascension*")
runin_files = glob.glob("*runin*")
forecast_files = glob.glob("*forecast*")
merged_files = automated_files+sequence_files+random_files+ascension_files+runin_files+forecast_files
filtered_files = []
for file_path in merged_files:
    if "." not in file_path:
        filtered_files.append(file_path)
for file_path in filtered_files:
    run_command(file_path)
