import os
import glob
import tarfile
import subprocess

def run_command(file_name):
    """Function to execute a single shell command."""
    file = f"{file_name}.tar.gz"
    if os.path.exists(file):
        print(file)
        with open(file, "rb") as f:
            f.seek(-65536, os.SEEK_END)
            tail = f.read()
        result = subprocess.run(["gzip", "-t"], input=tail, capture_output=True)
        if result.returncode != 0:
            print(f"CORRUPT: {file}")


automated_files = glob.glob("*automated*")
sequence_files = glob.glob("*sequence*")
random_files = glob.glob("*random*")
ascension_files = glob.glob("*ascension*")
merged_files = automated_files+sequence_files+random_files+ascension_files
filtered_files = []
for file_path in merged_files:
    if "." not in file_path:
        filtered_files.append(file_path)
for file_path in filtered_files:
    run_command(file_path)
