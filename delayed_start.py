import os
import time

for i in range(90*60):
   print(i)
   time.sleep(1)

if os.path.isfile("ml_models/core_state_lstm.pkl"):
    os.system("sbatch gFHR_automated_10s.sub")
    os.system("sbatch gFHR_automated_20s.sub")
    os.system("sbatch gFHR_automated_40s.sub")
    os.system("sbatch gFHR_automated_100s.sub")

