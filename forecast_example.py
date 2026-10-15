from pearlsim.ml_utilities import *

features = pd.read_csv("sequence_data/gFHR_sequence_inputs.csv.gz", index_col=0)
targets = pd.read_csv("sequence_data/gFHR_sequence_outputs.csv.gz", index_col=0)
print(create_forecast_dataset(features, targets))