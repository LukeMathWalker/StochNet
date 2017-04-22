import os
from stochnet.classes.TimeSeriesDataset import TimeSeriesDataset

current = os.getcwd()
working_path = os.path.dirname(current)
basename = os.path.abspath(working_path)

dataset_address = '/home/lucap/Documenti/Data storage/SIR_dataset_medium_raw.hdf5'

dataset = TimeSeriesDataset(dataset_address=dataset_address, data_format='hdf5')
nb_past_timesteps = 1
dataset.rescale(positivity=None)
