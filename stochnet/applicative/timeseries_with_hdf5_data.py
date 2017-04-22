import os
from stochnet.classes.TimeSeriesDataset import TimeSeriesDataset

current = os.getcwd()
working_path = os.path.dirname(current)
basename = os.path.abspath(working_path)

dataset_address = '/home/lucap/Documenti/Data storage/SIR_dataset_medium_raw.hdf5'

dataset = TimeSeriesDataset(dataset_address=dataset_address, data_format='hdf5')
nb_past_timesteps = 1
dataset.rescale(positivity=None)
print(dataset.scaler)
print(dataset.scaler.mean_)
print(dataset.scaler.scale_)
dataset.explode_into_training_pieces(1, mode='hdf5', filepath_for_saving='/home/lucap/Documenti/Data storage/SIR_dataset_medium_test.hdf5')
dataset.train_test_split(percentage_of_test_data=0.25)
