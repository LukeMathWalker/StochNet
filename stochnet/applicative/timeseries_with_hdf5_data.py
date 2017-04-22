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
dataset.explode_into_training_pieces(1, filepath_for_saving='/home/lucap/Documenti/Data storage/SIR_dataset_medium_test.hdf5')
print(dataset.X_data.shape)
print(dataset.y_data.shape)
dataset.train_test_split(percentage_of_test_data=0.25, filepath_for_saving='/home/lucap/Documenti/Data storage/SIR_dataset_medium_test_2.hdf5')
print(dataset.X_train.shape)
print(dataset.X_test.shape)
print(dataset.y_train.shape)
print(dataset.y_test.shape)
