import numpy as np
from stochnet.classes.TimeSeriesDataset import NumpyTimeSeriesDataset
from stochnet.utils.hdf5 import convert_from_numpy_to_hdf5, convert_ML_dataset_from_numpy_to_hdf5

np_data_filepath = '/home/lucap/Documenti/Data storage/SIR/SIR_dataset_timestep_2-1_05.npy'
np_dataset = NumpyTimeSeriesDataset(np_data_filepath)
np_dataset.remove_timestamps()
np_data = np_dataset.data

filepath_for_saving_raw = '/home/lucap/Documenti/Data storage/SIR/SIR_dataset_timestep_2-1_05_no_timestamp.hdf5'
convert_from_numpy_to_hdf5(np_data, filepath_for_saving_raw)

# nb_past_timesteps = 1
# np_dataset.format_dataset_for_ML(nb_past_timesteps=nb_past_timesteps, must_be_rescaled=True, percentage_of_test_data=0)
# np_X_data = np_dataset.X_train
# np_y_data = np_dataset.y_train
#
# filepath_for_saving_ML = '/home/lucap/Documenti/Data storage/SIR_dataset_medium.hdf5'
# convert_ML_dataset_from_numpy_to_hdf5(np_X_data, np_y_data, filepath_for_saving_ML)
