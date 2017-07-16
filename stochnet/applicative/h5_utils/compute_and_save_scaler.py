from stochnet.classes.TimeSeriesDataset import H5TimeSeriesDataset
import dill
import os

nb_past_timesteps = 1
for i in range(6, 7):
    dataset_address = '/home/lucap/Documenti/Data storage/SIR/timestep_2-5_dataset_big_0' + str(i) + '.hdf5'
    scaler_address = '/home/lucap/Documenti/Data storage/SIR/scaler_for_timestep_2-5_dataset_big_0' + str(i) + '.h5'
    dataset = H5TimeSeriesDataset(dataset_address=dataset_address)

    filepath_for_saving_no_split = 'temp.hdf5'
    filepath_for_saving_w_split = 'temp2.hdf5'
    dataset.format_dataset_for_ML(nb_past_timesteps=nb_past_timesteps, must_be_rescaled=True,
                                  percentage_of_test_data=0.0,
                                  filepath_for_saving_no_split=filepath_for_saving_no_split,
                                  filepath_for_saving_w_split=filepath_for_saving_w_split)
    with open(scaler_address, 'wb') as f:
        dill.dump(dataset.scaler, f)
    os.remove('temp.hdf5')
    os.remove('temp2.hdf5')
