from stochnet.utils.hdf5 import concatenate_hdf5_datasets
import numpy as np
import h5py

final_dataset_address = '/home/lucap/Documenti/Data storage/SIR/timestep_2-5_dataset_v_big_01_w_split_no_rescale.hdf5'

x_dummy_data = np.ones((1, 1, 3))
y_dummy_data = np.ones((1, 3))
# FIX
f = h5py.File(final_dataset_address, 'a', libver='latest')
f.create_dataset('X_train', data=x_dummy_data, maxshape=(None, 1, 3))
f.create_dataset('y_train', data=y_dummy_data, maxshape=(None, 3))
f.create_dataset('X_test', data=x_dummy_data, maxshape=(None, 1, 3))
f.create_dataset('y_test', data=y_dummy_data, maxshape=(None, 3))

for i in range(2, 7):
    dataset_address = '/home/lucap/Documenti/Data storage/SIR/timestep_2-5_dataset_big_0' + str(i) + '_w_split_no_rescale.hdf5'
    concatenate_hdf5_datasets(final_dataset_address, dataset_address, X_label_1='X_train', X_label_2='X_train', y_label_1='y_train', y_label_2='y_train')
