from stochnet.utils.hdf5 import concatenate_hdf5_datasets
import numpy as np
import h5py

final_dataset_address = '/home/lucap/Documenti/Data storage/SIR/timestep_2-1_dataset_01+02+03+04_no_timestamp.hdf5'

dummy_data = np.ones((1, 11, 3))

f = h5py.File(final_dataset_address, 'a', libver='latest')
f.create_dataset('data', data=dummy_data, maxshape=(None, 11, 3))

for i in range(1, 5):
    dataset_address = '/home/lucap/Documenti/Data storage/SIR/SIR_dataset_timestep_2-1_0' + str(i) + '_no_timestamp.hdf5'
    concatenate_hdf5_datasets(final_dataset_address, dataset_address, label_1='data', label_2='data')
