import h5py
import numpy as np
np.set_printoptions(precision=3, suppress=True)

f = h5py.File('/home/lucap/Documenti/Data storage/SIR/timestep_2-5_dataset_v_big_01_w_split_no_rescale.hdf5', 'r')
print('Using .values():')
print(list(f.values()))
print('Using .keys():')
print(list(f.keys()))
X_dst = f['X_train']
print('X data shape:')
print(X_dst.shape)
y_dst = f['y_train']
print('y data shape:')
print(y_dst.shape)
print('First 10^2 X datapoints:')
print(X_dst[:100, ...])
print('First 10^2 y datapoints:')
print(y_dst[:100, ...])
