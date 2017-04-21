import h5py


f = h5py.File('/home/lucap/Documenti/Data storage/SIR_dataset_medium.hdf5', 'r')
print('Using .values():')
print(list(f.values()))
print('Using .keys():')
print(list(f.keys()))
X_dst = f['X_data']
print('X data shape:')
print(X_dst.shape)
y_dst = f['y_data']
print('y data shape:')
print(y_dst.shape)
print('First 10^6 X datapoints:')
print(X_dst[:1000000, ...])
