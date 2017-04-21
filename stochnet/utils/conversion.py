import h5py


def convert_from_numpy_to_hdf5(np_X_data, np_y_data, filepath_for_saving):
    f = h5py.File(str(filepath_for_saving), 'a', libver='latest')
    X_dset = f.create_dataset("X_data", data=np_X_data, chunks=True)
    y_dset = f.create_dataset("y_data", data=np_y_data, chunks=True)
    f.close()
    return
