import h5py
from stochnet.classes.Errors import ShapeError

def convert_ML_dataset_from_numpy_to_hdf5(np_X_data, np_y_data, filepath_for_saving):
    f = h5py.File(str(filepath_for_saving), 'a', libver='latest')
    X_dset = f.create_dataset("X_data", data=np_X_data, chunks=True)
    y_dset = f.create_dataset("y_data", data=np_y_data, chunks=True)
    f.close()
    return


def convert_from_numpy_to_hdf5(np_data, filepath_for_saving):
    f = h5py.File(str(filepath_for_saving), 'a', libver='latest')
    dset = f.create_dataset("data", data=np_data, chunks=True)
    f.close()
    return


def merge_hdf5_dataset(filepath_1, filepath_2, filepath_new, X_label_1='X_data', y_label_1='y_data', X_label_2='X_data', y_label_2='y_data', X_label_new='X_data', y_label_new='y_data'):
    f_1 = h5py.File(str(filepath_1), 'r')
    f_2 = h5py.File(str(filepath_2), 'r')
    X_data_1 = f_1[X_label_1]
    X_data_2 = f_2[X_label_2]
    y_data_1 = f_1[y_label_1]
    y_data_2 = f_2[y_label_2]
    if X_data_1.shape[1:] != X_data_2.shape[1:] or y_data_1[1:] != y_data_2[1:]:
        raise ShapeError('Dataset dimensions (from axis 1 onward) do not match!')
    nb_samples_new = X_data_1.shape[0] + X_data_2.shape[0]
    X_shape_new = tuple([nb_samples_new, list(X_data_1.shape[1:])])
    y_shape_new = tuple([nb_samples_new, list(y_data_1.shape[1:])])
    f_new = h5py.File(str(filepath_new), 'a', libver='latest')
    X_data_new = f_new.create_dataset(X_label_new, X_shape_new, chunks='True')
    X_data_new[:X_data_1.shape[0], ...] = X_data_1
    X_data_new[X_data_1.shape[0]:, ...] = X_data_2
    y_data_new = f_new.create_dataset(y_label_new, y_shape_new, chunks='True')
    y_data_new[:y_data_1.shape[0], ...] = y_data_1
    y_data_new[y_data_1.shape[0]:, ...] = y_data_2
    return
