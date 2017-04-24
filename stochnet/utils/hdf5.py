import h5py
from tqdm import tqdm

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


def concatenate_hdf5_datasets(filepath_1, filepath_2, X_label_1='X_data', y_label_1='y_data', X_label_2='X_data', y_label_2='y_data'):
    f_1 = h5py.File(str(filepath_1), 'a')
    f_2 = h5py.File(str(filepath_2), 'r')
    X_data_1 = f_1[X_label_1]
    X_data_2 = f_2[X_label_2]
    y_data_1 = f_1[y_label_1]
    y_data_2 = f_2[y_label_2]
    old_nb_samples = X_data_1.shape[0]
    new_nb_samples = X_data_1.shape[0] + X_data_2.shape[0]
    X_data_1.resize(new_nb_samples, axis=0)
    y_data_1.resize(new_nb_samples, axis=0)

    chunk_size = 10**5

    nb_iteration = (new_nb_samples - old_nb_samples) // chunk_size
    for i in tqdm(range(nb_iteration)):
        print('\n')
        print(X_data_1[old_nb_samples + i * chunk_size: old_nb_samples + (i + 1) * chunk_size, ...].shape)
        print(X_data_2[i * chunk_size: (i + 1) * chunk_size, ...].shape)
        print('\n')
        print(y_data_1[old_nb_samples + i * chunk_size: old_nb_samples + (i + 1) * chunk_size, ...].shape)
        print(y_data_2[i * chunk_size: (i + 1) * chunk_size, ...].shape)
        X_data_1[old_nb_samples + i * chunk_size: old_nb_samples + (i + 1) * chunk_size, ...] = X_data_2[i * chunk_size: (i + 1) * chunk_size, ...]
        y_data_1[old_nb_samples + i * chunk_size: old_nb_samples + (i + 1) * chunk_size, ...] = y_data_2[i * chunk_size: (i + 1) * chunk_size, ...]
    if nb_iteration * chunk_size != new_nb_samples - old_nb_samples:
        X_data_1[old_nb_samples + nb_iteration * chunk_size:, ...] = X_data_2[nb_iteration * chunk_size:, ...]
        y_data_1[old_nb_samples + nb_iteration * chunk_size:, ...] = y_data_2[nb_iteration * chunk_size:, ...]
    return
