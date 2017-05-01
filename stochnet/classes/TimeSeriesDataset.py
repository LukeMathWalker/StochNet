from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from stochnet.classes.Errors import ShapeError
from keras import backend as K
import numpy as np
import h5py
from tqdm import tqdm
from bidict import bidict
from math import ceil


class TimeSeriesDataset:
    # TODO: add the possibility of dropping features.
    # Pay attentions to self.scaler and self.labels.
    def __init__(self, dataset_address, data_format='numpy', with_timestamps=True, labels=None):
        # data: [n_trajectories, n_timesteps, nb_features]
        # if with_timestamps is True the corresponding column is labeled
        # "Timestamps" indipendently of the desired user label
        self.read_data(dataset_address, data_format)
        self.memorize_dataset_shape()
        self.rescaled = False
        self.scaler_is_fitted = False
        self.with_timestamps = with_timestamps
        self.set_labels(labels)

    def read_data(self, dataset_address, data_format):
        self.data_format = data_format

        if self.data_format == 'numpy':
            with open(dataset_address, 'rb') as data_file:
                self.data = np.asarray(np.load(data_file), dtype=K.floatx())
        elif self.data_format == 'hdf5':
            self.path_raw_data = str(dataset_address)
            self.f_raw_data = h5py.File(str(dataset_address), 'a')
            self.data = self.f_raw_data['data']
        else:
            raise TypeError('''Unsupported data format. .npy and .hdf5 are\n
                                the available data formats.''')

    def memorize_dataset_shape(self):
        try:
            self.nb_trajectories = self.data.shape[0]
            self.nb_timesteps = self.data.shape[1]
            self.nb_features = self.data.shape[2]
        except:
            raise ShapeError('''The dataset is not properly formatted.\n
                              We expect the following shape:
                              [nb_trajectories, nb_timesteps, nb_features]''')

    def set_labels(self, labels):
        if labels is None:
            self.with_labels = False
        elif len(labels) != self.nb_features:
            raise ShapeError("There needs to be exactly one label for each feature!\n"
                             "We have {0} labels for {1} features.".format(len(labels), self.nb_features))
        else:
            self.labels = bidict(labels)
            self.set_timestamps_label()
            self.with_labels = True

    def set_timestamps_label(self):
        if self.with_timestamps is True:
            self.labels.inv.pop(0)
            self.labels['timestamps'] = 0

    def set_scaler(self, scaler):
        if self.rescaled is True:
            raise NotImplementedError("You need to undo the first rescale first!")
            # TODO: implement an inverse scaling method
        else:
            # TODO: add checks on scaler shape, fit, etc..
            self.scaler = scaler
            self.scaler_is_fitted = True

    def format_dataset_for_ML(self, keep_timestamps=False, nb_past_timesteps=1,
                              must_be_rescaled=True, positivity=None, percentage_of_test_data=0.25,
                              filepath_for_saving_no_split=None, filepath_for_saving_w_split=None):
        if keep_timestamps is False:
            self.remove_timestamps()

        if must_be_rescaled is True:
            self.rescale(positivity)

        self.explode_into_training_pieces(nb_past_timesteps, filepath_for_saving=filepath_for_saving_no_split)
        self.train_test_split(percentage_of_test_data=percentage_of_test_data, filepath_for_saving=filepath_for_saving_w_split)

    def remove_timestamps(self):
        # data[:,:,0] contains the timestamps if with_timestamps is True
        if self.with_timestamps is True:
            self.data = self.data[..., 1:]
            self.nb_features = self.nb_features - 1
            self.with_timestamps = False
            if self.with_labels is True:
                self.labels.pop('timestamps')
        # FIX: other labels indexes need to be diminished by 1

    def rescale(self, positivity=None):
        if self.rescaled is False:
            if not hasattr(self, 'scaler'):
                self.create_scaler(positivity)
            if self.data_format == 'numpy':
                self.rescale_numpy()
            elif self.data_format == 'hdf5':
                self.rescale_hdf5()
            self.rescaled = True

    def create_scaler(self, positivity=None):
        if positivity == 'needed':
            positive_eps = 2**(-25)
            self.scaler = MinMaxScaler(feature_range=(positive_eps, 1))
        else:
            self.scaler = StandardScaler()

    def rescale_numpy(self):
        flat_data = self.data.reshape(-1, self.nb_features)
        if self.scaler_is_fitted is True:
            self.data = self.scaler.transform(flat_data)
        else:
            self.data = self.scaler.fit_transform(flat_data)
        self.data = self.data.reshape(self.nb_trajectories, self.nb_timesteps, self.nb_features)
        return

    def rescale_hdf5(self):
        slice_size = 10**5
        nb_iteration = self.nb_trajectories // slice_size
        if self.scaler_is_fitted is False:
            self.fit_scaler_hdf5(nb_iteration, slice_size)
        self.apply_scaler_hdf5(nb_iteration, slice_size)

    def fit_scaler_hdf5(self, nb_iteration, slice_size):
        for i in tqdm(range(nb_iteration)):
            data_slice = self.data[i * slice_size: (i + 1) * slice_size, ...]
            flat_data_slice = np.asarray(data_slice, dtype=K.floatx()).reshape(-1, self.nb_features)
            self.scaler.partial_fit(X=flat_data_slice)
        if nb_iteration * slice_size != self.nb_trajectories:
            data_slice = self.data[nb_iteration * slice_size:, ...]
            flat_data_slice = np.asarray(data_slice, dtype=K.floatx()).reshape(-1, self.nb_features)
            self.scaler.partial_fit(X=flat_data_slice)

    def apply_scaler_hdf5(self, nb_iteration, slice_size):
        for i in tqdm(range(nb_iteration)):
            data_slice = self.data[i * slice_size: (i + 1) * slice_size, ...]
            flat_data_slice = np.asarray(data_slice, dtype=K.floatx()).reshape(-1, self.nb_features)
            flat_data_slice_transformed = self.scaler.transform(X=flat_data_slice)
            self.data[i * slice_size: (i + 1) * slice_size, ...] = flat_data_slice_transformed.reshape(-1, self.nb_timesteps, self.nb_features)
        if nb_iteration * slice_size != self.nb_trajectories:
            data_slice = self.data[nb_iteration * slice_size:, ...]
            flat_data_slice = np.asarray(data_slice, dtype=K.floatx()).reshape(-1, self.nb_features)
            flat_data_slice_transformed = self.scaler.transform(X=flat_data_slice)
            self.data[nb_iteration * slice_size:, ...] = flat_data_slice_transformed.reshape(-1, self.nb_timesteps, self.nb_features)

    def explode_into_training_pieces(self, nb_past_timesteps, filepath_for_saving=None):
        # TODO: add the possibility of seeing more than 1 timestep in the future
        self.check_if_nb_past_timesteps_is_valid(nb_past_timesteps)
        if self.data_format == 'numpy':
            self.X_data, self.y_data = self.explode_into_training_pieces_a_batch_of_traj(0, self.nb_trajectories, nb_past_timesteps)
        elif self.data_format == 'hdf5':
            if filepath_for_saving is None:
                raise ValueError('hdf5 mode needs a valid filepath for saving.')

            chunk_size = 10**7
            nb_training_pieces_from_one_trajectory = self.nb_timesteps - nb_past_timesteps
            nb_trajectory_per_chunk = max(chunk_size // nb_training_pieces_from_one_trajectory, 1)
            nb_training_pieces_per_chunk = nb_trajectory_per_chunk * nb_training_pieces_from_one_trajectory
            nb_iteration = self.nb_trajectories // nb_trajectory_per_chunk

            self.path_no_split = str(filepath_for_saving)
            self.f_ML_data_no_split = h5py.File(str(filepath_for_saving), 'a', libver='latest')
            self.X_data = self.f_ML_data_no_split.create_dataset("X_data", (self.nb_trajectories * nb_training_pieces_from_one_trajectory, nb_past_timesteps, self.nb_features), chunks=True)
            self.y_data = self.f_ML_data_no_split.create_dataset("y_data", (self.nb_trajectories * nb_training_pieces_from_one_trajectory, self.nb_features), chunks=True)
            for i in tqdm(range(nb_iteration)):
                X_data_chunk, y_data_chunk = self.explode_into_training_pieces_a_batch_of_traj(i * nb_trajectory_per_chunk, (i + 1) * nb_trajectory_per_chunk, nb_past_timesteps)
                self.X_data[i * nb_training_pieces_per_chunk: (i + 1) * nb_training_pieces_per_chunk, ...] = X_data_chunk
                self.y_data[i * nb_training_pieces_per_chunk: (i + 1) * nb_training_pieces_per_chunk, ...] = y_data_chunk
            if nb_trajectory_per_chunk * nb_iteration != self.nb_trajectories:
                X_data_chunk, y_data_chunk = self.explode_into_training_pieces_a_batch_of_traj(nb_iteration * nb_trajectory_per_chunk, self.nb_trajectories, nb_past_timesteps)
                self.X_data[nb_iteration * nb_training_pieces_per_chunk:, ...] = X_data_chunk
                self.y_data[nb_iteration * nb_training_pieces_per_chunk:, ...] = y_data_chunk
        else:
            raise ValueError('Unknown data format.')

    def explode_into_training_pieces_a_batch_of_traj(self, range_start, range_end, nb_past_timesteps):
        X_data = self.data[range_start:range_end, 0:nb_past_timesteps, :]
        y_data = self.data[range_start:range_end, nb_past_timesteps, :]
        for oldest_timestep in range(1, self.nb_timesteps - nb_past_timesteps):
            X_placeholder = self.data[range_start:range_end, oldest_timestep:(oldest_timestep + nb_past_timesteps), :]
            y_placeholder = self.data[range_start:range_end, oldest_timestep + nb_past_timesteps, :]
            X_data = np.concatenate((X_data, X_placeholder), axis=0)
            y_data = np.concatenate((y_data, y_placeholder), axis=0)
        return X_data, y_data

    def check_if_nb_past_timesteps_is_valid(self, nb_past_timesteps):
        if nb_past_timesteps + 1 > self.nb_timesteps:
            raise ValueError('You are asking for too many past timesteps!')
        elif nb_past_timesteps < 1:
            raise ValueError('You need to consider at least 1 timestep in the past!')

    def train_test_split(self, percentage_of_test_data=0.25, filepath_for_saving=None):
        if self.data_format == 'numpy':
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X_data, self.y_data,
                                                                                    test_size=percentage_of_test_data)
        elif self.data_format == 'hdf5':
            if filepath_for_saving is None:
                raise ValueError('hdf5 mode needs a valid filepath for saving.')
            nb_samples = self.X_data.shape[0]
            nb_past_timesteps = self.X_data.shape[1]
            nb_test = ceil(percentage_of_test_data * nb_samples)
            nb_train = nb_samples - nb_test

            self.path_w_split = str(filepath_for_saving)
            self.f_ML_data_w_split = h5py.File(str(filepath_for_saving), 'a', libver='latest')
            self.X_train = self.f_ML_data_w_split.create_dataset("X_train", (nb_train, nb_past_timesteps, self.nb_features), chunks=True)
            self.y_train = self.f_ML_data_w_split.create_dataset("y_train", (nb_train, self.nb_features), chunks=True)
            self.X_test = self.f_ML_data_w_split.create_dataset("X_test", (nb_test, nb_past_timesteps, self.nb_features), chunks=True)
            self.y_test = self.f_ML_data_w_split.create_dataset("y_test", (nb_test, self.nb_features), chunks=True)

            chunk_size = 10**6
            nb_iteration = nb_samples // chunk_size
            nb_test_per_chunk = ceil(percentage_of_test_data * chunk_size)
            nb_train_per_chunk = chunk_size - nb_test_per_chunk
            for i in tqdm(range(nb_iteration)):
                X_data_slice = np.asarray(self.X_data[i * chunk_size: (i + 1) * chunk_size, ...], dtype=K.floatx())
                y_data_slice = np.asarray(self.y_data[i * chunk_size: (i + 1) * chunk_size, ...], dtype=K.floatx())
                self.X_train[i * nb_train_per_chunk: (i + 1) * nb_train_per_chunk, ...], self.X_test[i * nb_test_per_chunk: (i + 1) * nb_test_per_chunk, ...], self.y_train[i * nb_train_per_chunk: (i + 1) * nb_train_per_chunk, ...], self.y_test[i * nb_test_per_chunk: (i + 1) * nb_test_per_chunk, ...] = train_test_split(X_data_slice, y_data_slice, test_size=percentage_of_test_data)
            if nb_iteration * chunk_size != nb_samples:
                X_data_slice = np.asarray(self.X_data[nb_iteration * chunk_size:, ...], dtype=K.floatx())
                y_data_slice = np.asarray(self.y_data[nb_iteration * chunk_size:, ...], dtype=K.floatx())
                self.X_train[nb_iteration * nb_train_per_chunk:, ...], self.X_test[nb_iteration * nb_test_per_chunk:, ...], self.y_train[nb_iteration * nb_train_per_chunk:, ...], self.y_test[nb_iteration * nb_test_per_chunk:, ...] = train_test_split(X_data_slice, y_data_slice, test_size=percentage_of_test_data)
        else:
            raise ValueError('Unknown data format.')
