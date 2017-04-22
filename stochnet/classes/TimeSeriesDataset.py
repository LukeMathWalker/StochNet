from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from stochnet.classes.Errors import ShapeError
from keras import backend as K
import numpy as np
import h5py
import tqdm
from bidict import bidict


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
        self.with_timestamps = with_timestamps
        self.set_labels(labels)

    def read_data(self, dataset_address, data_format):
        self.data_format = data_format

        if data_format == 'numpy':
            with open(dataset_address, 'rb') as data_file:
                self.data = np.asarray(np.load(data_file), dtype=K.floatx())
        elif data_format == 'hdf5':
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

    def format_dataset_for_ML(self, keep_timestamps=False, nb_past_timesteps=1,
                              must_be_rescaled=True, positivity=None, percentage_of_test_data=0.25):
        if keep_timestamps is False:
            self.remove_timestamps()

        if must_be_rescaled is True:
            self.rescale(positivity)

        self.explode_into_training_pieces(nb_past_timesteps)
        self.train_test_split(percentage_of_test_data=percentage_of_test_data)

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
            if positivity == 'needed':
                positive_eps = 2**(-25)
                self.scaler = MinMaxScaler(feature_range=(positive_eps, 1))
            else:
                self.scaler = StandardScaler()
            if self.data_format == 'numpy':
                # StandardScaler expects data of the form [n_samples, n_features]
                flat_data = self.data.reshape(-1, self.nb_features)
                self.data = self.scaler.fit_transform(flat_data)
                self.data = self.data.reshape(self.nb_trajectories, self.nb_timesteps, self.nb_features)
            elif self.data_format == 'hdf5':
                slice_size = 10**6
                nb_iteration = self.nb_trajectories // slice_size
                for i in range(nb_iteration):
                    data_slice = self.data[i * slice_size: (i + 1) * slice_size, ...]
                    flat_data_slice = np.asarray(data_slice, dtype=K.floatx()).reshape(-1, self.nb_features)
                    self.scaler.partial_fit(X=flat_data_slice)
                if nb_iteration * slice_size != self.nb_trajectories:
                    data_slice = self.data[nb_iteration * slice_size:, ...]
                    flat_data_slice = np.asarray(data_slice, dtype=K.floatx()).reshape(-1, self.nb_features)
                    self.scaler.partial_fit(X=flat_data_slice)
                for i in range(nb_iteration):
                    data_slice = self.data[i * slice_size: (i + 1) * slice_size, ...]
                    flat_data_slice = np.asarray(data_slice, dtype=K.floatx()).reshape(-1, self.nb_features)
                    self.scaler.transform(X=flat_data_slice)
                    self.data[i * slice_size: (i + 1) * slice_size, ...] = flat_data_slice.reshape(-1, self.nb_timesteps, self.nb_features)
                if nb_iteration * slice_size != self.nb_trajectories:
                    data_slice = self.data[nb_iteration * slice_size:, ...]
                    flat_data_slice = np.asarray(data_slice, dtype=K.floatx()).reshape(-1, self.nb_features)
                    self.scaler.transform(X=flat_data_slice)
                    self.data[nb_iteration * slice_size:, ...] = flat_data_slice.reshape(-1, self.nb_timesteps, self.nb_features)
            self.rescaled = True

    def explode_into_training_pieces(self, nb_past_timesteps, mode='numpy', filepath_for_saving=None):
        # TODO: add the possibility of seeing more than 1 timestep in the future
        self.check_if_nb_past_timesteps_is_valid(nb_past_timesteps)
        if mode == 'numpy':
            self.X_data, self.y_data = self.explode_into_training_pieces_a_batch_of_traj(0, self.nb_trajectories, nb_past_timesteps)
        elif mode == 'hdf5':
            if filepath_for_saving is None:
                raise ValueError('hdf5 mode needs a valid filepath for saving.')

            chunck_size = 10**6
            nb_training_pieces_from_one_trajectory = self.nb_timesteps - nb_past_timesteps
            nb_trajectory_per_chunk = min(chunck_size // nb_training_pieces_from_one_trajectory, 1)
            nb_training_pieces_per_chunk = nb_trajectory_per_chunk * nb_training_pieces_from_one_trajectory
            nb_iteration = self.nb_trajectories // nb_trajectory_per_chunk

            self.f_ML_data = h5py.File(str(filepath_for_saving), 'a', libver='latest')
            self.X_data = self.f_ML_data.create_dataset("X_data", (self.nb_trajectories * nb_training_pieces_from_one_trajectory, nb_past_timesteps, self.nb_features), chunks=True)
            self.y_data = self.f_ML_data.create_dataset("y_data", (self.nb_trajectories * nb_training_pieces_from_one_trajectory, self.nb_features), chunks=True)
            for i in tqdm(range(nb_iteration)):
                X_data_chunk, y_data_chunk = self.explode_into_training_pieces_a_batch_of_traj(i * nb_trajectory_per_chunk, (i + 1) * nb_trajectory_per_chunk, nb_past_timesteps)
                self.X_data[i * nb_training_pieces_per_chunk: (i + 1) * nb_training_pieces_per_chunk, ...] = X_data_chunk
                self.y_data[i * nb_training_pieces_per_chunk: (i + 1) * nb_training_pieces_per_chunk, ...] = y_data_chunk
            if nb_trajectory_per_chunk * nb_iteration != self.nb_trajectories:
                X_data_chunk, y_data_chunk = self.explode_into_training_pieces_a_batch_of_traj(nb_iteration * nb_trajectory_per_chunk, self.nb_trajectories, nb_past_timesteps)
                self.X_data[nb_iteration * nb_trajectory_per_chunk:, ...] = X_data_chunk
                self.y_data[nb_iteration * nb_trajectory_per_chunk:, ...] = y_data_chunk
        else:
            raise ValueError('Unknown mode')

    def explode_into_training_pieces_a_batch_of_traj(self, range_start, range_end, nb_past_timesteps):
        X_data, y_data = [], []
        for trajectory in range(range_start, range_end):
            for oldest_timestep in range(self.nb_timesteps - nb_past_timesteps):
                X_placeholder = self.data[trajectory, oldest_timestep:(oldest_timestep + nb_past_timesteps), :]
                y_placeholder = self.data[trajectory, oldest_timestep + nb_past_timesteps, :]
                X_data.append(X_placeholder)
                y_data.append(y_placeholder)
        X_data = np.array(X_data, dtype=K.floatx())
        y_data = np.array(y_data, dtype=K.floatx())
        return X_data, y_data

    def check_if_nb_past_timesteps_is_valid(self, nb_past_timesteps):
        if nb_past_timesteps + 1 > self.nb_timesteps:
            raise ValueError('You are asking for too many past timesteps!')
        elif nb_past_timesteps < 1:
            raise ValueError('You need to consider at least 1 timestep in the past!')

    def train_test_split(self, percentage_of_test_data=0.25):
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X_data, self.y_data,
                                                                                test_size=percentage_of_test_data)
