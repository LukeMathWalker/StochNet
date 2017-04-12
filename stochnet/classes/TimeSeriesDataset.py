from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from stochnet.classes.Errors import ShapeError
import numpy as np
from bidict import bidict


class TimeSeriesDataset:
    # TODO: add the possibility of dropping features.
    # Pay attentions to self.scaler and self.labels.
    def __init__(self, dataset_address, with_timestamps=True, labels=None):
        # data: [n_trajectories, n_timesteps, nb_features]
        # if with_timestamps is True the corresponding column is labeled
        # "Timestamps" indipendently of the desired user label
        with open(dataset_address, 'rb') as data_file:
            self.data = np.load(data_file)
        self.memorize_dataset_shape()
        self.rescaled = False
        self.with_timestamps = with_timestamps
        self.set_labels(labels)

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

    def rescale(self, positivity):
        if self.rescaled is False:
            if positivity == 'needed':
                positive_eps = 2**(-25)
                self.scaler = MinMaxScaler(feature_range=(positive_eps, 1))
            else:
                self.scaler = StandardScaler()
            # StandardScaler expects data of the form [n_samples, n_features]
            flat_data = self.data.reshape(-1, self.nb_features)
            self.data = self.scaler.fit_transform(flat_data)
            self.data = self.data.reshape(self.nb_trajectories, self.nb_timesteps, self.nb_features)
            self.rescaled = True

    def explode_into_training_pieces(self, nb_past_timesteps):
        # TODO: add the possibility of seeing more than 1 timestep in the future
        self.check_if_nb_past_timesteps_is_valid(nb_past_timesteps)
        X_data, y_data = [], []
        for trajectory in range(self.nb_trajectories):
            for oldest_timestep in range(self.nb_timesteps - nb_past_timesteps):
                X_placeholder = self.data[trajectory, oldest_timestep:(oldest_timestep + nb_past_timesteps), :]
                y_placeholder = self.data[trajectory, oldest_timestep + nb_past_timesteps, :]
                X_data.append(X_placeholder)
                y_data.append(y_placeholder)
        self.X_data = np.array(X_data)
        self.y_data = np.array(y_data)

    def check_if_nb_past_timesteps_is_valid(self, nb_past_timesteps):
        if nb_past_timesteps + 1 > self.nb_timesteps:
            raise ValueError('You are asking for too many past timesteps!')
        elif nb_past_timesteps < 1:
            raise ValueError('You need to consider at least 1 timestep in the past!')

    def train_test_split(self, percentage_of_test_data=0.25):
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X_data, self.y_data,
                                                                                test_size=percentage_of_test_data)
