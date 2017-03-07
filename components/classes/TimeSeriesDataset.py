from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import numpy as np

class TimeSeriesDataset:

    def __init__(self, dataset_address):
        # data: [n_trajectories, n_timesteps, nb_features]
        # data[:,:,0] contains time instants, which are not properly a feature
        with open(dataset_address, 'rb') as data_file:
            self.data = np.load(data_file)
        self.nb_trajectories = self.data.shape[0]
        self.nb_timesteps = self.data.shape[1]
        self.nb_features = self.data.shape[2]
        self.rescaled = False

    def format_dataset_for_ML(self, keep_timestamps=False, nb_past_timesteps=1,
                              must_be_rescaled=True):
        if keep_timestamps is False:
            self.remove_timestamps(self)

        if must_be_rescaled is True:
            self.rescale(self)

        self.explode_into_training_pieces(self, nb_past_timesteps)

    def remove_timestamps(self):
        self.data = self.data[..., 1:]
        self.nb_features = self.nb_features-1

    def rescale(self):
        self.scaler = StandardScaler()
        # StandardScaler expects data of the form [n_samples, n_features]
        flat_data = self.data.reshape(-1, self.nb_features)
        self.data = self.scaler.fit_transform(flat_data)
        self.rescaled = True

    def explode_into_training_pieces(self, nb_past_timesteps):
        X_data, y_data = [], []
        for trajectory in range(self.nb_trajectories):
            for oldest_timestep in range(self.nb_timesteps-nb_past_timesteps-1):
                X_placeholder = self.data[trajectory, oldest_timestep:(oldest_timestep+nb_past_timesteps), :]
                y_placeholder = self.data[trajectory, oldest_timestep+nb_past_timesteps, :]
                X_data.append(X_placeholder)
                y_data.append(y_placeholder)
        self.X_data = np.array(X_data)
        self.y_data = np.array(y_data)

    def train_test_split(self, percentage_of_test_data=0.25):
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X_data, self.y_data,
                                                                                test_size=percentage_of_test_data)
