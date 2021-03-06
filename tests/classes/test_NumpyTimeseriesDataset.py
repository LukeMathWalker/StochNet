import unittest
import numpy as np
import os
from stochnet.classes.TimeSeriesDataset import NumpyTimeSeriesDataset
from stochnet.classes.Errors import ShapeError
from bidict import ValueDuplicationError


class Test_NumpyTimeSeriesDataset_with_Valid_Input(unittest.TestCase):

    def setUp(self):
        # creates a valid input for NumpyTimeSeriesDataset __init__ method
        self.nb_trajectories = 2
        self.nb_timesteps = 3
        self.nb_features = 5
        data = np.random.rand(self.nb_trajectories, self.nb_timesteps, self.nb_features)
        self.dataset_address = 'data.npy'
        with open(self.dataset_address, 'wb') as data_file:
            np.save(data_file, data)

    def test_init_with_valid_address(self):
        _ = NumpyTimeSeriesDataset(self.dataset_address)

    def test_init_with_labels(self):
        labels = {key: value for (key, value) in zip('abcde', range(5))}
        _ = NumpyTimeSeriesDataset(self.dataset_address, labels=labels)

    def test_remove_timestamps_once(self):
        timeseries_dataset = NumpyTimeSeriesDataset(self.dataset_address, with_timestamps=True)
        nb_features = timeseries_dataset.nb_features
        timeseries_dataset.remove_timestamps()
        self.assertEqual(nb_features - 1, timeseries_dataset.nb_features)
        self.assertEqual(len(timeseries_dataset.data.shape), 3)

    def test_remove_timestamps_iterated(self):
        timeseries_dataset = NumpyTimeSeriesDataset(self.dataset_address, with_timestamps=False)
        nb_features = timeseries_dataset.nb_features
        for j in range(5):
            timeseries_dataset.remove_timestamps()
        self.assertEqual(nb_features, timeseries_dataset.nb_features)
        self.assertEqual(len(timeseries_dataset.data.shape), 3)

    def test_if_scaler_is_fitted(self):
        timeseries_dataset = NumpyTimeSeriesDataset(self.dataset_address)
        timeseries_dataset.rescale()
        scaler = timeseries_dataset.scaler
        self.assertTrue(hasattr(scaler, 'mean_'))
        self.assertTrue(hasattr(scaler, 'scale_'))
        self.assertEqual(len(timeseries_dataset.data.shape), 3)

    def test_if_the_output_of_explode_into_training_pieces_has_correct_shape(self):
        timeseries_dataset = NumpyTimeSeriesDataset(self.dataset_address)
        for nb_past_timesteps in range(1, self.nb_timesteps):
            timeseries_dataset.explode_into_training_pieces(nb_past_timesteps)
            correct_nb_of_samples = self.nb_trajectories * (self.nb_timesteps - nb_past_timesteps)
            correct_shape_for_X = (correct_nb_of_samples, nb_past_timesteps, self.nb_features)
            correct_shape_for_y = (correct_nb_of_samples, self.nb_features)
            self.assertEqual(timeseries_dataset.X_data.shape, correct_shape_for_X)
            self.assertEqual(timeseries_dataset.y_data.shape, correct_shape_for_y)

    def tearDown(self):
        os.remove(self.dataset_address)


class Test_NumpyTimeSeriesDataset_with_Invalid_Input(unittest.TestCase):

    dataset_address = 'data.npy'

    def test_init_with_invalid_address(self):
        invalid_dataset_address = self.dataset_address
        with self.assertRaises(FileNotFoundError):
            NumpyTimeSeriesDataset(invalid_dataset_address)

    def test_init_with_invalid_shape(self):
        data = np.random.rand(2, 5)
        with open(self.dataset_address, 'wb') as data_file:
            np.save(data_file, data)
        with self.assertRaises(ShapeError):
            NumpyTimeSeriesDataset(self.dataset_address)

    def test_init_using_labels_with_invalid_shape(self):
        data = np.random.rand(2, 3, 5)
        with open(self.dataset_address, 'wb') as data_file:
            np.save(data_file, data)
        labels_with_invalid_shape = {key: value for (key, value) in zip('abcd', range(4))}
        with self.assertRaises(ShapeError):
            _ = NumpyTimeSeriesDataset(self.dataset_address, labels=labels_with_invalid_shape)

    def test_init_using_not_unique_labels(self):
        data = np.random.rand(2, 3, 5)
        with open(self.dataset_address, 'wb') as data_file:
            np.save(data_file, data)
        not_unique_labels = {key: value for (key, value) in zip('abcde', [0, 1, 2, 2, 3])}
        with self.assertRaises(ValueDuplicationError):
            _ = NumpyTimeSeriesDataset(self.dataset_address, labels=not_unique_labels)

    def test_explode_into_training_pieces_with_too_many_past_timesteps(self):
        data = np.random.rand(2, 3, 5)
        with open(self.dataset_address, 'wb') as data_file:
            np.save(data_file, data)
        timeseries_dataset = NumpyTimeSeriesDataset(self.dataset_address)
        with self.assertRaises(ValueError):
            timeseries_dataset.explode_into_training_pieces(4)

    def test_explode_into_training_pieces_with_zero_past_timesteps(self):
        data = np.random.rand(2, 3, 5)
        with open(self.dataset_address, 'wb') as data_file:
            np.save(data_file, data)
        timeseries_dataset = NumpyTimeSeriesDataset(self.dataset_address)
        with self.assertRaises(ValueError):
            timeseries_dataset.explode_into_training_pieces(0)

    def tearDown(self):
        try:
            os.remove(self.dataset_address)
        except:
            pass


if __name__ == '__main__':
    unittest.main()
