import unittest
import numpy as np
import os
from stochnet.classes.TimeSeriesDataset import TimeSeriesDataset
from stochnet.classes.Errors import ShapeError


class Test_TimeSeriesDataset_with_Valid_Input(unittest.TestCase):

    def setUp(self):
        # creates a valid input for TimeSeriesDataset __init__ method
        data = np.random.rand(2, 3, 5)
        self.dataset_address = 'data.npy'
        with open(self.dataset_address, 'wb') as data_file:
            np.save(data_file, data)

    def test_init_with_valid_address(self):
        timeseries_dataset = TimeSeriesDataset(self.dataset_address)

    def test_remove_timestamps_once(self):
        timeseries_dataset = TimeSeriesDataset(self.dataset_address, with_timestamps=True)
        nb_features = timeseries_dataset.nb_features
        timeseries_dataset.remove_timestamps()
        self.assertEqual(nb_features-1, timeseries_dataset.nb_features)

    def test_remove_timestamps_iterated(self):
        timeseries_dataset = TimeSeriesDataset(self.dataset_address, with_timestamps=False)
        nb_features = timeseries_dataset.nb_features
        for j in range(5):
            timeseries_dataset.remove_timestamps()
        self.assertEqual(nb_features, timeseries_dataset.nb_features)

    def test_if_scaler_is_fitted(self):
        timeseries_dataset = TimeSeriesDataset(self.dataset_address)
        timeseries_dataset.rescale()
        scaler = timeseries_dataset.scaler
        self.assertTrue(hasattr(scaler, 'mean_'))
        self.assertTrue(hasattr(scaler, 'scale_'))

    def tearDown(self):
        os.remove(self.dataset_address)

class Test_TimeSeriesDataset_with_Invalid_Input(unittest.TestCase):

    dataset_address = 'data.npy'

    def test_init_with_invalid_address(self):
        invalid_dataset_address = self.dataset_address
        with self.assertRaises(FileNotFoundError):
            TimeSeriesDataset(invalid_dataset_address)

    def test_init_with_invalid_shape(self):
        data = np.random.rand(2, 5)
        with open(self.dataset_address, 'wb') as data_file:
            np.save(data_file, data)
        with self.assertRaises(ShapeError):
            TimeSeriesDataset(self.dataset_address)

    def test_explode_into_training_pieces_with_invalid_nb_of_past_timesteps(self):
        data = np.random.rand(2, 3, 5)
        with open(self.dataset_address, 'wb') as data_file:
            np.save(data_file, data)
        timeseries_dataset = TimeSeriesDataset(self.dataset_address)
        with self.assertRaises(ValueError):
            timeseries_dataset.explode_into_training_pieces(4)

    def tearDown(self):
        try:
            os.remove(self.dataset_address)
        except:
            pass

if __name__ == '__main__':
    unittest.main()
