import unittest
import numpy as np
import os
from stochnet.classes.TimeSeriesDataset import TimeSeriesDataset


class Test_TimeSeriesDataset_with_Valid_Input(unittest.TestCase):

    def setUp(self):
        # creates a valid input for TimeSeriesDataset __init__ method
        data = np.random.rand(2, 3, 5)
        self.dataset_address = 'data.npy'
        with open(self.dataset_address, 'wb') as data_file:
            np.save(data_file, data)

    def test_init_with_valid_address(self):
        timeseries_dataset = TimeSeriesDataset(self.dataset_address)

    def tearDown(self):
        os.remove(self.dataset_address)

class Test_TimeSeriesDataset_with_Invalid_Input(unittest.TestCase):

    def test_init_with_invalid_address(self):
        invalid_dataset_address = 'data.npy'
        with self.assertRaises(FileNotFoundError):
            TimeSeriesDataset(invalid_dataset_address)


if __name__ == '__main__':
    unittest.main()
