import unittest
import numpy as np
import os
from StochNet.components.classes.TimeSeriesDataset import TimeSeriesDataset


class Test_TimeSeriesDataset_with_Valid_Input(unittest.TestCase):

    def setUp(self):
        # creates a valid input for TimeSeriesDataset __init__ method
        data = np.random.rand(2, 3, 5)
        self.dataset_address = 'data.npy'
        with open(self.dataset_address, 'wb') as data_file:
            np.save(data_file, data)

    def test_init_with_valid_address(self):
        timeseries_dataset = TimeSeriesDataset(self.dataset_address)

    # def test_init_with_invalid_address(self):
    #     dataset_address = 'data.npy'
    #     timeseries_dataset = TimeSeriesDataset(dataset_address)

    def tearDown(self):
        os.remove(self.dataset_address)


if __name__ == '__main__':
    unittest.main()
