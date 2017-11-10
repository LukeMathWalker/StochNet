import numpy as np


class Schlogl():

    @staticmethod
    def get_species():
        return ['A', 'B', 'X']

    @classmethod
    def get_n_species(cls):
        return len(cls.get_species())

    @classmethod
    def get_initial_settings(cls, n_settings):
        n_species = cls.get_n_species()
        settings = np.random.randint(low=30, high=200, size=(n_settings, n_species))
        return settings

    @classmethod
    def get_histogram_bounds(cls):
        n_species_for_histogram = len(cls.get_species_for_histogram())
        histogram_bounds = [[0.5, 200.5]] * n_species_for_histogram
        return histogram_bounds

    @staticmethod
    def get_species_for_histogram():
        return ['X']
