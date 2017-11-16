import numpy as np


class Gene():

    @staticmethod
    def get_species():
        return ['G0', 'G1', 'M', 'P']

    @classmethod
    def get_n_species(cls):
        return len(cls.get_species())

    @classmethod
    def get_initial_settings(cls, n_settings):
        G0 = np.random.randint(low=0, high=2, size=(n_settings, 1))
        G1 = np.ones_like(G0, dtype=float) - G0
        M = np.random.randint(low=0, high=6, size=(n_settings, 1))
        P = np.random.randint(low=400, high=800, size=(n_settings, 1))
        settings = np.concatenate((G0, G1, M, P), axis=1)
        return settings

    @classmethod
    def get_histogram_bounds(cls):
        n_species_for_histogram = len(cls.get_species_for_histogram())
        histogram_bounds = [[0.5, 1800.5]] * n_species_for_histogram
        return histogram_bounds

    @staticmethod
    def get_species_for_histogram():
        return ['P']
