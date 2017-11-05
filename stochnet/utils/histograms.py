import numpy as np
from numpy.linalg import norm


def histogram_distance(h_X, h_Y, interval_length):
    histogram_distance = norm(h_X - h_Y, ord=1) * interval_length
    return histogram_distance


# def get_valid_and_sorted_samples(samples, x_min, x_max):
#     valid_samples = samples[samples < x_max]
#     valid_samples = valid_samples[x_min <= valid_samples]
#     return np.sort(valid_samples)


def get_histogram(samples, histogram_bounds, n_bins):
    interval_lengths = [(b[1] - b[0]) / n_bins for b in histogram_bounds]
    hist, edges = np.histogramdd(samples, bins=n_bins, range=histogram_bounds)
    nb_samples = samples.shape[0]
    return hist / nb_samples
