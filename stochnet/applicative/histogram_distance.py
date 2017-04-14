import numpy as np
from numpy.linalg import norm
from numpy.random import normal
from stochnet.utils.histograms import histogram_distance, get_histogram

"""
Procedura di base:
1) carico la rete addestrata
2) recupero dall'architettura della rete m, la lunghezza della memoria usata per addestrarla
3) recupero la lunghezza del timestep in termini assoluti
4) fisso una sequenza (X_1,...,X_m)
5) valuto la rete su quella sequenza di input e ottengo una distribuzione di probabilità
6) generò 1 milione di sample da quella distribuzione di probabilità per X_{m+1}
7) genero 1 milione di sample per X_{m+1} usando l'SSA
8) calcolo l'histogram distance
"""

# def histogram_distance(h_X, h_Y, interval_length):
#     histogram_distance = norm(h_X - h_Y, ord=1) * interval_length
#     return histogram_distance
#
# def get_valid_and_sorted_samples(samples, x_min, x_max):
#     valid_samples = samples[samples < x_max]
#     valid_samples = valid_samples[x_min <= valid_samples]
#     return np.sort(valid_samples)
#
# def get_histogram(samples, x_min, x_max, nb_of_sub_intervals):
#     interval_length = (x_max - x_min) / nb_of_sub_intervals
#     sorted_v_samples = get_valid_and_sorted_samples(samples, x_min, x_max)
#     nb_samples = len(samples)
#     nb_valid_samples = len(sorted_v_samples)
#     interval_index = 1
#     sample_index = 0
#     hist = np.zeros(nb_of_sub_intervals)
#     while interval_index <= nb_of_sub_intervals and sample_index < nb_valid_samples:
#         if sorted_v_samples[sample_index] < x_min + interval_index * interval_length:
#             hist[interval_index - 1] += 1
#             sample_index += 1
#         else:
#             interval_index += 1
#
#     nb_of_out_of_bounds_samples = nb_samples - nb_valid_samples
#     print("There were {0} out of the chosen interval.".format(nb_of_out_of_bounds_samples))
#     return hist / nb_samples


X = normal(loc=0.0, scale=1.0, size=10**6)
Y = normal(loc=0.0, scale=1.0, size=10**6)
x_min = -5
x_max = 5
nb_of_sub_intervals = 10**2
interval_length = abs(x_max - x_min) / nb_of_sub_intervals
h_X = get_histogram(X, x_min, x_max, nb_of_sub_intervals)
h_Y = get_histogram(Y, x_min, x_max, nb_of_sub_intervals)
hist_dist = histogram_distance(h_X, h_Y, interval_length)
print(hist_dist)
