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
