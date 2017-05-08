import numpy as np
from numpy.random import randint
import matplotlib.pyplot as plt

from keras.layers import Input, LSTM, Dense, Dropout, Flatten
from keras.constraints import maxnorm
from keras.utils.generic_utils import get_custom_objects
import keras.activations

from stochnet.classes.TimeSeriesDataset import TimeSeriesDataset
from stochnet.classes.NeuralNetworks import StochNeuralNetwork
from stochnet.classes.TopLayers import MultivariateNormalCholeskyOutputLayer, MixtureOutputLayer
from stochnet.utils.histograms import histogram_distance, get_histogram

import os
import stochpy
import pandas as pd
import shutil
import tensorflow as tf

def generate_simulation_settings_array(min_value=10, max_value=200, nb_of_settings=10):
    simulation_settings = randint(low=min_value, high=max_value, size=(nb_of_settings, 3))
    return simulation_settings


def SSA_simulation(simulation_settings, endtime, nb_of_trajectories, time_step_for_resampling, delete=True):
    print('SSA SIMULATION')
    smod = stochpy.SSA()
    smod.Model("SIR.psc")
    smod.ChangeParameter("Beta", 3)
    smod.ChangeParameter("Gamma", 1)
    set_initial_parameters(smod, simulation_settings)
    smod.DoStochSim(method="direct", trajectories=nb_of_trajectories, mode="time", end=endtime)
    smod.Export2File(analysis='timeseries', datatype='species', IsAverage=False, directory='SIR', quiet=False)

    trajectory = pd.read_table(filepath_or_buffer='SIR/SIR.psc_species_timeseries1.txt', delim_whitespace=True, header=1).drop(labels="Reaction", axis=1).drop(labels="N", axis=1).drop(labels='Fired', axis=1).as_matrix()
    resampled_trajectory = time_resampling(trajectory, time_step=time_step_for_resampling, end_time=endtime)
    trajectories = resampled_trajectory[np.newaxis, :]
    basename = 'SIR/SIR.psc_species_timeseries'
    for j in range(2, nb_of_trajectories + 1):
        path = basename + str(j) + '.txt'
        trajectory = pd.read_table(filepath_or_buffer=path, delim_whitespace=True, header=1).drop(labels="Reaction", axis=1).drop(labels='Fired', axis=1).drop(labels="N", axis=1).as_matrix()
        resampled_trajectory = time_resampling(trajectory, time_step=time_step_for_resampling, end_time=endtime)
        trajectories = np.concatenate((trajectories, resampled_trajectory[np.newaxis, :]), axis=0)

    if delete is True:
        shutil.rmtree('SIR')

    return trajectories


def set_initial_parameters(simulation_obj, setting_collection):
    S = int(setting_collection['S'])
    I = int(setting_collection['I'])
    R = int(setting_collection['R'])
    simulation_obj.ChangeInitialSpeciesCopyNumber('S', S)
    simulation_obj.ChangeInitialSpeciesCopyNumber("I", I)
    simulation_obj.ChangeInitialSpeciesCopyNumber("R", R)
    simulation_obj.ChangeInitialSpeciesCopyNumber("N", S + I + R)
    return


def time_resampling(data, time_step=2**(-5), starting_time=0, end_time=4):
    time_index = 0
    # Il nuovo array dei tempi
    time_array = np.linspace(starting_time, end_time, num=(end_time - starting_time) / time_step + 1)
    # new_data conterrà i dati con la nuova scansione temporale
    # la prima colonna contiene gli istanti di tempo, e quindi corrisponde a time_array
    new_data = np.zeros((time_array.shape[0], data.shape[1]))
    new_data[:, 0] = time_array
    for j in range(len(time_array)):
        while time_index < data.shape[0] - 1 and data[time_index + 1][0] < time_array[j]:
            time_index = time_index + 1
        if time_index == data.shape[0] - 1:
            new_data[j, 1:] = data[time_index, 1:]
        else:
            new_data[j, 1:] = data[time_index, 1:]
    return new_data


def get_endtime_state(data):
    return data[-1, :]


def sample_from_distribution(NN, NN_prediction, nb_samples, sess=None):
    if sess is None:
        sess = tf.Session()
    samples = NN.TopLayer_obj.sample(NN_prediction, nb_samples, sess)
    return samples


np.set_printoptions(suppress=True)
sess = tf.Session()

nb_of_trajectories_for_hist = 3 * 10**3
nb_of_initial_configurations = 20
# nb_past_timesteps = 1
nb_features = 3
time_step_size = 5. / 11.
# initial_sequence_endtime = (nb_past_timesteps - 1) * time_step_size
# initial_sequences = generate_initial_sequences(initial_sequence_endtime, nb_of_initial_configurations, time_step_size)
initial_sequences = generate_simulation_settings_array(nb_of_settings=nb_of_initial_configurations)
initial_sequences = initial_sequences.reshape(nb_of_initial_configurations, 1, nb_features)

stoch_filepath = '/home/lucap/Documenti/Tesi Magistrale/StochNet/stochnet/models/SIR_timestep_2-1/model_02/SIR_-21.6554496214.h5'
NN = StochNeuralNetwork.load(stoch_filepath)

model_filepath = '/home/lucap/Documenti/Tesi Magistrale/StochNet/stochnet/models/SIR_timestep_2-1/model_02/model.h5'

get_custom_objects().update({"exp": lambda x: tf.exp(x),
                             "loss_function": NN.TopLayer_obj.loss_function})

NN.load_model(model_filepath)
initial_sequences_rescaled = NN.scaler.transform(initial_sequences.reshape(-1, nb_features)).reshape(nb_of_initial_configurations, -1, nb_features)
S_histogram_distance = np.zeros(nb_of_initial_configurations)

for i in range(nb_of_initial_configurations):
    print('\n\n')
    print(initial_sequences[i])
    NN_prediction = NN.predict(initial_sequences_rescaled[i][np.newaxis, :, :])
    NN_samples_rescaled = sample_from_distribution(NN, NN_prediction, nb_of_trajectories_for_hist, sess)
    NN_samples = NN.scaler.inverse_transform(NN_samples_rescaled.reshape(-1, nb_features)).reshape(nb_of_trajectories_for_hist, -1, nb_features)
    S_samples_NN = NN_samples[:, 0, 0]
    S_NN_hist = get_histogram(S_samples_NN, -0.5, 200.5, 201)
    plt.figure(i)
    plt.plot(S_NN_hist, label='NN')

    simulation_setting = {'S': initial_sequences[i, 0, 0], 'I': initial_sequences[i, 0, 1], 'R': initial_sequences[i, 0, 2]}
    endtime = time_step_size

    trajectories = SSA_simulation(simulation_setting, endtime, nb_of_trajectories_for_hist, time_step_size)
    S_samples_SSA = trajectories[:, -1, 1]
    S_SSA_hist = get_histogram(S_samples_SSA, -0.5, 200.5, 201)
    plt.plot(S_SSA_hist, label='SSA')
    plt.legend()
    plt.savefig('test_' + str(i) + '.png', bbox_inches='tight')

    plt.close()
    S_histogram_distance[i] = histogram_distance(S_NN_hist, S_SSA_hist, 1)
    # print("Histogram distance:")
    # print(S_histogram_distance)
print(S_histogram_distance)
print(np.mean(S_histogram_distance))
