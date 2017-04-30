import numpy as np
from numpy.random import randint
import matplotlib.pyplot as plt

from keras.utils.generic_utils import get_custom_objects

from stochnet.classes.NeuralNetworks import StochNeuralNetwork
from stochnet.utils.histograms import histogram_distance, get_histogram

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

    datapoint = pd.read_table(filepath_or_buffer='SIR/SIR.psc_species_timeseries1.txt', delim_whitespace=True, header=1).drop(labels="Reaction", axis=1).drop(labels="N", axis=1).drop(labels='Fired', axis=1).as_matrix()
    resampled_datapoint = np.stack((datapoint[0, :], datapoint[-1, :]), axis=0)
    dataset = resampled_datapoint[np.newaxis, :]
    basename = 'SIR/SIR.psc_species_timeseries'
    for j in range(2, nb_of_trajectories + 1):
        path = basename + str(j) + '.txt'
        datapoint = pd.read_table(filepath_or_buffer=path, delim_whitespace=True, header=1).drop(labels="Reaction", axis=1).drop(labels='Fired', axis=1).drop(labels="N", axis=1).as_matrix()
        resampled_datapoint = np.stack((datapoint[0, :], datapoint[-1, :]), axis=0)
        dataset = np.concatenate((dataset, resampled_datapoint[np.newaxis, :]), axis=0)

    if delete is True:
        shutil.rmtree('SIR')

    return dataset


def set_initial_parameters(simulation_obj, setting_collection):
    S = int(setting_collection['S'])
    I = int(setting_collection['I'])
    R = int(setting_collection['R'])
    simulation_obj.ChangeInitialSpeciesCopyNumber('S', S)
    simulation_obj.ChangeInitialSpeciesCopyNumber("I", I)
    simulation_obj.ChangeInitialSpeciesCopyNumber("R", R)
    simulation_obj.ChangeInitialSpeciesCopyNumber("N", S + I + R)
    return


def sample_from_distribution(NN, NN_prediction, nb_samples, sess=None):
    if sess is None:
        sess = tf.Session()
    samples = NN.TopLayer_obj.sample(NN_prediction, nb_samples, sess)
    return samples


np.set_printoptions(suppress=True)
sess = tf.Session()

nb_of_trajectories_for_hist = 10**3
nb_of_initial_configurations = 15
nb_features = 3
time_step_size = 2**(-5)
steps_in_the_future = 2**4
endtime = time_step_size * steps_in_the_future

initial_sequences = generate_simulation_settings_array(nb_of_settings=nb_of_initial_configurations)
initial_sequences = initial_sequences.reshape(nb_of_initial_configurations, 1, nb_features)

stoch_filepath = '/home/lucap/Documenti/Tesi Magistrale/StochNet/stochnet/models/model_09/SIR_-4.83225399017.h5'
NN = StochNeuralNetwork.load(stoch_filepath)

model_filepath = '/home/lucap/Documenti/Tesi Magistrale/StochNet/stochnet/models/model_09/model.h5'

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
    for j in range(steps_in_the_future - 1):
        print(NN_prediction.shape)
        print(NN_samples_rescaled.shape)
        print('\n')
        NN_prediction = NN.predict(NN_samples_rescaled)
        NN_samples_rescaled = sample_from_distribution(NN, NN_prediction, 1, sess).reshape(nb_of_trajectories_for_hist, 1, 3)
    NN_samples = NN.scaler.inverse_transform(NN_samples_rescaled.reshape(-1, nb_features)).reshape(nb_of_trajectories_for_hist, -1, nb_features)
    S_samples_NN = NN_samples[:, 0, 0]
    S_NN_hist = get_histogram(S_samples_NN, -0.5, 200.5, 201)
    print(S_NN_hist)
    plt.figure(i)
    plt.plot(S_NN_hist, label='NN')

    simulation_setting = {'S': initial_sequences[i, 0, 0], 'I': initial_sequences[i, 0, 1], 'R': initial_sequences[i, 0, 2]}

    trajectories = SSA_simulation(simulation_setting, endtime, nb_of_trajectories_for_hist, time_step_size)
    S_samples_SSA = trajectories[:, -1, 1]
    S_SSA_hist = get_histogram(S_samples_SSA, -0.5, 200.5, 201)
    print(S_SSA_hist)
    plt.plot(S_SSA_hist, label='SSA')
    plt.legend()
    plt.savefig('test_' + str(i) + '.png', bbox_inches='tight')

    plt.close()
    S_histogram_distance[i] = histogram_distance(S_NN_hist, S_SSA_hist, 1)
    # print("Histogram distance:")
    # print(S_histogram_distance)
print(S_histogram_distance)
print(np.mean(S_histogram_distance))
