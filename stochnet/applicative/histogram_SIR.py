import numpy as np
from numpy.random import randint

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


def generate_initial_sequences(endtime, nb_of_initial_sequences, time_step_for_resampling):
    print('INITIAL SEQUENCES - GENERATING')
    simulation_settings = generate_simulation_settings(nb_of_settings=nb_of_initial_sequences)
    initial_sequences = SSA_simulation(simulation_settings[0], endtime, 1, time_step_for_resampling)
    for i in range(1, nb_of_initial_sequences):
        initial_sequence = SSA_simulation(simulation_settings[i], endtime, 1, time_step_for_resampling)
        initial_sequences = np.concatenate((initial_sequences, initial_sequence), axis=0)
    return initial_sequences


def generate_simulation_settings(min_value=10, max_value=200, nb_of_settings=10):
    simulation_settings = []
    for i in range(nb_of_settings):
        simulation_setting = {}
        simulation_setting['S'] = randint(low=min_value, high=max_value)
        simulation_setting['I'] = randint(low=min_value, high=max_value)
        simulation_setting['R'] = randint(low=min_value, high=max_value)
        simulation_settings.append(simulation_setting)
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
    S = setting_collection['S']
    I = setting_collection['I']
    R = setting_collection['R']
    simulation_obj.ChangeInitialSpeciesCopyNumber("S", S)
    simulation_obj.ChangeInitialSpeciesCopyNumber("I", I)
    simulation_obj.ChangeInitialSpeciesCopyNumber("R", I)
    simulation_obj.ChangeInitialSpeciesCopyNumber("N", S + I + R)
    return


def time_resampling(data, time_step=2**(-7), starting_time=0, end_time=4):
    time_index = 0
    # Il nuovo array dei tempi
    time_array = np.linspace(starting_time, end_time, num=(end_time - starting_time) / time_step + 1)
    # new_data conterrà i dati con la nuova scansione temporale
    # la prima colonna contiene gli istanti di tempo, e quindi corrisponde a time_array
    new_data = np.zeros((time_array.shape[0], data.shape[1]))
    new_data[:, 0] = time_array
    for j in range(len(time_array)):
        # se la simulazione non presenta più eventi prima che sia arrivato l'end_time
        # continuiamo a copiare i valori relativi all'ultimo evento fino a riempire l'array
        if time_index == data.shape[0] - 1:
            new_data[j, 1:] = data[time_index, 1:]
        else:
            # se ci troviamo prima dell'evento di indice time_index+1 copiamo i numeri di molecole precedenti all'evento
            if data[time_index + 1][0] > time_array[j]:
                new_data[j, 1:] = data[time_index, 1:]
            # altrimenti aggiorniamo il time_index e copiamo i numeri di molecole corrispondenti all'evento di indice time_index (già aumentato di 1)
            else:
                time_index = time_index + 1
                new_data[j, 1:] = data[time_index, 1:]
    return new_data


def get_endtime_state(data):
    return data[-1, :]


def get_NN(nb_past_timesteps, nb_features):
    input_tensor = Input(shape=(nb_past_timesteps, nb_features))
    flatten1 = Flatten()(input_tensor)
    NN_body = Dense(2048, kernel_constraint=maxnorm(1.67525276))(flatten1)

    number_of_components = 2
    components = []
    for j in range(number_of_components):
        components.append(MultivariateNormalCholeskyOutputLayer(nb_features))

    TopModel_obj = MixtureOutputLayer(components)

    NN = StochNeuralNetwork(input_tensor, NN_body, TopModel_obj)
    return NN


def get_NN_filepath():
    current = os.getcwd()
    working_path = os.path.dirname(current)
    basename = os.path.abspath(working_path)
    weights_filepath = os.path.join(basename, 'models/dill_test_SIR_-5.71435209751.h5')
    return weights_filepath


def sample_from_distribution(NN, NN_prediction, nb_samples):
    sess = tf.Session()
    samples = NN.TopLayer_obj.sample(NN_prediction, nb_samples, sess)
    return samples


np.set_printoptions(suppress=True)

nb_of_trajectories_for_hist = 10**3
nb_of_initial_configurations = 25
nb_past_timesteps = 1
nb_features = 3
time_step_size = 2**(-5)
initial_sequence_endtime = (nb_past_timesteps - 1) * time_step_size
initial_sequences = generate_initial_sequences(initial_sequence_endtime, nb_of_initial_configurations, time_step_size)
initial_sequences = initial_sequences[..., 1:]
print(initial_sequences.shape)


stoch_filepath = '/home/lucap/Documenti/Tesi Magistrale/StochNet/stochnet/models/model_08/dill_SIR_-8.89851900291.h5'
NN = StochNeuralNetwork.load(stoch_filepath)

model_filepath = '/home/lucap/Documenti/Tesi Magistrale/StochNet/stochnet/models/model_08/model.h5'

get_custom_objects().update({"exp": lambda x: tf.exp(x),
                             "loss_function": NN.TopLayer_obj.loss_function})

NN.load_model(model_filepath)
initial_sequences_rescaled = NN.scaler.transform(initial_sequences.reshape(-1, nb_features)).reshape(nb_of_initial_configurations, -1, nb_features)

S_histogram_distance = np.zeros(nb_of_initial_configurations)

for i in range(nb_of_initial_configurations):
    print('\n\n')
    NN_prediction = NN.predict(initial_sequences_rescaled[i][np.newaxis, :, :])
    NN_samples_rescaled = sample_from_distribution(NN, NN_prediction, nb_of_trajectories_for_hist)
    NN_samples = NN.scaler.inverse_transform(NN_samples_rescaled.reshape(-1, nb_features)).reshape(nb_of_trajectories_for_hist, -1, nb_features)
    S_samples_NN = NN_samples[:, 0, 1]
    S_NN_hist = get_histogram(S_samples_NN, 0.5, 200.5, 200)
    print(S_NN_hist)

    SSA_initial_state = get_endtime_state(initial_sequences[i])
    # print("Initial state:")
    # print(SSA_initial_state)
    simulation_setting = {'S': SSA_initial_state[0], 'I': SSA_initial_state[1], 'R': SSA_initial_state[2]}
    endtime = time_step_size

    trajectories = SSA_simulation(simulation_setting, endtime, nb_of_trajectories_for_hist, time_step_size)
    S_samples_SSA = trajectories[:, -1, 1]
    S_SSA_hist = get_histogram(S_samples_SSA, 0.5, 200.5, 200)
    print(S_SSA_hist)
    S_histogram_distance[i] = histogram_distance(S_NN_hist, S_SSA_hist, 1)
    # print("Histogram distance:")
    # print(S_histogram_distance)
print(S_histogram_distance)
print(np.mean(S_histogram_distance))
