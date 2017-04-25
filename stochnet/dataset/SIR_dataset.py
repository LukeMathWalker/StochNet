import numpy as np
from numpy.random import randint
import stochpy
import pandas as pd
import os
import shutil
from tqdm import tqdm


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


def set_initial_parameters(simulation_obj, setting_collection, current_index):
    S = setting_collection[current_index]['S']
    I = setting_collection[current_index]['I']
    R = setting_collection[current_index]['R']
    smod.ChangeInitialSpeciesCopyNumber("S", S)
    smod.ChangeInitialSpeciesCopyNumber("I", I)
    smod.ChangeInitialSpeciesCopyNumber("R", R)
    smod.ChangeInitialSpeciesCopyNumber("N", S + I + R)
    return


def generate_simulation_settings(min_value=10, max_value=200, nb_of_settings=10):
    simulation_settings = []
    for i in range(nb_of_settings):
        simulation_setting = {}
        simulation_setting['S'] = randint(low=min_value, high=max_value)
        simulation_setting['I'] = randint(low=min_value, high=max_value)
        simulation_setting['R'] = randint(low=min_value, high=max_value)
        simulation_settings.append(simulation_setting)
    return simulation_settings


nb_of_settings = 300
trajectories_nb = 1000
simulation_settings = generate_simulation_settings(min_value=10, max_value=200, nb_of_settings=nb_of_settings)
endtime = 1
time_step_for_resampling = 2**(-5)

smod = stochpy.SSA()
smod.Model("SIR.psc")
smod.ChangeParameter("Beta", 3)
smod.ChangeParameter("Gamma", 1)

for i in tqdm(range(nb_of_settings)):
    set_initial_parameters(smod, simulation_settings, i)

    smod.DoStochSim(method="direct", trajectories=trajectories_nb, mode="time", end=endtime)
    smod.Export2File(analysis='timeseries', datatype='species', IsAverage=False, directory='SIR', quiet=False)

    datapoint = pd.read_table(filepath_or_buffer='SIR/SIR.psc_species_timeseries1.txt', delim_whitespace=True, header=1).drop(labels="Reaction", axis=1).drop(labels="N", axis=1).drop(labels='Fired', axis=1).as_matrix()
    resampled_datapoint = time_resampling(datapoint, time_step=time_step_for_resampling, end_time=endtime)
    dataset = resampled_datapoint[np.newaxis, :]
    basename = 'SIR/SIR.psc_species_timeseries'
    for j in range(2, trajectories_nb + 1):
        path = basename + str(j) + '.txt'
        datapoint = pd.read_table(filepath_or_buffer=path, delim_whitespace=True, header=1).drop(labels="Reaction", axis=1).drop(labels='Fired', axis=1).drop(labels="N", axis=1).as_matrix()
        resampled_datapoint = time_resampling(datapoint, time_step=time_step_for_resampling, end_time=endtime)
        dataset = np.concatenate((dataset, resampled_datapoint[np.newaxis, :]), axis=0)

    dataset_filepath = 'dataset_' + str(i) + '.npy'
    with open(dataset_filepath, 'wb'):
        np.save(dataset_filepath, dataset)

shutil.rmtree('SIR')

for i in range(nb_of_settings):
    partial_dataset_filepath = 'dataset_' + str(i) + '.npy'
    with open(partial_dataset_filepath, 'rb'):
        partial_dataset = np.load(partial_dataset_filepath)
    if i == 0:
        final_dataset = partial_dataset
    else:
        final_dataset = np.concatenate((final_dataset, partial_dataset), axis=0)
    os.remove(partial_dataset_filepath)

with open('SIR_dataset_timestep_2-5_01.npy', 'wb') as dataset_filepath:
    np.save(dataset_filepath, final_dataset)
