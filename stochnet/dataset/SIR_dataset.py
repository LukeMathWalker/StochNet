import numpy as np
import stochpy
import pandas as pd
import os
import shutil


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


def set_initial_parameters(simulation_obj, setting_collection, current_index):
    S = setting_collection[current_index]['S']
    I = setting_collection[current_index]['I']
    smod.ChangeInitialSpeciesCopyNumber("S", S)
    smod.ChangeInitialSpeciesCopyNumber("I", I)
    smod.ChangeInitialSpeciesCopyNumber("N", S + I)


nb_of_different_simulation_settings = 5
simulation_settings = [{'S': 60, 'I': 5},
                       {'S': 100, 'I': 10},
                       {'S': 200, 'I': 50},
                       {'S': 50, 'I': 25},
                       {'S': 100, 'I': 25}]
if nb_of_different_simulation_settings != len(simulation_settings):
    raise ValueError("Mancano o ci sono troppi setting iniziali.")
trajectories_nb = 30
endtime = 4
time_step_for_resampling = 2**(-7)

for i in range(nb_of_different_simulation_settings):
    smod = stochpy.SSA()
    smod.Model("SIR.psc")
    smod.ChangeParameter("Beta", 3)
    smod.ChangeParameter("Gamma", 1)
    set_initial_parameters(smod, simulation_settings, i)
    R = 0
    smod.ChangeInitialSpeciesCopyNumber("R", R)

    smod.DoStochSim(method="direct", trajectories=trajectories_nb, mode="time", end=endtime)
    smod.Export2File(analysis='timeseries', datatype='species', IsAverage=False, directory='SIR', quiet=False)

    datapoint = pd.read_table(filepath_or_buffer='SIR/SIR.psc_species_timeseries1.txt', delim_whitespace=True, header=1).drop(labels="Reaction", axis=1).drop(labels="N", axis=1).drop(labels='Fired', axis=1).as_matrix()
    resampled_datapoint = time_resampling(datapoint, time_step=time_step_for_resampling)
    dataset = resampled_datapoint[np.newaxis, :]
    basename = 'SIR/SIR.psc_species_timeseries'
    for j in range(2, trajectories_nb + 1):
        path = basename + str(j) + '.txt'
        datapoint = pd.read_table(filepath_or_buffer=path, delim_whitespace=True, header=1).drop(labels="Reaction", axis=1).drop(labels='Fired', axis=1).drop(labels="N", axis=1).as_matrix()
        resampled_datapoint = time_resampling(datapoint)
        dataset = np.concatenate((dataset, resampled_datapoint[np.newaxis, :]), axis=0)

    dataset_filepath = 'dataset_' + str(i) + '.npy'
    with open(dataset_filepath, 'wb'):
        np.save(dataset_filepath, dataset)

shutil.rmtree('SIR')

for i in range(nb_of_different_simulation_settings):
    partial_dataset_filepath = 'dataset_' + str(i) + '.npy'
    with open(partial_dataset_filepath, 'rb'):
        partial_dataset = np.load(dataset_filepath)
    if i == 0:
        final_dataset = partial_dataset
    else:
        final_dataset = np.concatenate((final_dataset, partial_dataset), axis=0)
    os.remove(partial_dataset_filepath)

with open('SIR_dataset.npy', 'wb') as dataset_filepath:
    np.save(dataset_filepath, final_dataset)
