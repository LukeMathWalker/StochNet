import numpy as np
from time import time
import sys
import os
from importlib import import_module


def save_simulation_data(dataset, dataset_folder, prefix, id_number):
    partial_dataset_filename = str(prefix) + str(id_number) + '.npy'
    partial_dataset_filepath = os.path.join(dataset_folder, partial_dataset_filename)
    with open(partial_dataset_filepath, 'wb') as f:
        np.save(f, dataset)
    return


def single_simulation(CRN, initial_values, nb_trajectories, dataset_folder, prefix, id_number):
    CRN.set_species_initial_value(initial_values)
    trajectories = CRN.run(number_of_trajectories=nb_trajectories, show_labels=False)
    dataset = np.array(trajectories)
    save_simulation_data(dataset, dataset_folder, prefix, id_number)


if __name__ == '__main__':
    start = time()
    nb_trajectories = int(sys.argv[1])
    timestep = float(sys.argv[2])
    endtime = float(sys.argv[3])
    dataset_folder = str(sys.argv[4])
    model_name = str(sys.argv[5])
    prefix = str(sys.argv[6])
    id_number = int(sys.argv[7])

    CRN_module = import_module("stochnet.CRN_models." + model_name)
    CRN_class = getattr(CRN_module, model_name)
    CRN = CRN_class(endtime, timestep)
    settings_fp = os.path.join(dataset_folder, 'settings.npy')
    settings = np.load(settings_fp)
    single_simulation(CRN, settings[id_number], nb_trajectories, dataset_folder, prefix, id_number)
