import numpy as np
import os
import sys
from tqdm import tqdm
from stochnet.utils.file_organization import ProjectFileExplorer
from importlib import import_module


def build_simulation_dataset(model, settings, nb_trajectories,
                             dataset_folder, prefix='partial_', how='concat'):
    nb_settings = settings.shape[0]
    perform_simulations(model, settings, nb_settings, nb_trajectories,
                        dataset_folder, prefix=prefix)
    if how == 'concat':
        dataset = concatenate_simulations(nb_settings, dataset_folder, prefix=prefix)
    elif how == 'stack':
        dataset = stack_simulations(nb_settings, dataset_folder, prefix=prefix)
    else:
        raise ValueError("'how' accepts only two arguments: 'concat' and 'stack'.")
    return dataset


def perform_simulations(CRN, settings, nb_settings, nb_trajectories,
                        dataset_folder, prefix='partial_'):
    for j in tqdm(range(nb_settings)):
        initial_values = settings[j]
        single_simulation(CRN, initial_values, nb_trajectories, dataset_folder, prefix, j)
    return


def single_simulation(CRN, initial_values, nb_trajectories, dataset_folder, prefix, id_number):
    CRN.set_species_initial_value(initial_values)
    trajectories = CRN.run(number_of_trajectories=nb_trajectories, show_labels=False)
    dataset = np.array(trajectories)
    save_simulation_data(dataset, dataset_folder, prefix, id_number)
    return


def save_simulation_data(dataset, dataset_folder, prefix, id_number):
    partial_dataset_filename = str(prefix) + str(id_number) + '.npy'
    partial_dataset_filepath = os.path.join(dataset_folder, partial_dataset_filename)
    with open(partial_dataset_filepath, 'wb') as f:
        np.save(f, dataset)
    return


def concatenate_simulations(nb_settings, dataset_folder, prefix='partial_'):
    # final_dataset has the following shape:
    # [nb_settings * nb_trajectories, nb_past_timesteps + 1, nb_features]
    for i in tqdm(range(nb_settings)):
        partial_dataset_filename = str(prefix) + str(i) + '.npy'
        partial_dataset_filepath = os.path.join(dataset_folder, partial_dataset_filename)
        with open(partial_dataset_filepath, 'rb') as f:
            partial_dataset = np.load(f)
        if i == 0:
            final_dataset = partial_dataset
        else:
            final_dataset = np.concatenate((final_dataset, partial_dataset), axis=0)
        os.remove(partial_dataset_filepath)
    return final_dataset


def stack_simulations(nb_settings, dataset_folder, prefix='partial_'):
    # final_dataset has the following shape:
    # [nb_settings, nb_trajectories, nb_past_timesteps + 1, nb_features]
    for i in tqdm(range(nb_settings)):
        partial_dataset_filename = str(prefix) + str(i) + '.npy'
        partial_dataset_filepath = os.path.join(dataset_folder, partial_dataset_filename)
        with open(partial_dataset_filepath, 'rb') as f:
            partial_dataset = np.load(f)
        if i == 0:
            final_dataset = partial_dataset[np.newaxis, ...]
        else:
            final_dataset = np.concatenate((final_dataset, partial_dataset[np.newaxis, ...]), axis=0)
        os.remove(partial_dataset_filepath)
    return final_dataset


if __name__ == '__main__':
    dataset_id = int(sys.argv[1])
    nb_settings = int(sys.argv[2])
    nb_trajectories = int(sys.argv[3])
    timestep = float(sys.argv[4])
    endtime = float(sys.argv[5])
    project_folder = str(sys.argv[6])
    model_name = str(sys.argv[7])

    project_explorer = ProjectFileExplorer(project_folder)
    dataset_explorer = project_explorer.get_DatasetFileExplorer(timestep, dataset_id)

    CRN_module = import_module("stochnet.CRN_models." + model_name)
    CRN_class = getattr(CRN_module, model_name)
    CRN = CRN_class(endtime, timestep)

    settings = CRN_class.get_initial_settings(nb_settings)
    dataset = build_simulation_dataset(CRN, settings, nb_trajectories,
                                       dataset_explorer.dataset_folder, 'concat')

    with open(dataset_explorer.dataset_fp, 'wb') as f:
        np.save(f, dataset)
