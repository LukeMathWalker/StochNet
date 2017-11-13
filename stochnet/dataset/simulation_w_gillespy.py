import numpy as np
import os
import sys
from tqdm import tqdm
from stochnet.utils.file_organization import ProjectFileExplorer
from importlib import import_module
from time import time
import subprocess


def build_simulation_dataset(model_name, nb_settings, nb_trajectories, timestep, endtime,
                             dataset_folder, prefix='partial_', how='concat'):
    perform_simulations(model_name, nb_settings, nb_trajectories, timestep, endtime,
                        dataset_folder, prefix=prefix)
    if how == 'concat':
        dataset = concatenate_simulations(nb_settings, dataset_folder, prefix=prefix)
    elif how == 'stack':
        dataset = stack_simulations(nb_settings, dataset_folder, prefix=prefix)
    else:
        raise ValueError("'how' accepts only two arguments: 'concat' and 'stack'.")
    return dataset


def perform_simulations(model_name, nb_settings, nb_trajectories, timestep, endtime,
                        dataset_folder, prefix='partial_'):
    # parallel for cycle
    program_module = import_module("stochnet.dataset.single_simulation_w_gillespy")
    program_address = program_module.__file__
    cmd = "seq 0 {7} | rush \"python {0} {1} {2} {3} \'{4}\' \'{5}\' \'{6}\' {{}}\"".format(program_address,
                                                                                            nb_trajectories,
                                                                                            timestep,
                                                                                            endtime,
                                                                                            dataset_folder,
                                                                                            model_name,
                                                                                            prefix,
                                                                                            nb_settings)
    subprocess.call(cmd, shell=True)
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
    start = time()
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

    settings = CRN_class.get_initial_settings(nb_settings)
    settings_fp = os.path.join(dataset_explorer.dataset_folder, 'settings.npy')
    np.save(settings_fp, settings)

    nb_settings = settings.shape[0]

    dataset = build_simulation_dataset(model_name, nb_settings, nb_trajectories, timestep, endtime,
                                       dataset_explorer.dataset_folder, how='concat')

    with open(dataset_explorer.dataset_fp, 'wb') as f:
        np.save(f, dataset)

    end = time()
    execution_time = end - start
    with open(dataset_explorer.log_fp, 'a') as f:
        f.write("Simulating {0} {1} trajectories for {2} different settings (with endtime {3}) took {4} seconds.\n".format(nb_trajectories,
                                                                                                                           model_name,
                                                                                                                           nb_settings,
                                                                                                                           endtime,
                                                                                                                           execution_time))
