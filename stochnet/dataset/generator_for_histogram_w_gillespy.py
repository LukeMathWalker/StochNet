import sys
import numpy as np
from importlib import import_module
from stochnet.utils.file_organization import ProjectFileExplorer
from stochnet.dataset.simulation_w_gillespy import build_simulation_dataset


def get_histogram_settings(nb_histogram_settings, x_fp):
    """
    Randomly selects a subset of the x dataset to be used as initial setup
    in the construction of the histogram dataset.
    """
    with open(x_fp, 'rb') as f:
        x_data = np.load(f)

    nb_samples = x_data.shape[0]
    settings_index = list(np.random.randint(low=0, high=nb_samples - 1,
                                            size=nb_histogram_settings))
    settings = x_data[settings_index, 0, :]
    return settings


if __name__ == '__main__':

    timestep = float(sys.argv[1])
    nb_past_timesteps = int(sys.argv[2])
    dataset_id = int(sys.argv[3])
    nb_histogram_settings = int(sys.argv[4])
    nb_trajectories = int(sys.argv[5])
    project_folder = str(sys.argv[6])
    model_name = str(sys.argv[7])

    project_explorer = ProjectFileExplorer(project_folder)
    dataset_explorer = project_explorer.get_DatasetFileExplorer(timestep, dataset_id)

    CRN_module = import_module("stochnet.CRN_models." + model_name)
    CRN_class = getattr(CRN_module, model_name)
    CRN = CRN_class(endtime=nb_past_timesteps * timestep, timestep=timestep)

    settings = get_histogram_settings(nb_histogram_settings, dataset_explorer.x_fp)
    with open(dataset_explorer.histogram_settings_fp, 'wb') as f:
        np.save(f, settings)

    histogram_dataset = build_simulation_dataset(CRN, settings, nb_trajectories,
                                                 dataset_explorer.dataset_folder,
                                                 prefix='histogram_partial_')

    with open(dataset_explorer.histogram_dataset_fp, 'wb') as f:
        np.save(f, histogram_dataset)
