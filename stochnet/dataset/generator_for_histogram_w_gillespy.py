import sys
import numpy as np
import os
from stochnet.utils.file_organization import ProjectFileExplorer
from stochnet.dataset.simulation_w_gillespy import build_simulation_dataset
from time import time


def get_histogram_settings(nb_histogram_settings, x_fp):
    """
    Randomly selects a subset of the x dataset
    to be used as initial setup
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
    start = time()

    timestep = float(sys.argv[1])
    histogram_timestep = float(sys.argv[2])
    nb_past_timesteps = int(sys.argv[3])
    source_dataset_id = int(sys.argv[4])
    target_dataset_id = int(sys.argv[5])
    nb_histogram_settings = int(sys.argv[6])
    nb_trajectories = int(sys.argv[7])
    project_folder = str(sys.argv[8])
    model_name = str(sys.argv[9])
    algorithm = str(sys.argv[10])
    endtime = float(sys.argv[11])
    random_seed = int(sys.argv[12])

    np.random.seed(random_seed)

    project_explorer = ProjectFileExplorer(project_folder)
    dataset_explorer = project_explorer.get_DatasetFileExplorer(
        timestep,
        source_dataset_id
    )

    settings = get_histogram_settings(nb_histogram_settings,
                                      dataset_explorer.x_fp)
    with open(dataset_explorer.histogram_settings_fp, 'wb') as f:
        np.save(f, settings)

    # TODO: remove this workaround

    target_dataset_explorer = project_explorer.get_DatasetFileExplorer(
        histogram_timestep,
        target_dataset_id
    )

    settings_fp = os.path.join(target_dataset_explorer.dataset_folder,
                               'settings.npy')
    np.save(settings_fp, settings)

    histogram_dataset = build_simulation_dataset(
        model_name, nb_histogram_settings, nb_trajectories,
        histogram_timestep, endtime,
        target_dataset_explorer.dataset_folder,
        algorithm=algorithm,
        prefix='histogram_partial_',
        how='stack'
    )

    with open(target_dataset_explorer.histogram_dataset_fp, 'wb') as f:
        np.save(f, histogram_dataset)
    end = time()
    execution_time = end - start
    with open(target_dataset_explorer.log_fp, 'a') as f:
        f.write("Simulating {0} {1} histogram trajectories "
                "for {2} different settings until {3} using {4} "
                "took {5} seconds.\n".format(
                    nb_trajectories,
                    model_name,
                    nb_histogram_settings,
                    endtime,
                    algorithm,
                    execution_time
                ))
