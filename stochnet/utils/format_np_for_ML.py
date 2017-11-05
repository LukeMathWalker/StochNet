import sys
import dill
import numpy as np
from stochnet.classes.TimeSeriesDataset import NumpyTimeSeriesDataset
from stochnet.utils.file_organization import ProjectFileExplorer


if __name__ == '__main__':

    nb_past_timesteps = int(sys.argv[1])
    dataset_id = int(sys.argv[2])
    timestep = float(sys.argv[3])
    project_folder = str(sys.argv[4])

    project_explorer = ProjectFileExplorer(project_folder)
    dataset_explorer = project_explorer.get_DatasetFileExplorer(timestep, dataset_id)

    timeseries = NumpyTimeSeriesDataset(dataset_explorer.dataset_fp,
                                        with_timestamps=True,
                                        labels=None)

    timeseries.format_dataset_for_ML(keep_timestamps=False,
                                     nb_past_timesteps=nb_past_timesteps,
                                     must_be_rescaled=False,
                                     positivity=None,
                                     train_test_split=False)

    with open(dataset_explorer.x_fp, 'wb') as f:
        np.save(f, timeseries.X_data)

    with open(dataset_explorer.y_fp, 'wb') as f:
        np.save(f, timeseries.y_data)

    del timeseries

    rescaled_timeseries = NumpyTimeSeriesDataset(dataset_explorer.dataset_fp,
                                                 with_timestamps=True,
                                                 labels=None)
    rescaled_timeseries.format_dataset_for_ML(keep_timestamps=False,
                                              nb_past_timesteps=nb_past_timesteps,
                                              must_be_rescaled=True,
                                              positivity=None,
                                              train_test_split=False)

    with open(dataset_explorer.rescaled_x_fp, 'wb') as f:
        np.save(f, rescaled_timeseries.X_data)

    with open(dataset_explorer.rescaled_y_fp, 'wb') as f:
        np.save(f, rescaled_timeseries.y_data)

    with open(dataset_explorer.scaler_fp, 'wb') as f:
        dill.dump(rescaled_timeseries.scaler, f)
