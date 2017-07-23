import os
import numpy as np
import dill
from stochnet.utils.iterator import NumpyArrayIterator
from stochnet.utils import change_scaling


def create_dir_if_it_does_not_exist(file_path):
    if not os.path.exists(file_path):
        os.makedirs(file_path)


class ProjectFileExplorer():

    def __init__(self, project_folder):
        self.project_folder = project_folder
        self.data_root_folder = os.path.join(project_folder, 'dataset/data')
        create_dir_if_it_does_not_exist(self.data_root_folder)
        self.models_root_folder = os.path.join(project_folder, 'models')
        create_dir_if_it_does_not_exist(self.models_root_folder)

    def get_DatasetFileExplorer(self, timestep, dataset_id):
        return DatasetFileExplorer(self.data_root_folder, timestep, dataset_id)

    def get_ModelFileExplorer(self, timestep, model_id):
        return ModelFileExplorer(self.models_root_folder, timestep, model_id)


class DatasetFileExplorer():

    def __init__(self, data_root_folder, timestep, dataset_id):
        self.data_root_folder = data_root_folder
        self.dataset_id = dataset_id
        self.timestep = timestep
        self.dataset_folder = os.path.join(self.data_root_folder,
                                           str(self.timestep) + '/' + str(self.dataset_id))
        create_dir_if_it_does_not_exist(self.dataset_folder)
        self.dataset_fp = os.path.join(self.dataset_folder, 'dataset.npy')
        self.x_fp = os.path.join(self.dataset_folder, 'x.npy')
        self.y_fp = os.path.join(self.dataset_folder, 'y.npy')
        self.rescaled_x_fp = os.path.join(self.dataset_folder, 'x_rescaled.npy')
        self.rescaled_y_fp = os.path.join(self.dataset_folder, 'y_rescaled.npy')
        self.scaler_fp = os.path.join(self.dataset_folder, 'scaler.h5')
        self.histogram_settings_fp = os.path.join(self.dataset_folder, 'histogram_settings.npy')
        self.histogram_dataset_fp = os.path.join(self.dataset_folder, 'histogram_dataset.npy')

    def get_HistogramFileExplorer(self, model_id):
        return HistogramFileExplorer(self.dataset_folder, model_id)


class ModelFileExplorer():

    def __init__(self, models_root_folder, timestep, model_id):
        self.models_root_folder = models_root_folder
        self.model_id = model_id
        self.timestep = timestep
        self.model_folder = os.path.join(self.models_root_folder,
                                         str(self.timestep) + '/' + str(self.model_id))
        create_dir_if_it_does_not_exist(self.model_folder)
        self.weights_fp = os.path.join(self.model_folder, 'best_weights.h5')
        self.keras_fp = os.path.join(self.model_folder, 'keras_model.h5')
        self.StochNet_fp = os.path.join(self.model_folder, 'StochNet_object.h5')


class HistogramFileExplorer():

    def __init__(self, dataset_folder, model_id):
        self.dataset_folder = dataset_folder
        self.model_id = model_id
        self.histogram_folder = os.path.join(self.dataset_folder,
                                             'histogram/model_' + str(self.model_id))
        create_dir_if_it_does_not_exist(self.histogram_folder)
        self.log_fp = os.path.join(self.histogram_folder, 'log.txt')


def get_train_and_validation_generator_w_scaler(train_explorer, val_explorer, batch_size=64):
    rescaled_x_train, rescaled_y_train, scaler_train = get_rescaled_dataset(train_explorer)
    rescaled_x_val, rescaled_y_val, scaler_val = get_rescaled_dataset(val_explorer)

    rescaled_x_val = change_scaling(rescaled_x_val, scaler_val, scaler_train)
    rescaled_y_val = change_scaling(rescaled_y_val, scaler_val, scaler_train)

    training_generator = NumpyArrayIterator(rescaled_x_train, rescaled_y_train,
                                            batch_size=batch_size,
                                            shuffle=True)
    validation_generator = NumpyArrayIterator(rescaled_x_val, rescaled_y_val,
                                              batch_size=batch_size,
                                              shuffle=True)
    return training_generator, validation_generator, scaler_train


def get_rescaled_dataset(dataset_explorer):
    with open(dataset_explorer.rescaled_x_fp, 'rb') as f:
        rescaled_x = np.load(f)
    with open(dataset_explorer.rescaled_y_fp, 'rb') as f:
        rescaled_y = np.load(f)
    with open(dataset_explorer.scaler_fp, 'rb') as f:
        scaler = dill.load(f)
    return rescaled_x, rescaled_y, scaler
