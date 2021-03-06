import os


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

    # TODO: define another way to initialize
    # DatasetFileExplorer using only dataset_folder
    def __init__(self, data_root_folder, timestep, dataset_id):
        self.data_root_folder = data_root_folder
        self.dataset_id = dataset_id
        self.timestep = timestep
        self.dataset_folder = os.path.join(
            self.data_root_folder,
            str(self.timestep) + '/' + str(self.dataset_id)
        )
        create_dir_if_it_does_not_exist(self.dataset_folder)
        self.log_fp = os.path.join(self.dataset_folder, 'log.txt')
        self.dataset_fp = os.path.join(self.dataset_folder,
                                       'dataset.npy')
        self.x_fp = os.path.join(self.dataset_folder, 'x.npy')
        self.y_fp = os.path.join(self.dataset_folder, 'y.npy')
        self.rescaled_x_fp = os.path.join(self.dataset_folder,
                                          'x_rescaled.npy')
        self.rescaled_y_fp = os.path.join(self.dataset_folder,
                                          'y_rescaled.npy')
        self.scaler_fp = os.path.join(self.dataset_folder, 'scaler.h5')
        self.histogram_settings_fp = os.path.join(
            self.dataset_folder,
            'histogram_settings.npy'
        )
        self.histogram_dataset_fp = os.path.join(
            self.dataset_folder,
            'histogram_dataset.npy'
        )

    def get_HistogramFileExplorer(self, model_id, nb_steps):
        return HistogramFileExplorer(self.dataset_folder,
                                     model_id,
                                     nb_steps)


class ModelFileExplorer():

    def __init__(self, models_root_folder, timestep, model_id):
        self.models_root_folder = models_root_folder
        self.model_id = model_id
        self.timestep = timestep
        self.model_folder = os.path.join(
            self.models_root_folder,
            str(self.timestep) + '/' + str(self.model_id)
        )
        create_dir_if_it_does_not_exist(self.model_folder)
        self.log_fp = os.path.join(self.model_folder, 'log.txt')
        self.weights_fp = os.path.join(self.model_folder,
                                       'best_weights.h5')
        self.keras_fp = os.path.join(self.model_folder,
                                     'keras_model.h5')
        self.StochNet_fp = os.path.join(self.model_folder,
                                        'StochNet_object.h5')


class HistogramFileExplorer():

    def __init__(self, dataset_folder, model_id, nb_steps):
        self.dataset_folder = dataset_folder
        self.model_id = model_id
        self.histogram_folder = os.path.join(
            self.dataset_folder,
            (
                'histogram/model_' +
                str(self.model_id) +
                '/{0}'.format(nb_steps)
            )
        )
        create_dir_if_it_does_not_exist(self.histogram_folder)
        self.log_fp = os.path.join(self.histogram_folder, 'log.txt')
