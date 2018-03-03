import luigi
from luigi.contrib.external_program import ExternalPythonProgramTask
from luigi.util import inherits, requires
from importlib import import_module
from stochnet.utils.file_organization import ProjectFileExplorer
from stochnet.luigi_blocks.config import global_params
from stochnet.utils.envs import get_python_2_env, get_python_3_env
import os


@inherits(global_params)
class GenerateDataset(ExternalPythonProgramTask):

    dataset_id = luigi.IntParameter()
    nb_settings = luigi.IntParameter()
    nb_trajectories = luigi.IntParameter()

    virtualenv = get_python_2_env()

    def program_args(self):
        program_module = import_module("stochnet.dataset."
                                       "simulation_w_gillespy")
        program_address = program_module.__file__
        return ['python', program_address, self.dataset_id, self.nb_settings,
                self.nb_trajectories, self.timestep, self.endtime,
                self.project_folder, self.CRN_name, self.algorithm]

    def output(self):
        project_explorer = ProjectFileExplorer(self.project_folder)
        dataset_explorer = project_explorer.get_DatasetFileExplorer(
            self.timestep, self.dataset_id
        )
        return [luigi.LocalTarget(dataset_explorer.dataset_fp),
                luigi.LocalTarget(dataset_explorer.log_fp)]


@requires(GenerateDataset)
class FormatDataset(ExternalPythonProgramTask):

    virtualenv = get_python_3_env()

    def program_args(self):
        program_module = import_module("stochnet.utils.format_np_for_ML")
        program_address = program_module.__file__
        return ['python', program_address, self.nb_past_timesteps,
                self.dataset_id, self.timestep, self.project_folder]

    def output(self):
        project_explorer = ProjectFileExplorer(self.project_folder)
        dataset_explorer = project_explorer.get_DatasetFileExplorer(
            self.timestep, self.dataset_id
        )
        return [luigi.LocalTarget(dataset_explorer.x_fp),
                luigi.LocalTarget(dataset_explorer.y_fp),
                luigi.LocalTarget(dataset_explorer.rescaled_x_fp),
                luigi.LocalTarget(dataset_explorer.rescaled_y_fp),
                luigi.LocalTarget(dataset_explorer.scaler_fp),
                luigi.LocalTarget(dataset_explorer.log_fp)]


@requires(FormatDataset)
class GenerateHistogramData(ExternalPythonProgramTask):

    nb_histogram_settings = luigi.IntParameter()
    nb_histogram_trajectories = luigi.IntParameter()
    endtime = luigi.FloatParameter()

    virtualenv = get_python_2_env()

    def program_args(self):
        program_module = import_module("stochnet.dataset."
                                       "generator_for_histogram_w_gillespy")
        program_address = program_module.__file__
        return ['python', program_address, self.timestep,
                self.nb_past_timesteps, self.dataset_id,
                self.nb_histogram_settings, self.nb_histogram_trajectories,
                self.project_folder, self.CRN_name, self.algorithm,
                self.endtime, self.random_seed]

    def output(self):
        project_explorer = ProjectFileExplorer(self.project_folder)
        dataset_explorer = project_explorer.get_DatasetFileExplorer(
            self.timestep, self.dataset_id
        )
        return [luigi.LocalTarget(dataset_explorer.histogram_settings_fp),
                luigi.LocalTarget(dataset_explorer.histogram_dataset_fp),
                luigi.LocalTarget(dataset_explorer.log_fp)]


@inherits(global_params)
class TrainNN(ExternalPythonProgramTask):

    training_dataset_id = luigi.IntParameter()
    validation_dataset_id = luigi.IntParameter()
    model_id = luigi.IntParameter()
    nb_settings = luigi.IntParameter()
    nb_trajectories = luigi.IntParameter()

    virtualenv = get_python_3_env()

    def requires(self):
        return [
            self.clone(FormatDataset, dataset_id=self.training_dataset_id),
            self.clone(FormatDataset, dataset_id=self.validation_dataset_id)
        ]

    def program_args(self):
        program_address = os.path.join(self.project_folder,
                                       'model_training_w_numpy.py')
        return ['python', program_address, self.timestep,
                self.nb_past_timesteps, self.training_dataset_id,
                self.validation_dataset_id, self.project_folder,
                self.model_id]

    def output(self):
        project_explorer = ProjectFileExplorer(self.project_folder)
        model_explorer = project_explorer.get_ModelFileExplorer(self.timestep,
                                                                self.model_id)
        return [luigi.LocalTarget(model_explorer.weights_fp),
                luigi.LocalTarget(model_explorer.keras_fp),
                luigi.LocalTarget(model_explorer.StochNet_fp)]


@inherits(global_params)
class HistogramDistance(ExternalPythonProgramTask):

    training_dataset_id = luigi.IntParameter()
    validation_dataset_id = luigi.IntParameter()
    model_id = luigi.IntParameter()
    nb_settings = luigi.IntParameter()
    nb_histogram_settings = luigi.IntParameter()
    nb_histogram_trajectories = luigi.IntParameter()
    nb_trajectories = luigi.IntParameter()

    virtualenv = get_python_3_env()

    def requires(self):
        return [self.clone(TrainNN),
                self.clone(GenerateHistogramData,
                           dataset_id=self.training_dataset_id),
                self.clone(GenerateHistogramData,
                           dataset_id=self.validation_dataset_id)]

    def program_args(self):
        program_module = import_module("stochnet.applicative."
                                       "histogram_w_gillespy")
        program_address = program_module.__file__
        return ['python', program_address, self.timestep,
                self.nb_past_timesteps, self.training_dataset_id,
                self.validation_dataset_id, self.model_id,
                self.project_folder, self.CRN_name]

    def output(self):
        # TODO: parametrize histogram folders
        project_explorer = ProjectFileExplorer(self.project_folder)
        train_explorer = project_explorer.get_DatasetFileExplorer(
            self.timestep, self.training_dataset_id
        )
        train_histogram_explorer = train_explorer.get_HistogramFileExplorer(
            self.model_id, 5
        )
        val_explorer = project_explorer.get_DatasetFileExplorer(
            self.timestep,
            self.validation_dataset_id
        )
        val_histogram_explorer = val_explorer.get_HistogramFileExplorer(
            self.model_id, 5
        )
        return [luigi.LocalTarget(train_histogram_explorer.log_fp),
                luigi.LocalTarget(val_histogram_explorer.log_fp)]
