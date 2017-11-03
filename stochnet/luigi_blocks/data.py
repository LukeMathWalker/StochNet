import luigi
import luigi.contrib.external_program
from luigi.util import inherits, requires
from importlib import import_module
from stochnet.utils.file_organization import ProjectFileExplorer
from stochnet.luigi_blocks.config import global_params
from stochnet.utils.envs import get_python_2_env, get_python_3_env


@inherits(global_params)
class GenerateDataset(luigi.contrib.external_program.ExternalPythonProgramTask):

    dataset_id = luigi.IntParameter()
    nb_settings = luigi.IntParameter()
    nb_trajectories = luigi.IntParameter()

    virtualenv = get_python_2_env()

    def program_args(self):
        program_module = import_module("stochnet.dataset.simulation_w_gillespy")
        program_address = program_module.__file__
        return ['python', program_address, self.dataset_id, self.nb_settings,
                self.nb_trajectories, self.timestep, self.endtime,
                self.project_folder, self.CRN_name]

    def output(self):
        project_explorer = ProjectFileExplorer(self.project_folder)
        dataset_explorer = project_explorer.get_DatasetFileExplorer(self.timestep, self.dataset_id)
        return luigi.LocalTarget(dataset_explorer.dataset_fp)


@requires(GenerateDataset)
class FormatDataset(luigi.contrib.external_program.ExternalPythonProgramTask):

    virtualenv = get_python_3_env()

    def program_args(self):
        program_module = import_module("stochnet.utils.format_np_for_ML")
        program_address = program_module.__file__
        return ['python', program_address, self.nb_past_timesteps,
                self.dataset_id, self.timestep, self.project_folder]

    def output(self):
        project_explorer = ProjectFileExplorer(self.project_folder)
        dataset_explorer = project_explorer.get_DatasetFileExplorer(self.timestep, self.dataset_id)
        return [luigi.LocalTarget(dataset_explorer.x_fp),
                luigi.LocalTarget(dataset_explorer.y_fp),
                luigi.LocalTarget(dataset_explorer.rescaled_x_fp),
                luigi.LocalTarget(dataset_explorer.rescaled_y_fp),
                luigi.LocalTarget(dataset_explorer.scaler_fp)]


@requires(FormatDataset)
class GenerateHistogramData(luigi.contrib.external_program.ExternalPythonProgramTask):

    nb_histogram_settings = luigi.IntParameter()
    nb_histogram_trajectories = luigi.IntParameter()

    virtualenv = get_python_2_env()

    def program_args(self):
        program_module = import_module("stochnet.dataset.generator_for_histogram_w_gillespy")
        program_address = program_module.__file__
        return ['python', program_address, self.timestep,
                self.nb_past_timesteps, self.dataset_id,
                self.nb_histogram_settings, self.nb_histogram_trajectories,
                self.project_folder, self.CRN_name]

    def output(self):
        project_explorer = ProjectFileExplorer(self.project_folder)
        dataset_explorer = project_explorer.get_DatasetFileExplorer(self.timestep, self.dataset_id)
        return [luigi.LocalTarget(dataset_explorer.histogram_settings_fp),
                luigi.LocalTarget(dataset_explorer.histogram_dataset_fp)]
