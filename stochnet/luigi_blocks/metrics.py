import luigi
import luigi.contrib.external_program
from luigi.util import inherits
from importlib import import_module
from stochnet.utils.file_organization import ProjectFileExplorer
from stochnet.luigi_blocks.config import global_params
from stochnet.utils.envs import get_python_3_env
from stochnet.luigi_blocks.models import TrainNN
from stochnet.luigi_blocks.data import GenerateHistogramData
from datetime import datetime 


@inherits(global_params)
class HistogramDistance(luigi.contrib.external_program.ExternalPythonProgramTask):

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
        print(datetime.now().time())
        program_module = import_module("stochnet.applicative.histogram_w_gillespy")
        program_address = program_module.__file__
        return ['python', program_address, self.timestep,
                self.nb_past_timesteps, self.training_dataset_id,
                self.validation_dataset_id, self.model_id,
                self.project_folder, self.CRN_name]

    def output(self):
        project_explorer = ProjectFileExplorer(self.project_folder)
        train_explorer = project_explorer.get_DatasetFileExplorer(self.timestep, self.training_dataset_id)
        train_histogram_explorer = train_explorer.get_HistogramFileExplorer(self.model_id)
        val_explorer = project_explorer.get_DatasetFileExplorer(self.timestep, self.validation_dataset_id)
        val_histogram_explorer = val_explorer.get_HistogramFileExplorer(self.model_id)
        return [luigi.LocalTarget(train_histogram_explorer.log_fp),
                luigi.LocalTarget(val_histogram_explorer.log_fp)]
