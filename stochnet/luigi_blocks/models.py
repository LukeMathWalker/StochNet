import os
import luigi
import luigi.contrib.external_program
from luigi.util import inherits
from stochnet.luigi_blocks.config import global_params
from stochnet.utils.envs import get_python_3_env
from stochnet.luigi_blocks.data import FormatDataset
from stochnet.utils.file_organization import ProjectFileExplorer


@inherits(global_params)
class TrainNN(luigi.contrib.external_program.ExternalPythonProgramTask):

    training_dataset_id = luigi.IntParameter()
    validation_dataset_id = luigi.IntParameter()
    model_id = luigi.IntParameter()
    nb_settings = luigi.IntParameter()
    nb_trajectories = luigi.IntParameter()

    virtualenv = get_python_3_env()

    def requires(self):
        return [self.clone(FormatDataset, dataset_id=self.training_dataset_id),
                self.clone(FormatDataset, dataset_id=self.validation_dataset_id)]

    def program_args(self):
        program_address = os.path.join(self.project_folder,
                                       'model_training_w_numpy.py')
        return ['python', program_address, self.timestep,
                self.nb_past_timesteps, self.training_dataset_id,
                self.validation_dataset_id, self.project_folder,
                self.model_id]

    def output(self):
        project_explorer = ProjectFileExplorer(self.project_folder)
        model_explorer = project_explorer.get_ModelFileExplorer(self.timestep, self.model_id)
        return [luigi.LocalTarget(model_explorer.weights_fp),
                luigi.LocalTarget(model_explorer.keras_fp),
                luigi.LocalTarget(model_explorer.StochNet_fp)]
