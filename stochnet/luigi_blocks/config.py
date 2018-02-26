import luigi


class global_params(luigi.Config):
    timestep = luigi.FloatParameter()
    project_folder = luigi.Parameter()
    endtime = luigi.FloatParameter()
    CRN_name = luigi.Parameter()
    nb_past_timesteps = luigi.IntParameter()
    algorithm = luigi.FloatParameter()
