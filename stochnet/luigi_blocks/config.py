import luigi


class global_params(luigi.Config):
    project_folder = luigi.Parameter()
    timestep = luigi.FloatParameter()
    endtime = luigi.FloatParameter()
    CRN_name = luigi.Parameter()
    nb_past_timesteps = luigi.IntParameter()
