import gillespy
import numpy as np
import math


class SIR(gillespy.Model):
    """
    This is a simple example for mass-action degradation of species S.
    """

    def __init__(self, endtime, timestep):

        # Initialize the model.
        gillespy.Model.__init__(self, name="SIR")

        # Parameters
        beta = gillespy.Parameter(name='beta', expression='3.')
        gamma = gillespy.Parameter(name='gamma', expression='1.')
        self.add_parameter([beta, gamma])

        # Species
        S = gillespy.Species(name='S', initial_value=100)
        I = gillespy.Species(name='I', initial_value=100)
        R = gillespy.Species(name='R', initial_value=100)
        self.add_species([S, I, R])

        # Reactions
        infection = gillespy.Reaction(name='infection',
                                      reactants={S: 1, I: 1},
                                      products={I: 2},
                                      propensity_function='beta*S*I/(S+I+R)')
        recover = gillespy.Reaction(name='recover',
                                    reactants={I: 1},
                                    products={R: 1},
                                    rate=gamma)
        self.add_reaction([infection, recover])
        nb_of_steps = int(math.ceil((endtime / timestep))) + 1
        self.timespan(np.linspace(0, endtime, nb_of_steps))

    def set_species_initial_value(self, species_initial_value):
        self.listOfSpecies['S'].initial_value = species_initial_value[0]
        self.listOfSpecies['I'].initial_value = species_initial_value[1]
        self.listOfSpecies['R'].initial_value = species_initial_value[2]
        return

    @staticmethod
    def get_species():
        return ['S', 'I', 'R']

    @classmethod
    def get_n_species(cls):
        return len(cls.get_species())

    @classmethod
    def get_initial_settings(cls, n_settings):
        n_species = cls.get_n_species()
        settings = np.random.randint(low=30, high=200, size=(n_settings, n_species))
        return settings

    @classmethod
    def get_histogram_bounds(cls):
        n_species_for_histogram = len(cls.get_species_for_histogram())
        histogram_bounds = [[0.5, 200.5]] * n_species_for_histogram
        return histogram_bounds

    @staticmethod
    def get_species_for_histogram(cls):
        return ['S', 'I']
