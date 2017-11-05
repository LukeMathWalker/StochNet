import gillespy
import numpy as np
import math


class SIRS(gillespy.Model):
    """
    This is a simple example for mass-action degradation of species S with immunity loss.
    """

    def __init__(self, endtime, timestep):

        # Initialize the model.
        gillespy.Model.__init__(self, name="SIR")

        # Parameters
        beta = gillespy.Parameter(name='beta', expression='3.')
        gamma = gillespy.Parameter(name='gamma', expression='1.')
        alpha = gillespy.Parameter(name='alpha', expression='0.2')
        self.add_parameter([beta, gamma, alpha])

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
        immunity_loss = gillespy.Reaction(name='recover',
                                          reactants={R: 1},
                                          products={S: 1},
                                          rate=alpha)
        self.add_reaction([infection, recover, immunity_loss])
        nb_of_steps = int(math.ceil((endtime / timestep))) + 1
        self.timespan(np.linspace(0, endtime, nb_of_steps))

    def set_species_initial_value(self, species_initial_value):
        self.listOfSpecies['S'].initial_value = species_initial_value[0]
        self.listOfSpecies['I'].initial_value = species_initial_value[1]
        self.listOfSpecies['R'].initial_value = species_initial_value[2]
        return

    def get_n_species(self):
        species = self.get_all_species()
        return len(species)

    def get_initial_settings(self, n_settings):
        n_species = self.get_n_species()
        settings = np.random.randint(low=30, high=200, size=(n_settings, n_species))
        return settings
