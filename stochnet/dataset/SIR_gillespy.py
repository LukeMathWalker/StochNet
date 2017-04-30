import gillespy
import numpy as np
from tqdm import tqdm


class SIR(gillespy.Model):
    """
    This is a simple example for mass-action degradation of species S.
    """

    def __init__(self):

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
        self.timespan(np.linspace(0, 20, 2))

    def set_species_initial_value(self, species_initial_value):
        self.listOfSpecies['S'].initial_value = species_initial_value[0]
        self.listOfSpecies['I'].initial_value = species_initial_value[1]
        self.listOfSpecies['R'].initial_value = species_initial_value[2]
        return


if __name__ == '__main__':
    species_initial_value = [100, 50, 80]
    test = SIR()

    nb_settings = 3
    settings = [[10, 20, 30], [20, 30, 40], [30, 40, 50]]
    num_trajectories = 1000
    for j in tqdm(range(nb_settings)):
        species_initial_value = settings[j]
        test.set_species_initial_value(species_initial_value)
        simple_trajectories = test.run(number_of_trajectories=num_trajectories)
        # print(np.array(simple_trajectories).shape)
        # print(simple_trajectories[0])
