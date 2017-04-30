import gillespy
import numpy as np
import os
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
        self.timespan(np.linspace(0, 5, 12))

    def set_species_initial_value(self, species_initial_value):
        self.listOfSpecies['S'].initial_value = species_initial_value[0]
        self.listOfSpecies['I'].initial_value = species_initial_value[1]
        self.listOfSpecies['R'].initial_value = species_initial_value[2]
        return


if __name__ == '__main__':
    test = SIR()

    nb_settings = 1
    settings = np.random.randint(low=30, high=200, size=(nb_settings, 3))
    num_trajectories = 1
    for j in tqdm(range(nb_settings)):
        species_initial_value = settings[j]
        test.set_species_initial_value(species_initial_value)
        trajectories = test.run(number_of_trajectories=num_trajectories, show_labels=False)
        dataset = np.array(trajectories)
        print(dataset)
    #     dataset_filepath = 'dataset_' + str(j) + '.npy'
    #     with open(dataset_filepath, 'wb'):
    #         np.save(dataset_filepath, dataset)
    #
    # for i in range(nb_settings):
    #     partial_dataset_filepath = 'dataset_' + str(i) + '.npy'
    #     with open(partial_dataset_filepath, 'rb'):
    #         partial_dataset = np.load(partial_dataset_filepath)
    #     if i == 0:
    #         final_dataset = partial_dataset
    #     else:
    #         final_dataset = np.concatenate((final_dataset, partial_dataset), axis=0)
    #     os.remove(partial_dataset_filepath)
    #
    # with open('SIR_dataset_timestep_2-1_05.npy', 'wb') as dataset_filepath:
    #     np.save(dataset_filepath, final_dataset)
