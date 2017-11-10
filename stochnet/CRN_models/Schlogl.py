import gillespy
import numpy as np
import math

class Schlogl(gillespy.Model):

    def __init__(self, endtime, timestep):

        # Initialize the model.
        gillespy.Model.__init__(self, name="Schlogl")

        # Parameters
        a = gillespy.Parameter(name='a', expression='3.')
        b = gillespy.Parameter(name='b', expression='1.')
        c = gillespy.Parameter(name='c', expression='1.')
        d = gillespy.Parameter(name='d', expression='1.')
        self.add_parameter([a, b, c, d])

        # Species
        A = gillespy.Species(name='A', initial_value=10**5)
        B = gillespy.Species(name='B', initial_value=2 * 10**5)
        X = gillespy.Species(name='X', initial_value=200)
        self.add_species([A, B, X])

        # Reactions
        r1 = gillespy.Reaction(name='r1',
                               reactants={A: 1, X: 2},
                               products={X: 3},
                               propensity_function='a*A*X*(X-1)/2')
        r2 = gillespy.Reaction(name='r2',
                               reactants={X: 3},
                               products={A: 1, X: 2},
                               propensity_function='b*X*(X-1)*(X-2)/6')
        r3 = gillespy.Reaction(name='r3',
                               reactants={B: 1},
                               products={X: 1},
                               propensity_function='c*B')
        r4 = gillespy.Reaction(name='r4',
                               reactants={X: 1},
                               products={B: 1},
                               propensity_function='d*X')
        self.add_reaction([r1, r2, r3, r4])
        nb_of_steps = int(math.ceil((endtime / timestep))) + 1
        self.timespan(np.linspace(0, endtime, nb_of_steps))

    def set_species_initial_value(self, species_initial_value):
        self.listOfSpecies['A'].initial_value = species_initial_value[0]
        self.listOfSpecies['B'].initial_value = species_initial_value[1]
        self.listOfSpecies['X'].initial_value = species_initial_value[2]
        return

    @staticmethod
    def get_species():
        return ['A', 'B', 'X']

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
    def get_species_for_histogram():
        return ['X']
