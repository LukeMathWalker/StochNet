import gillespy
import numpy as np
import math


class Gene(gillespy.Model):

    def __init__(self, endtime, timestep):

        # Initialize the model.
        gillespy.Model.__init__(self, name="Gene")

        self.alpha = 166
        self.beta = 30
        self.Psteady = 10000

        # Parameters
        Kp = gillespy.Parameter(name='Kp', expression=350)
        Kt = gillespy.Parameter(name='Kt', expression=0.001 * self.beta * self.Psteady)
        Kd1 = gillespy.Parameter(name='Kd1', expression=0.001)
        Kd2 = gillespy.Parameter(name='Kd2', expression=self.beta * 0.05)
        Kb = gillespy.Parameter(name='Kb', expression=self.alpha)
        Ku = gillespy.Parameter(name='Ku', expression=1)
        self.add_parameter([Kp, Kt, Kd1, Kd2, Kb, Ku])

        # Species
        G0 = gillespy.Species(name='G0', initial_value=0)
        G1 = gillespy.Species(name='G1', initial_value=1)
        M = gillespy.Species(name='M', initial_value=1)
        P = gillespy.Species(name='P', initial_value=500)
        self.add_species([G0, G1, M, P])

        # Reactions
        prodM = gillespy.Reaction(name='prodM',
                                  reactants={G1: 1},
                                  products={G1: 1, M: 1},
                                  rate=Kp)
        prodP = gillespy.Reaction(name='prodP',
                                  reactants={M: 1},
                                  products={M: 1, P: 1},
                                  rate=Kt)
        degM = gillespy.Reaction(name='degM',
                                 reactants={M: 1},
                                 products={},
                                 rate=Kd1)
        degP = gillespy.Reaction(name='degP',
                                 reactants={P: 1},
                                 products={},
                                 rate=Kd2)
        reg1G = gillespy.Reaction(name='reg1G',
                                  reactants={G1: 1, P: 1},
                                  products={G0: 1},
                                  rate=Kb)
        reg2G = gillespy.Reaction(name='reg2G',
                                  reactants={G0: 1},
                                  products={G1: 1, P: 1},
                                  rate=Ku)
        self.add_reaction([prodM, prodP, degM, degP, reg1G, reg2G])

        nb_of_steps = int(math.ceil((endtime / timestep))) + 1
        self.timespan(np.linspace(0, endtime, nb_of_steps))

    def set_species_initial_value(self, species_initial_value):
        self.listOfSpecies['G0'].initial_value = species_initial_value[0]
        self.listOfSpecies['G1'].initial_value = species_initial_value[1]
        self.listOfSpecies['M'].initial_value = species_initial_value[2]
        self.listOfSpecies['P'].initial_value = species_initial_value[3]

    @staticmethod
    def get_species():
        return ['G0', 'G1', 'M', 'P']

    @classmethod
    def get_n_species(cls):
        return len(cls.get_species())

    @classmethod
    def get_initial_settings(cls, n_settings):
        G0 = np.random.randint(low=0, high=2, size=(n_settings, 1))
        G1 = np.ones_like(G0, dtype=float) - G0
        M = np.random.randint(low=0, high=6, size=(n_settings, 1))
        P = np.random.randint(low=400, high=800, size=(n_settings, 1))
        settings = np.concatenate((G0, G1, M, P), axis=1)
        return settings

    @classmethod
    def get_histogram_bounds(cls):
        n_species_for_histogram = len(cls.get_species_for_histogram())
        histogram_bounds = [[0.5, 1800.5]] * n_species_for_histogram
        return histogram_bounds

    @staticmethod
    def get_species_for_histogram():
        return ['P']
