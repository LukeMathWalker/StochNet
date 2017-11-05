import gillespy
import numpy as np


class Gene(gillespy.Model):
    """
    This is a simple example for mass-action degradation of species S.
    """

    def __init__(self, alpha=166, beta=0.1, Psteady=350):

        # Initialize the model.
        gillespy.Model.__init__(self, name="Gene")

        # Parameters
        Kp = gillespy.Parameter(name='Kp', expression=35)
        Kt = gillespy.Parameter(name='Kt', expression=0.001 * beta * Psteady)
        Kd1 = gillespy.Parameter(name='Kd1', expression=0.001)
        Kd2 = gillespy.Parameter(name='Kd2', expression=beta * 0.001)
        Kb = gillespy.Parameter(name='Kb', expression=alpha)
        Ku = gillespy.Parameter(name='Ku', expression=1)
        self.add_parameter([Kp, Kt, Kd1, Kd2, Kb, Ku])

        # Species
        G0 = gillespy.Species(name='G0', initial_value=0)
        G1 = gillespy.Species(name='G1', initial_value=1)
        M = gillespy.Species(name='M', initial_value=1)
        P = gillespy.Species(name='P', initial_value=350)
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
        self.timespan(np.linspace(0, 1000000, 1000001))

    def set_species_initial_value(self, species_initial_value):
        self.listOfSpecies['G0'].initial_value = species_initial_value[0]
        self.listOfSpecies['G1'].initial_value = species_initial_value[1]
        self.listOfSpecies['M'].initial_value = species_initial_value[2]
        self.listOfSpecies['P'].initial_value = species_initial_value[3]

    def get_n_species(self):
        species = self.get_all_species()
        return len(species)

    def get_initial_settings(self, n_settings):
        n_species = self.get_n_species()
        settings = np.random.randint(low=30, high=200, size=(n_settings, n_species))
        return settings
