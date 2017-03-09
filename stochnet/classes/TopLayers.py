import tensorflow as tf
from Keras.layers import Dense, merge
from stochnet.classes.Tensor_RandomVariables import Categorical


class CategoricalOutputLayer:

    def __init__(self, number_of_classes):
        self.number_of_classes = number_of_classes

    def get_layer(self):
        return Dense(self.number_of_classes, activation='softmax')

    def get_tensor_random_variable(self, NN_prediction):
        # NN_prediction is expected to be of the following shape:
        # [batch_size, self.number_of_classes]
        return Categorical(NN_prediction)


class MultivariateNormalCholeskyOutputLayer:

    def __init__(self, sample_space_dimension):
        self.sample_space_dimension = sample_space_dimension

    def get_layer(self):
        mu = Dense(self.sample_space_dimension, activation=None)
        chol_diag = Dense(self.sample_space_dimension, activation=tf.exp)
        number_of_sub_diag_entries = self.sample_space_dimension * (self.sample_space_dimension - 1) // 2
        chol_sub_diag = Dense(number_of_sub_diag_entries, activation=None)
        return merge([mu, chol_diag, chol_sub_diag], mode='concat')

    def get_tensor_random_variable(self, NN_prediction):
        # TODO: finire di implementare
