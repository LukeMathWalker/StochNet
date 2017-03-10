import tensorflow as tf
import numpy as np
from keras.layers import Dense, merge
from stochnet.classes.Tensor_RandomVariables import Categorical, MultivariateNormalCholesky, Mixture


class CategoricalOutputLayer:

    sample_space_dimension = 1

    def __init__(self, number_of_classes):
        self.number_of_classes = number_of_classes
        self.number_of_output_neurons = number_of_classes

    def add_layer_on_top(self, base_model):
        return Dense(self.number_of_classes, activation='softmax')(base_model)

    def get_tensor_random_variable(self, NN_prediction):
        # NN_prediction is expected to be of the following shape:
        # [batch_size, self.number_of_classes]
        return Categorical(NN_prediction)

    def loss_function(self, y_true, y_pred):
        return -self.get_tensor_random_variable(y_pred).log_prob(y_true)


class MultivariateNormalCholeskyOutputLayer:

    def __init__(self, sample_space_dimension):
        self.sample_space_dimension = sample_space_dimension
        self.number_of_sub_diag_entries = self.sample_space_dimension * (self.sample_space_dimension - 1) // 2
        self.number_of_output_neurons = 2 * self.sample_space_dimension + self.number_of_sub_diag_entries

    def add_layer_on_top(self, base_model):
        mu = Dense(self.sample_space_dimension, activation=None)(base_model)
        chol_diag = Dense(self.sample_space_dimension, activation=tf.exp)(base_model)
        chol_sub_diag = Dense(self.number_of_sub_diag_entries, activation=None)(base_model)
        return merge([mu, chol_diag, chol_sub_diag], mode='concat')

    def get_tensor_random_variable(self, NN_prediction):
        # NN_prediction is expected to come from a layer such as the one produced
        # by add_layer_on_top. In particular, it has to be of the following shape:
        # [batch_size, self.number_of_output_neurons]
        mu = tf.slice(NN_prediction, [0, 0], [-1, self.sample_space_dimension])
        cholesky_diag = tf.slice(NN_prediction, [0, self.sample_space_dimension], [-1, 2 * self.sample_space_dimension])
        cholesky_sub_diag = tf.slice(NN_prediction, [0, 2 * self.sample_space_dimension], [-1, -1])
        cholesky = self.batch_to_lower_triangular_matrix(cholesky_diag, cholesky_sub_diag)
        return MultivariateNormalCholesky(mu, cholesky)

    def batch_to_lower_triangular_matrix(self, batch_diag, batch_sub_diag):
        # batch_diag, batch_sub_diag: [batch_size, *]
        # sltm = strictly_lower_triangular_matrix
        # ltm = lower_triangular_matrix
        print("Batch_diag shape")
        print(batch_diag.shape)
        batch_sltm = tf.map_fn(self.to_strictly_lower_triangular_matrix, batch_sub_diag)
        print("Batch_sltm shape")
        print(batch_sltm.shape)
        batch_diagonal_matrix = tf.map_fn(self.to_diagonal_matrix, batch_diag)
        print("Batch_sltm shape")
        print(batch_diagonal_matrix.shape)
        batch_ltm = batch_sltm + batch_diagonal_matrix
        return batch_ltm

    def to_strictly_lower_triangular_matrix(self, sub_diag):
        # sltm = strictly_lower_triangular_matrix
        sltm__indices = list(zip(*np.tril_indices(self.sample_space_dimension, -1)))
        sltm__indices = tf.constant([list(i) for i in sltm__indices], dtype=tf.int64)
        sltm = tf.sparse_to_dense(sparse_indices=sltm__indices,
                                  output_shape=[self.sample_space_dimension, self.sample_space_dimension],
                                  sparse_values=sub_diag,
                                  default_value=0,
                                  validate_indices=True)
        return sltm

    def to_diagonal_matrix(self, diag):
        return tf.diag(diag)

    def loss_function(self, y_true, y_pred):
        return -self.get_tensor_random_variable(y_pred).log_prob(y_true)


class MixtureOutputLayer:

    def __init__(self, components):
        self.number_of_components = len(components)
        self.categorical = CategoricalOutputLayer(self.number_of_components)
        self.components = list(components)
        self.set_number_of_output_neurons()

    def set_number_of_output_neurons(self):
        self.number_of_output_neurons = self.categorical.number_of_output_neurons
        for component in self.components:
            self.number_of_output_neurons += component.number_of_output_neurons

    def add_layer_on_top(self, base_model):
        # list comprehension preserves the order of the original list.
        categorical_layer = self.categorical.add_layer_on_top(base_model)
        components_layers = [component.add_layer_on_top(base_model) for component in self.components]
        mixture_layers = [categorical_layer] + components_layers
        return merge(mixture_layers, mode='concat')

    def get_tensor_random_variable(self, NN_prediction):
        # NN_prediction is expected to come from a layer such as the one produced
        # by add_layer_on_top. In particular, it has to be of the following shape:
        # [batch_size, self.number_of_output_neurons]
        categorical_predictions = tf.slice(NN_prediction, [0, 0], [-1, self.categorical.number_of_output_neurons])
        categorical_random_variable = self.categorical.get_tensor_random_variable(categorical_predictions)
        components_random_variable = []
        start_slicing_index = self.categorical.number_of_output_neurons
        for component in self.components:
            component_predictions = tf.slice(NN_prediction, [0, start_slicing_index], [-1, start_slicing_index + component.number_of_output_neurons])
            component_random_variable = component.get_tensor_random_variable(component_predictions)
            components_random_variable.append(component_random_variable)
            start_slicing_index += component.number_of_output_neurons
        return Mixture(categorical_random_variable, components_random_variable)

    def loss_function(self, y_true, y_pred):
        return -self.get_tensor_random_variable(y_pred).log_prob(y_true)
