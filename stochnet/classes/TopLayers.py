import tensorflow as tf
import numpy as np
from Keras.layers import Dense, merge
from stochnet.classes.Tensor_RandomVariables import Categorical, MultivariateNormalCholesky


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
        self.number_of_sub_diag_entries = self.sample_space_dimension * (self.sample_space_dimension - 1) // 2

    def get_layer(self):
        mu = Dense(self.sample_space_dimension, activation=None)
        chol_diag = Dense(self.sample_space_dimension, activation=tf.exp)
        chol_sub_diag = Dense(self.number_of_sub_diag_entries, activation=None)
        return merge([mu, chol_diag, chol_sub_diag], mode='concat')

    def get_tensor_random_variable(self, NN_prediction):
        # NN_prediction is expected to come from a layer such as the one produced
        # by get_layer. In particultar, it has to be of the following shape:
        # [batch_size, 2*self.sample_space_dimension+self.number_of_sub_diag_entries]
        mu = tf.slice(NN_prediction, [0, 0], [-1, self.sample_space_dimension])
        cholesky_diag = tf.slice(NN_prediction, [0, self.sample_space_dimension], [-1, 2 * self.sample_space_dimension])
        cholesky_sub_diag = tf.slice(NN_prediction, [0, 2 * self.sample_space_dimension], [-1, -1])
        cholesky = self.batch_to_lower_triangular_matrix(cholesky_diag, cholesky_sub_diag)
        return MultivariateNormalCholesky(mu, cholesky)

    def batch_to_lower_triangular_matrix(self, batch_diag, batch_sub_diag):
        # batch_diag, batch_sub_diag: [batch_size, *]
        # sltm = strictly_lower_triangular_matrix
        # ltm = lower_triangular_matrix
        batch_sltm = tf.map_fn(self.to_strictly_lower_triangular_matrix, batch_sub_diag)
        batch_diagonal_matrix = tf.map_fn(self.to_diagonal_matrix, batch_diag)
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


class MixtureOutputLayer:

    def __init__(self, components, sample_space_dimension):
        self.number_of_components = len(components)
        self.sample_space_dimension
        # TODO: finish here
