import tensorflow as tf
import numpy as np
import abc
from keras.layers import Dense
from keras.layers.merge import concatenate
from stochnet.classes.Tensor_RandomVariables import Categorical, MultivariateNormalCholesky, MultivariateLogNormal, Mixture, MultivariateNormalDiag
from stochnet.classes.Errors import ShapeError, DimensionError


class RandomVariableOutputLayer(abc.ABC):

    @abc.abstractmethod
    def add_layer_on_top(self, base_model):
        """Adds a layer on top of an existing neural network model which allows
        to learn those parameters which are needed to instantiate a random
        variable of the family indicated in the class name.
        The top layer output tensor has the following shape:
        [batch_size, self.number_of_output_neurons]
        """

    @abc.abstractmethod
    def get_tensor_random_variable(self, NN_prediction):
        """Given the values of a tensor produced by a neural network with a layer
        on top of the form of the one provided by the add_layer_on_top method,
        it returns an instance of the corresponding tensor random variable class
        inizialized using the parameters provived by the neural network output.
        Additional checks might be needed for certain families of random variables.
        """

    @abc.abstractmethod
    def sample(self, NN_prediction, sample_shape=(), sess=None):
        """Get tensor random random variable correspinding to the parameters provided by
        NN_prediction and sample from those.
        The method returns a numpy array with the following shape:
        [number_of_samples, self.sample_space_dimension]"""
        self.check_NN_prediction_shape(NN_prediction)
        if sess is None:
            with tf.Session():
                samples = self.get_tensor_random_variable(NN_prediction).sample(sample_shape=sample_shape).eval()
        else:
            samples = sess.run(self.get_tensor_random_variable(NN_prediction).sample(sample_shape=sample_shape))
        return samples

    @abc.abstractmethod
    def check_NN_prediction_shape(self, NN_prediction):
        """The method check that NN_prediction has the following shape:
        [batch_size, number_of_output_neurons]
        """
        NN_prediction_shape = list(NN_prediction.shape)
        if len(NN_prediction_shape) != 2 or NN_prediction_shape[1] != self.number_of_output_neurons:
            raise ShapeError("The neural network predictions passed as input "
                             "are required to be of the following shape: "
                             "[batch_size, number_of_output_neurons].\n"
                             "Your neural network predictions had the following "
                             "shape: {0}".format(NN_prediction_shape))

    @abc.abstractmethod
    def get_description(self, NN_prediction):
        """Return a string containing a description of the random variables inizialized
        using the parameters in NN_prediction"""
        random_variable = self.get_tensor_random_variable(NN_prediction)
        description = random_variable.get_description()
        return description
        # TODO: fix


class CategoricalOutputLayer(RandomVariableOutputLayer):

    sample_space_dimension = 1

    def __init__(self, number_of_classes):
        # it calls the setter method implicitly
        self.number_of_classes = number_of_classes

    @property
    def number_of_classes(self):
        return self._number_of_classes

    @number_of_classes.setter
    def number_of_classes(self, new_number_of_classes):
        # _number_of_classes and number_of_output_neurons do need to be coincide
        if new_number_of_classes > 0:
            self._number_of_classes = new_number_of_classes
            self.number_of_output_neurons = new_number_of_classes
        else:
            raise ValueError('''We can't define a Categorical random variable is there isn't at least one class!''')

    def add_layer_on_top(self, base_model):
        logits_on_top = Dense(self._number_of_classes, activation=None)(base_model)
        return logits_on_top

    def get_tensor_random_variable(self, NN_prediction):
        self.check_NN_prediction_shape(NN_prediction)
        return Categorical(NN_prediction)

    def sample(self, NN_prediction, sample_shape=(), sess=None):
        return super().sample(NN_prediction, sample_shape, sess)

    def get_description(self, NN_prediction):
        return super().get_description(NN_prediction)

    def check_NN_prediction_shape(self, NN_prediction):
        super().check_NN_prediction_shape(NN_prediction)

    def loss_function(self, y_true, y_pred):
        loss = tf.nn.softmax_cross_entropy_with_logits(labels=y_true, logits=y_pred)
        return loss

    def log_likelihood_function(self, y_true, y_pred):
        log_likelihood = -tf.nn.softmax_cross_entropy_with_logits(labels=y_true, logits=y_pred)
        return log_likelihood


class MultivariateNormalDiagOutputLayer(RandomVariableOutputLayer):

    def __init__(self, sample_space_dimension, mu_regularizer=None, diag_regularizer=None):
        self.sample_space_dimension = sample_space_dimension
        self.mu_regularizer = mu_regularizer
        self.diag_regularizer = diag_regularizer

    @property
    def sample_space_dimension(self):
        return self._sample_space_dimension

    @sample_space_dimension.setter
    def sample_space_dimension(self, new_sample_space_dimension):
        if new_sample_space_dimension > 1:
            self._sample_space_dimension = new_sample_space_dimension
            self.number_of_output_neurons = 2 * self._sample_space_dimension
        else:
            raise ValueError('''The sample space dimension for a multivariate normal variable needs to be at least two!''')

    def add_layer_on_top(self, base_model):
        mu = Dense(self._sample_space_dimension, activation=None, activity_regularizer=self.mu_regularizer)(base_model)
        diag = Dense(self._sample_space_dimension, activation=tf.exp, activity_regularizer=self.diag_regularizer)(base_model)
        return concatenate([mu, diag], axis=-1)

    def get_tensor_random_variable(self, NN_prediction):
        self.check_NN_prediction_shape(NN_prediction)
        mu = tf.slice(NN_prediction, [0, 0], [-1, self._sample_space_dimension])
        diag = tf.slice(NN_prediction, [0, self._sample_space_dimension], [-1, self._sample_space_dimension])
        return MultivariateNormalDiag(mu, diag)

    def sample(self, NN_prediction, sample_shape=(), sess=None):
        return super().sample(NN_prediction, sample_shape, sess)

    def get_description(self, NN_prediction):
        return super().get_description(NN_prediction)

    def check_NN_prediction_shape(self, NN_prediction):
        super().check_NN_prediction_shape(NN_prediction)

    def loss_function(self, y_true, y_pred):
        return -self.get_tensor_random_variable(y_pred).log_prob(y_true)

    def log_likelihood(self, y_true, y_pred):
        return self.get_tensor_random_variable(y_pred).log_prob(y_true)


class MultivariateNormalCholeskyOutputLayer(RandomVariableOutputLayer):

    def __init__(self, sample_space_dimension):
        self.sample_space_dimension = sample_space_dimension

    @property
    def sample_space_dimension(self):
        return self._sample_space_dimension

    @sample_space_dimension.setter
    def sample_space_dimension(self, new_sample_space_dimension):
        if new_sample_space_dimension > 1:
            self._sample_space_dimension = new_sample_space_dimension
            self.number_of_sub_diag_entries = self._sample_space_dimension * (self._sample_space_dimension - 1) // 2
            self.number_of_output_neurons = 2 * self._sample_space_dimension + self.number_of_sub_diag_entries
        else:
            raise ValueError('''The sample space dimension for a multivariate normal variable needs to be at least two!''')

    def add_layer_on_top(self, base_model):
        mu = Dense(self._sample_space_dimension, activation=None)(base_model)
        chol_diag = Dense(self._sample_space_dimension, activation=tf.exp)(base_model)
        chol_sub_diag = Dense(self.number_of_sub_diag_entries, activation=None)(base_model)
        return concatenate([mu, chol_diag, chol_sub_diag], axis=-1)

    def get_tensor_random_variable(self, NN_prediction):
        self.check_NN_prediction_shape(NN_prediction)
        mu = tf.slice(NN_prediction, [0, 0], [-1, self._sample_space_dimension])
        cholesky_diag = tf.slice(NN_prediction, [0, self._sample_space_dimension], [-1, self._sample_space_dimension])
        cholesky_sub_diag = tf.slice(NN_prediction, [0, 2 * self._sample_space_dimension], [-1, self.number_of_sub_diag_entries])
        with tf.control_dependencies([tf.assert_positive(cholesky_diag)]):
            cholesky = self.batch_to_lower_triangular_matrix(cholesky_diag, cholesky_sub_diag)
        return MultivariateNormalCholesky(mu, cholesky)

    def sample(self, NN_prediction, sample_shape=(), sess=None):
        return super().sample(NN_prediction, sample_shape, sess)

    def get_description(self, NN_prediction):
        return super().get_description(NN_prediction)

    def check_NN_prediction_shape(self, NN_prediction):
        super().check_NN_prediction_shape(NN_prediction)

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
        sltm__indices = list(zip(*np.tril_indices(self._sample_space_dimension, -1)))
        sltm__indices = tf.constant([list(i) for i in sltm__indices], dtype=tf.int64)
        sltm = tf.sparse_to_dense(sparse_indices=sltm__indices,
                                  output_shape=[self._sample_space_dimension, self._sample_space_dimension],
                                  sparse_values=sub_diag,
                                  default_value=0,
                                  validate_indices=True)
        return sltm

    def to_diagonal_matrix(self, diag):
        return tf.diag(diag)

    def loss_function(self, y_true, y_pred):
        return -self.get_tensor_random_variable(y_pred).log_prob(y_true)

    def log_likelihood(self, y_true, y_pred):
        return self.get_tensor_random_variable(y_pred).log_prob(y_true)


class MultivariateLogNormalOutputLayer(RandomVariableOutputLayer):
    # TODO: eliminate the duplicated code between MultivariteNormalCholesky and MultivariateLogNormal
    def __init__(self, sample_space_dimension):
        self.sample_space_dimension = sample_space_dimension

    @property
    def sample_space_dimension(self):
        return self._sample_space_dimension

    @sample_space_dimension.setter
    def sample_space_dimension(self, new_sample_space_dimension):
        if new_sample_space_dimension > 1:
            self._sample_space_dimension = new_sample_space_dimension
            self.number_of_sub_diag_entries = self._sample_space_dimension * (self._sample_space_dimension - 1) // 2
            self.number_of_output_neurons = 2 * self._sample_space_dimension + self.number_of_sub_diag_entries
        else:
            raise ValueError('''The sample space dimension for a multivariate log-normal variable needs to be at least two!''')

    def add_layer_on_top(self, base_model):
        normal_mean = Dense(self._sample_space_dimension, activation=None)(base_model)
        normal_chol_diag = Dense(self._sample_space_dimension, activation=tf.exp)(base_model)
        normal_chol_sub_diag = Dense(self.number_of_sub_diag_entries, activation=None)(base_model)
        return concatenate([normal_mean, normal_chol_diag, normal_chol_sub_diag], axis=-1)

    def get_tensor_random_variable(self, NN_prediction):
        self.check_NN_prediction_shape(NN_prediction)
        normal_mean = tf.slice(NN_prediction, [0, 0], [-1, self._sample_space_dimension])
        normal_cholesky_diag = tf.slice(NN_prediction, [0, self._sample_space_dimension], [-1, self._sample_space_dimension])
        normal_cholesky_sub_diag = tf.slice(NN_prediction, [0, 2 * self._sample_space_dimension], [-1, self.number_of_sub_diag_entries])
        with tf.control_dependencies([tf.assert_positive(normal_cholesky_diag)]):
            normal_cholesky = self.batch_to_lower_triangular_matrix(normal_cholesky_diag, normal_cholesky_sub_diag)
        return MultivariateLogNormal(normal_mean, normal_cholesky)

    def sample(self, NN_prediction, sample_shape=(), sess=None):
        return super().sample(NN_prediction, sample_shape, sess)

    def get_description(self, NN_prediction):
        return super().get_description(NN_prediction)

    def check_NN_prediction_shape(self, NN_prediction):
        super().check_NN_prediction_shape(NN_prediction)

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
        sltm__indices = list(zip(*np.tril_indices(self._sample_space_dimension, -1)))
        sltm__indices = tf.constant([list(i) for i in sltm__indices], dtype=tf.int64)
        sltm = tf.sparse_to_dense(sparse_indices=sltm__indices,
                                  output_shape=[self._sample_space_dimension, self._sample_space_dimension],
                                  sparse_values=sub_diag,
                                  default_value=0,
                                  validate_indices=True)
        return sltm

    def to_diagonal_matrix(self, diag):
        return tf.diag(diag)

    def loss_function(self, y_true, y_pred):
        return -self.get_tensor_random_variable(y_pred).log_prob(y_true)

    def log_likelihood(self, y_true, y_pred):
        return self.get_tensor_random_variable(y_pred).log_prob(y_true)


class MixtureOutputLayer(RandomVariableOutputLayer):

    def __init__(self, components):
        self.number_of_components = len(components)
        self.categorical = CategoricalOutputLayer(self.number_of_components)
        self.components = list(components)
        self.set_sample_space_dimension()
        self.set_number_of_output_neurons()

    def set_sample_space_dimension(self):
        sample_space_dims = [component.sample_space_dimension for component in self.components]
        if all(x == sample_space_dims[0] for x in sample_space_dims):
            self.sample_space_dimension = sample_space_dims[0]
        else:
            raise DimensionError("The random variables which have been passed "
                                 "as mixture components sample from spaces with "
                                 "different dimensions.\n"
                                 "This is the list of sample spaces dimensions:\n"
                                 "{0}".format(sample_space_dims))

    def set_number_of_output_neurons(self):
        self.number_of_output_neurons = self.categorical.number_of_output_neurons
        for component in self.components:
            self.number_of_output_neurons += component.number_of_output_neurons

    def add_layer_on_top(self, base_model):
        # list comprehension preserves the order of the original list.
        categorical_layer = self.categorical.add_layer_on_top(base_model)
        components_layers = [component.add_layer_on_top(base_model) for component in self.components]
        mixture_layers = [categorical_layer] + components_layers
        return concatenate(mixture_layers, axis=-1)

    def get_tensor_random_variable(self, NN_prediction):
        self.check_NN_prediction_shape(NN_prediction)
        categorical_predictions = tf.slice(NN_prediction, [0, 0], [-1, self.categorical.number_of_output_neurons])
        categorical_random_variable = self.categorical.get_tensor_random_variable(categorical_predictions)
        components_random_variable = []
        start_slicing_index = self.categorical.number_of_output_neurons
        for component in self.components:
            component_predictions = tf.slice(NN_prediction, [0, start_slicing_index], [-1, component.number_of_output_neurons])
            component_random_variable = component.get_tensor_random_variable(component_predictions)
            components_random_variable.append(component_random_variable)
            start_slicing_index += component.number_of_output_neurons
        return Mixture(categorical_random_variable, components_random_variable)

    def sample(self, NN_prediction, sample_shape=(), sess=None):
        return super().sample(NN_prediction, sample_shape, sess)

    def get_description(self, NN_prediction):
        return super().get_description(NN_prediction)

    def check_NN_prediction_shape(self, NN_prediction):
        super().check_NN_prediction_shape(NN_prediction)

    def loss_function(self, y_true, y_pred):
        return -self.get_tensor_random_variable(y_pred).log_prob(y_true)

    def log_likelihood_function(self, y_true, y_pred):
        return self.get_tensor_random_variable(y_pred).log_prob(y_true)
