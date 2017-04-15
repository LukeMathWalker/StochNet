from tensorflow.contrib.distributions import MultivariateNormalCholesky as tf_MultivariateNormalCholesky
from tensorflow.contrib.distributions import Categorical as tf_Categorical
from tensorflow.contrib.distributions import Mixture as tf_Mixture
from tensorflow.contrib.distributions import bijector, TransformedDistribution
import tensorflow as tf
import numpy as np


class Categorical:

    def __init__(self, logits, validate_args=False):
        self.distribution_obj = tf_Categorical(logits=logits, validate_args=validate_args)
        self.number_of_classes = self.distribution_obj.num_classes

    def sample(self, sample_shape=()):
        return self.distribution_obj.sample(sample_shape=sample_shape)

    @property
    def nb_of_indipendent_random_variables(self):
        return np.array(self.distribution_obj.get_batch_shape()).prod()

    def get_description(self):
        descriptions = []
        with tf.Session():
            self.number_of_classes = self.number_of_classes.eval()
            flattened_class_probabilities = tf.reshape(self.distribution_obj.p, [-1, self.number_of_classes]).eval()
        description_preamble = "Categorical random variable with {0} classes.\n\n".format(self.number_of_classes)
        for j in range(self.nb_of_indipendent_random_variables):
            description = description_preamble + "\tClass probabilities: \n\t{0}\n\n".format(flattened_class_probabilities[j, :])
            descriptions.append(description)
        return descriptions


class MultivariateNormalCholesky:

    def __init__(self, mu, chol, validate_args=False):
        self.distribution_obj = tf_MultivariateNormalCholesky(mu, chol, validate_args=validate_args)

    def log_prob(self, value):
        return self.distribution_obj.log_prob(value)

    @property
    def mean(self):
        return self.distribution_obj.mu

    @property
    def covariance(self):
        return self.distribution_obj.sigma

    @property
    def sample_space_dimension(self):
        return self.distribution_obj.get_event_shape().as_list()[0]

    def sample(self, sample_shape=()):
        return self.distribution_obj.sample(sample_shape=sample_shape)

    @property
    def nb_of_indipendent_random_variables(self):
        return np.array(self.distribution_obj.get_batch_shape()).prod()

    def get_description(self):
        descriptions = []
        with tf.Session():
            flattened_means = tf.reshape(self.mean, [-1, self.sample_space_dimension]).eval()
            flattened_sigmas = tf.reshape(self.covariance, [-1, self.sample_space_dimension, self.sample_space_dimension]).eval()
        description_preamble = "Multivariate Normal random variable.\n\n"
        for j in range(self.nb_of_indipendent_random_variables):
            description = description_preamble + "\tMean:\n\t\t{0}\n\tCovariance matrix:\n\t\t{1}\n".format(flattened_means[j, :], flattened_sigmas[j, ...])
            descriptions.append(description)
        return descriptions


class MultivariateLogNormal:

    def __init__(self, mu, chol, validate_args=False):
        self.distribution_obj = TransformedDistribution(distribution=tf_MultivariateNormalCholesky(mu, chol),
                                                bijector=bijector.Inline(
                                                    forward_fn=tf.exp,
                                                    inverse_fn=tf.log,
                                                    inverse_log_det_jacobian_fn=(lambda y: -tf.reduce_sum(tf.log(y), reduction_indices=-1))),
                                                name="LogNormalTransformedDistribution")
        self.normal_mean = mu
        self.normal_covariance = chol

    def log_prob(self, value):
        with tf.control_dependencies([tf.assert_positive(value)]):
            return self.distribution_obj.log_prob(value)

    @property
    def sample_space_dimension(self):
        return self.distribution_obj.get_event_shape().as_list()[0]

    def sample(self, sample_shape=()):
        return self.distribution_obj.sample(sample_shape=sample_shape)

    @property
    def nb_of_indipendent_random_variables(self):
        return np.array(self.distribution_obj.get_batch_shape()).prod()

    def get_description(self):
        descriptions = []
        with tf.Session():
            flattened_means = tf.reshape(self.normal_mean, [-1, self.sample_space_dimension]).eval()
            flattened_sigmas = tf.reshape(self.normal_covariance, [-1, self.sample_space_dimension, self.sample_space_dimension]).eval()
        description_preamble = "Multivariate Log-Normal random variable.\n\n"
        for j in range(self.nb_of_indipendent_random_variables):
            description = description_preamble + "\tNormal mean:\n\t\t{0}\n\tNormal covariance matrix:\n\t\t{1}\n".format(flattened_means[j, :], flattened_sigmas[j, ...])
            descriptions.append(description)
        return descriptions


# class MultivariateLogNormal_2:
#
#     def __init__(self, normal_mean, normal_cholesky, validate_args=False):
#         self.distribution_obj = TransformedDistribution(distribution=tf_MultivariateNormalCholesky(normal_mean, normal_cholesky),
#                                                         bijector=bijector.Inline(
#                                                             forward_fn=tf.exp,
#                                                             inverse_fn=tf.log,
#                                                             inverse_log_det_jacobian_fn=(lambda y: -tf.reduce_sum(tf.log(y), reduction_indices=-1))),
#                                                         name="LogNormalTransformedDistribution")
#         self.normal_mean = normal_mean
#         self.normal_cholesky = normal_cholesky
#
#     def log_prob(self, value):
#         MNC = MultivariateNormalCholesky(self.normal_mean, self.normal_cholesky)
#         with tf.control_dependencies([tf.assert_positive(value)]):
#             MNC_log_prob = MNC.log_prob(tf.log(value))
#         log_prob = MNC_log_prob - tf.reduce_sum(tf.log(value), reduction_indices=-1)
#         return log_prob
#
#     @property
#     def sample_space_dimension(self):
#         return self.distribution_obj.get_event_shape().as_list()[0]
#
#     def sample(self):
#         return self.distribution_obj.sample()
#
#     @property
#     def nb_of_indipendent_random_variables(self):
#         return np.array(self.distribution_obj.get_batch_shape()).prod()
#
#     def get_description(self):
#         descriptions = []
#         with tf.Session():
#             flattened_means = tf.reshape(self.normal_mean, [-1, self.sample_space_dimension]).eval()
#             flattened_sigmas = tf.reshape(self.normal_covariance, [-1, self.sample_space_dimension, self.sample_space_dimension]).eval()
#         description_preamble = "Multivariate Log-Normal random variable.\n\n"
#         for j in range(self.nb_of_indipendent_random_variables):
#             description = description_preamble + "\tNormal mean:\n\t\t{0}\n\tNormal covariance matrix:\n\t\t{1}\n".format(flattened_means[j, :], flattened_sigmas[j, ...])
#             descriptions.append(description)
#         return descriptions


class Mixture:

    def __init__(self, cat, components, validate_args=False):
        self.cat = cat
        self.components = list(components)
        self.number_of_components = len(components)
        tf_cat = cat.distribution_obj
        tf_components = [component.distribution_obj for component in components]
        self.distribution_obj = tf_Mixture(tf_cat, tf_components, validate_args=validate_args)

    def log_prob(self, value):
        return self.distribution_obj.log_prob(value)

    def sample(self, sample_shape=()):
        return self.distribution_obj.sample(sample_shape=sample_shape)

    @property
    def nb_of_indipendent_random_variables(self):
        return np.array(self.distribution_obj.get_batch_shape()).prod()

    def get_description(self):
        descriptions = []
        description_preamble = "Mixture random variable with {0} components.\n\n".format(self.number_of_components)
        cat_descriptions = self.cat.get_description()
        component_descriptions = [component.get_description() for component in self.components]
        for j in range(self.nb_of_indipendent_random_variables):
            description = description_preamble + cat_descriptions[j]
            for component_description in component_descriptions:
                description += component_description[j]
            descriptions.append(description)
        return descriptions
        # TODO: improve text formatting
