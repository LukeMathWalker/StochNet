from tensorflow.contrib.distributions import MultivariateNormalCholesky as tf_MultivariateNormalCholesky
from tensorflow.contrib.distributions import Categorical as tf_Categorical
from tensorflow.contrib.distributions import Mixture as tf_Mixture
import tensorflow as tf
import numpy as np


class Categorical:

    def __init__(self, logits, validate_args=False):
        self.distribution_obj = tf_Categorical(logits=logits, validate_args=validate_args)
        self.number_of_classes = self.distribution_obj.num_classes

    def sample(self):
        return self.distribution_obj.sample()

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

    def sample(self):
        return self.distribution_obj.sample()

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

    def sample(self):
        return self.distribution_obj.sample()

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
