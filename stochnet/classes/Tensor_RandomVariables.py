from tensorflow.contrib.distributions import MultivariateNormalCholesky as tf_MultivariateNormalCholesky
from tensorflow.contrib.distributions import Categorical as tf_Categorical
from tensorflow.contrib.distributions import Mixture as tf_Mixture


class Categorical:

    def __init__(self, probs, validate_args=False):
        self.distribution_obj = tf_Categorical(p=probs, validate_args=validate_args)
        self.number_of_classes = self.distribution_obj.event_size


class MultivariateNormalCholesky:

    def __init__(self, mu, chol, validate_args=False):
        self.distribution_obj = tf_MultivariateNormalCholesky(mu, chol, validate_args=validate_args)


class Mixture:

    def __init__(self, cat, components, validate_args=False):
        self.distribution_obj = tf_Mixture(cat, components, validate_args=validate_args)

    def log_prob(self, value):
        return self.distribution_obj.log_prob(value)
