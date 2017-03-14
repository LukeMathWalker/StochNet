import tensorflow as tf
import random
import numpy as np
from stochnet.classes.Tensor_RandomVariables import Categorical, MultivariateNormalCholesky


class Test_Categorical_with_Valid_Input(tf.test.TestCase):

    def setUp(self):
        batch_size = random.randrange(25)
        self.batch_shape = (batch_size, batch_size)
        number_of_classes = random.randrange(15)
        self.tensor_shape = (batch_size, batch_size, number_of_classes)
        logits = tf.random_uniform(self.tensor_shape, minval=-2**7, maxval=2**7, dtype=tf.float32)
        self.categorical = Categorical(logits=logits)

    def test_that_sample_returns_a_correctly_shaped_sample(self):
        sample = self.categorical.sample()
        self.assertIsInstance(sample, tf.Tensor)
        self.assertIsInstance(sample.shape, tf.TensorShape)
        self.assertEqual(sample.shape, self.batch_shape)

    def test_that_nb_of_indipendent_random_variables_works_properly(self):
        correct_nb_of_indipendent_rv = np.array(self.batch_shape).prod()
        self.assertEqual(self.categorical.nb_of_indipendent_random_variables,
                         correct_nb_of_indipendent_rv)

    def test_get_description_returns_a_list_with_the_correct_length(self):
        descriptions = self.categorical.get_description()
        nb_of_indipendent_rv = self.categorical.nb_of_indipendent_random_variables
        self.assertEqual(len(descriptions), nb_of_indipendent_rv)


class Test_MultivariateNormalCholesky_with_Valid_Input(tf.test.TestCase):

    def setUp(self):
        batch_size = random.randrange(25)
        self.batch_shape = (batch_size, batch_size)
        self.sample_space_dimension = random.randrange(2, 10)
        self.mu_shape = (batch_size, batch_size, self.sample_space_dimension)
        self.sigma_shape = (batch_size, batch_size, self.sample_space_dimension, self.sample_space_dimension)
        mu = tf.random_uniform(self.mu_shape, minval=-2**7, maxval=2**7, dtype=tf.float32)
        diagonal = tf.random_uniform(self.mu_shape, minval=2**(-2), maxval=2**7, dtype=tf.float32)
        cholesky = tf.matrix_diag(diagonal)
        self.MNC = MultivariateNormalCholesky(mu=mu, chol=cholesky)

    def test_that_sample_returns_a_correctly_shaped_sample(self):
        sample = self.MNC.sample()
        self.assertIsInstance(sample, tf.Tensor)
        self.assertIsInstance(sample.shape, tf.TensorShape)
        self.assertEqual(sample.shape, self.batch_shape + (self.sample_space_dimension,))

    def test_that_nb_of_indipendent_random_variables_works_properly(self):
        correct_nb_of_indipendent_rv = np.array(self.batch_shape).prod()
        self.assertEqual(self.MNC.nb_of_indipendent_random_variables,
                         correct_nb_of_indipendent_rv)

    def test_sample_space_dimension_is_correct(self):
        self.assertIsInstance(self.MNC.sample_space_dimension, int)
        self.assertEqual(self.MNC.sample_space_dimension, self.sample_space_dimension)

    def test_get_description_returns_a_list_with_the_correct_length(self):
        descriptions = self.MNC.get_description()
        nb_of_indipendent_rv = self.MNC.nb_of_indipendent_random_variables
        self.assertEqual(len(descriptions), nb_of_indipendent_rv)
