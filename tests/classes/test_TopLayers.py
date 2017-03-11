import unittest
import tensorflow as tf
from stochnet.classes.TopLayers import CategoricalOutputLayer, MultivariateNormalCholeskyOutputLayer
from keras.layers import Input
import random

# TODO: find a way to write "common tests" only once


class Test_CategoricalOutputLayer_with_Valid_Input(tf.test.TestCase):

    def setUp(self):
        self.number_of_classes = random.randrange(100)
        self.categorical_output_layer = CategoricalOutputLayer(self.number_of_classes)

    def test_that_changing_number_of_classes_changes_number_of_output_neurons_accordingly(self):
        new_number_of_classes = random.randrange(100)
        self.categorical_output_layer.number_of_classes = new_number_of_classes
        self.assertEqual(self.categorical_output_layer.number_of_classes,
                         self.categorical_output_layer.number_of_output_neurons)

    def test_that_add_layer_on_top_returns_a_layer_with_the_correct_shape(self):
        input_tensor = Input(shape=(random.randrange(100),))
        output_layer = self.categorical_output_layer.add_layer_on_top(input_tensor)
        self.assertEqual(output_layer.shape.as_list(), [None, self.categorical_output_layer.number_of_output_neurons])

    def test_that_loss_function_returns_a_correctly_shaped_tensor(self):
        with self.test_session():
            batch_size = random.randrange(100)
            tensor_shape = (batch_size, self.categorical_output_layer.number_of_output_neurons)
            logits = tf.random_uniform(tensor_shape, maxval=2**7, dtype=tf.float32)
            correct_labels = tf.random_uniform(tensor_shape, maxval=self.number_of_classes, dtype=tf.int32)
            loss = self.categorical_output_layer.loss_function(correct_labels, logits).eval()
            self.assertEqual(len(loss.shape), 1)
            self.assertEqual(loss.shape[0], batch_size)


class Test_CategoricalOutputLayer_with_Invalid_Input(tf.test.TestCase):

    def test_init_using_zero_classes(self):
        number_of_classes = 0
        with self.assertRaises(ValueError):
            CategoricalOutputLayer(number_of_classes)

    def test_init_using_a_negative_number_of_classes(self):
        number_of_classes = -random.randrange(100)
        with self.assertRaises(ValueError):
            CategoricalOutputLayer(number_of_classes)


class Test_MultivariateNormalCholeskyOutputLayer_with_Valid_Input(unittest.TestCase):

    def setUp(self):
        self.sample_space_dimension = random.randrange(100)
        self.MNC_output_layer = MultivariateNormalCholeskyOutputLayer(self.sample_space_dimension)

    def test_that_changing_sample_space_dimension_doesnt_break_parameter_coherence(self):
        new_sample_space_dimension = random.randrange(100)
        self.MNC_output_layer.sample_space_dimension = new_sample_space_dimension
        new_correct_number_of_sub_diag_entries = new_sample_space_dimension * (new_sample_space_dimension - 1) // 2
        new_correct_number_of_output_neurons = 2 * new_sample_space_dimension + new_correct_number_of_sub_diag_entries
        self.assertEqual(self.MNC_output_layer.number_of_sub_diag_entries, new_correct_number_of_sub_diag_entries)
        self.assertEqual(self.MNC_output_layer.number_of_output_neurons, new_correct_number_of_output_neurons)

    def test_that_add_layer_on_top_returns_a_layer_with_the_correct_shape(self):
        input_tensor = Input(shape=(random.randrange(100),))
        output_layer = self.MNC_output_layer.add_layer_on_top(input_tensor)
        self.assertEqual(output_layer.shape.as_list(), [None, self.MNC_output_layer.number_of_output_neurons])
