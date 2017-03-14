import tensorflow as tf
from stochnet.classes.TopLayers import CategoricalOutputLayer, MultivariateNormalCholeskyOutputLayer, MixtureOutputLayer
from stochnet.classes.Errors import ShapeError, DimensionError
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
        batch_size = random.randrange(100)
        tensor_shape = (batch_size, self.categorical_output_layer.number_of_output_neurons)
        logits = tf.random_uniform(tensor_shape, maxval=2**7, dtype=tf.float32)
        correct_labels = tf.random_uniform(tensor_shape, maxval=self.number_of_classes, dtype=tf.int32)
        with self.test_session():
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

    def test_get_random_tensor_variable_using_input_without_batch_dimension(self):
        number_of_classes = random.randrange(100)
        categorical_output_layer = CategoricalOutputLayer(number_of_classes)
        tensor_shape = (number_of_classes,)
        NN_prediction_with_invalid_shape = tf.random_uniform(tensor_shape, maxval=2**7, dtype=tf.float32)
        with self.assertRaises(ShapeError):
            categorical_output_layer.get_tensor_random_variable(NN_prediction_with_invalid_shape)

    def test_get_random_tensor_variable_using_input_with_wrong_number_of_output_neurons(self):
        number_of_classes = random.randrange(50, 100)
        batch_size = random.randrange(100)
        categorical_output_layer = CategoricalOutputLayer(number_of_classes)
        tensor_shape = (batch_size, number_of_classes + random.randrange(2, 25))
        NN_prediction_with_invalid_shape = tf.random_uniform(tensor_shape, maxval=2**7, dtype=tf.float32)
        with self.assertRaises(ShapeError):
            categorical_output_layer.get_tensor_random_variable(NN_prediction_with_invalid_shape)


class Test_MultivariateNormalCholeskyOutputLayer_with_Valid_Input(tf.test.TestCase):

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


class Test_MultivariateNormalCholeskyOutputLayer_with_Invalid_Input(tf.test.TestCase):

    def test_init_using_invalid_sample_space_dimension(self):
        for invalid_sample_space_dimension in range(1, -5):
            with self.assertRaises(ValueError):
                MultivariateNormalCholeskyOutputLayer(invalid_sample_space_dimension)

    def test_get_random_tensor_variable_using_input_without_batch_dimension(self):
        sample_space_dimension = random.randrange(100)
        MNC_output_layer = MultivariateNormalCholeskyOutputLayer(sample_space_dimension)
        tensor_shape = (sample_space_dimension,)
        NN_prediction_with_invalid_shape = tf.random_uniform(tensor_shape, maxval=2**7, dtype=tf.float32)
        with self.assertRaises(ShapeError):
            MNC_output_layer.get_tensor_random_variable(NN_prediction_with_invalid_shape)

    def test_get_random_tensor_variable_using_input_with_wrong_number_of_output_neurons(self):
        sample_space_dimension = random.randrange(50, 100)
        batch_size = random.randrange(100)
        MNC_output_layer = MultivariateNormalCholeskyOutputLayer(sample_space_dimension)
        tensor_shape = (batch_size, MNC_output_layer.number_of_output_neurons + random.randrange(2, 25))
        NN_prediction_with_invalid_shape = tf.random_uniform(tensor_shape, maxval=2**7, dtype=tf.float32)
        with self.assertRaises(ShapeError):
            MNC_output_layer.get_tensor_random_variable(NN_prediction_with_invalid_shape)

    def test_get_random_tensor_variable_using_a_non_positive_cholesky_diagonal(self):
        with self.test_session():
            sample_space_dimension = random.randrange(50, 100)
            batch_size = random.randrange(100)
            MNC_output_layer = MultivariateNormalCholeskyOutputLayer(sample_space_dimension)
            tensor_shape = (batch_size, MNC_output_layer.number_of_output_neurons)
            NN_prediction_with_negative_cholesky_diag = tf.random_uniform(tensor_shape,
                                                                          minval=-2**7,
                                                                          maxval=0,
                                                                          dtype=tf.float32)
            random_var = MNC_output_layer.get_tensor_random_variable(NN_prediction_with_negative_cholesky_diag)
            with self.assertRaises(tf.errors.InvalidArgumentError):
                # The InvalidArgumentError is raised only when the tensorflow
                # graph is actually executed, that's why we need to explicitly
                # ask for the MNC covariance matrix.
                random_var.covariance.eval()


class Test_MixtureOutputLayer_with_Invalid_Input(tf.test.TestCase):

    def test_init_using_components_sampling_from_spaces_with_different_dimensions(self):
        components = []
        for j in range(2, 5):
            components.append(MultivariateNormalCholeskyOutputLayer(j))
        with self.assertRaises(DimensionError):
            MixtureOutputLayer(components)

    def test_init_using_an_empty_list_of_components(self):
        with self.assertRaises(ValueError):
            MixtureOutputLayer([])
