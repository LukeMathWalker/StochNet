import os
from tensorflow.python import debug as tf_debug
from stochnet.classes.TimeSeriesDataset import TimeSeriesDataset
from stochnet.classes.NeuralNetworks import StochNeuralNetwork
from stochnet.classes.TopLayers import MultivariateNormalCholeskyOutputLayer, MultivariateLogNormalOutputLayer, MixtureOutputLayer
from keras.layers import Input, LSTM, Dense
from keras.callbacks import EarlyStopping
from keras.constraints import maxnorm
from keras import backend as K
import tensorflow as tf

# sess = tf.Session()
# sess = tf_debug.LocalCLIDebugWrapperSession(sess)
# K.set_session(sess)

# We need to get to the proper directory first
# We are in ./stochnet/applicative
# We want to be in ./stochnet
# os.getcwd returns the directory we are working in and we use dirname to get its parent directory
current = os.getcwd()
working_path = os.path.dirname(current)
basename = os.path.abspath(working_path)
dataset_address = os.path.join(basename, 'dataset/SIR_dataset_upgraded.npy')
data_labels = {'Timestamps': 0, 'Susceptible': 1, 'Infected': 2, 'Removed': 3}

dataset = TimeSeriesDataset(dataset_address, labels=data_labels)

nb_past_timesteps = 2
dataset.format_dataset_for_ML(nb_past_timesteps=nb_past_timesteps, must_be_rescaled=True, positivity='needed', percentage_of_test_data=0.25)

input_tensor = Input(shape=(nb_past_timesteps, dataset.nb_features))
hidden1 = LSTM(150, kernel_constraint=maxnorm(5), recurrent_constraint=maxnorm(5))(input_tensor)
NN_body = Dense(75, kernel_constraint=maxnorm(5))(hidden1)

number_of_components = 2
components = []
for j in range(number_of_components):
    components.append(MultivariateLogNormalOutputLayer(dataset.nb_features))
TopModel_obj = MixtureOutputLayer(components)

#TopModel_obj = MultivariateLogNormalOutputLayer(dataset.nb_features)

NN = StochNeuralNetwork(input_tensor, NN_body, TopModel_obj)
callbacks = [EarlyStopping(monitor='val_loss', patience=2, verbose=1, mode='min')]
NN.fit(dataset.X_train, dataset.y_train, epochs=10, validation_split=0.2, callbacks=callbacks)

test_set_prediction = NN.predict(dataset.X_test)
NN.visualize_performance_by_sampling(dataset.X_test, dataset.y_test, test_set_prediction,
                                     max_display=2, fitted_scaler=dataset.scaler,
                                     feature_labels=dataset.labels)
