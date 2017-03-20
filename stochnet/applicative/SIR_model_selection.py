import os
import numpy as np
from copy import deepcopy
from stochnet.classes.TimeSeriesDataset import TimeSeriesDataset
from stochnet.classes.NeuralNetworks import StochNeuralNetwork
from stochnet.classes.TopLayers import MultivariateNormalCholeskyOutputLayer, MixtureOutputLayer
from keras.layers import Input, LSTM, Dense, Dropout
from keras.callbacks import EarlyStopping
from keras.constraints import maxnorm
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import RandomizedSearchCV

##### BROKEN BROKEN BROKEN #####

def create_stochnet(epochs=5, nb_LSTM_nodes=150, nb_Dense_nodes=75,
                    weight_constraint=None, dropout_rate=0.0, callbacks=None):

    input_tensor = Input(shape=(nb_past_timesteps, dataset.nb_features))
    hidden1 = LSTM(nb_LSTM_nodes, kernel_constraint=maxnorm(weight_constraint),
                   recurrent_constraint=maxnorm(weight_constraint))(input_tensor)
    dropout1 = Dropout(dropout_rate)(hidden1)
    NN_body = Dense(nb_Dense_nodes)(dropout1)
    NN = StochNeuralNetwork(input_tensor, NN_body, TopModel_obj)
    return NN.model


def report(results, n_top=1):
    for i in range(1, n_top + 1):
        candidates = np.flatnonzero(results['rank_test_score'] == i)
        for candidate in candidates:
            print("Model with rank: {0}".format(i))
            print("Mean validation score: {0:.3f} (std: {1:.3f})".format(
                  results['mean_test_score'][candidate],
                  results['std_test_score'][candidate]))
            print("Parameters: {0}".format(results['params'][candidate]))
            print("")
    return


# We need to get to the proper directory first
# We are in ./stochnet/applicative
# We want to be in ./stochnet
# os.getcwd returns the directory we are working in and we use dirname to get its parent directory
current = os.getcwd()
working_path = os.path.dirname(current)
basename = os.path.abspath(working_path)
dataset_address = os.path.join(basename, 'dataset/SIR_dataset.npy')
data_labels = {'Timestamps': 0, 'Susceptible': 1, 'Infected': 2, 'Removed': 3}

dataset = TimeSeriesDataset(dataset_address, labels=data_labels)

nb_past_timesteps = 5
dataset.format_dataset_for_ML(nb_past_timesteps=nb_past_timesteps, percentage_of_test_data=0.25)

number_of_components = 2
components = []
for j in range(number_of_components):
    components.append(MultivariateNormalCholeskyOutputLayer(dataset.nb_features))
TopModel_obj = MixtureOutputLayer(components)

stochnet = KerasRegressor(build_fn=create_stochnet, verbose=2)

param_dist = {"epochs": [6, 10, 15],
              "nb_LSTM_nodes": [30, 50],
              "nb_Dense_nodes": [30, 50],
              "weight_constraint": [1, 3, 5],
              "dropout_rate": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]}

nb_random_search = 1
random_search = RandomizedSearchCV(stochnet, param_distributions=param_dist,
                                   n_iter=nb_random_search, cv=2,
                                   verbose=5, refit=False, n_jobs=4,
                                   pre_dispatch='n_jobs', error_score='raise')

random_search.fit(dataset.X_train, dataset.y_train)

report(random_search.cv_results_, n_top=2)
