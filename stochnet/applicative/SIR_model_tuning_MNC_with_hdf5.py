import os
import json
from tabulate import tabulate
from hyperopt import Trials, STATUS_OK, tpe
from hyperas import optim
from hyperas.distributions import choice, uniform
from keras.layers import Dense, Dropout, Input, LSTM, Flatten
from keras.constraints import maxnorm
from keras.callbacks import EarlyStopping, ModelCheckpoint
from stochnet.classes.TimeSeriesDataset import TimeSeriesDataset
from stochnet.classes.NeuralNetworks import StochNeuralNetwork
from stochnet.classes.TopLayers import MultivariateNormalCholeskyOutputLayer, MixtureOutputLayer
from stochnet.utils.iterator import HDF5Iterator


def data():
    '''
    Data providing function:

    This function is separated from model() so that hyperopt
    won't reload data for each evaluation run.
    '''

    dataset_address = '/home/lpalmier/workspace/Data/SIR/SIR_dataset_timestep_2-5_06.hdf5'
    validation_dataset_address = '/home/lpalmier/workspace/Data/SIR/SIR_dataset_timestep_2-5_02.hdf5'

    nb_past_timesteps = 1

    dataset = TimeSeriesDataset(dataset_address=dataset_address, data_format='hdf5')
    validation_dataset = TimeSeriesDataset(dataset_address=validation_dataset_address, data_format='hdf5')

    filepath_for_saving_no_split = '/home/lpalmier/workspace/Data/SIR/SIR_dataset_timestep_2-5_06_no_split.hdf5'
    filepath_for_saving_w_split = '/home/lpalmier/workspace/Data/SIR/SIR_dataset_timestep_2-5_06_w_split.hdf5'
    dataset.format_dataset_for_ML(nb_past_timesteps=nb_past_timesteps, must_be_rescaled=True, positivity=None,
                                  percentage_of_test_data=0.0,
                                  filepath_for_saving_no_split=filepath_for_saving_no_split,
                                  filepath_for_saving_w_split=filepath_for_saving_w_split)

    filepath_for_saving_val_no_split = '/home/lpalmier/workspace/Data/SIR/SIR_dataset_timestep_2-5_02_no_split.hdf5'
    filepath_for_saving_val_w_split = '/home/lpalmier/workspace/Data/SIR/SIR_dataset_timestep_2-5_02_w_split.hdf5'
    validation_dataset.format_dataset_for_ML(nb_past_timesteps=nb_past_timesteps, must_be_rescaled=True, positivity=None,
                                  percentage_of_test_data=0.0,
                                  filepath_for_saving_no_split=filepath_for_saving_val_no_split,
                                  filepath_for_saving_w_split=filepath_for_saving_val_w_split)
    nb_features = dataset.nb_features

    training_filepath = filepath_for_saving_w_split
    validation_filepath = filepath_for_saving_val_w_split

    batch_size = 64
    X_train = HDF5Iterator(training_filepath, batch_size=batch_size,
                                      shuffle=True, X_label='X_train', y_label='y_train')
    X_test = HDF5Iterator(validation_filepath, batch_size=batch_size, shuffle=True, X_label='X_train', y_label='y_train')
    Y_train = None
    Y_test = None
    return X_train, X_test, Y_train, Y_test


def model(X_train, Y_train, X_test, Y_test):
    """
    Model providing function:
    Create Keras model with double curly brackets dropped-in as needed.
    Return value has to be a valid python dictionary with two customary keys:
        - loss: Specify a numeric evaluation metric to be minimized
        - status: Just use STATUS_OK and see hyperopt documentation if not feasible
    The last one is optional, though recommended, namely:
        - model: specify the model just created so that we can later use it again.
    """
    input_tensor = Input(shape=(1, 3))
    flatten1 = Flatten()(input_tensor)
    hidden1 = Dense({{choice([64, 128, 256, 512, 1024, 2048, 4096])}}, kernel_constraint=maxnorm({{uniform(1, 3)}}))(flatten1)
    dropout1 = Dropout({{uniform(0.2, 0.7)}})(hidden1)
    dense2 = Dense({{choice([64, 128, 256, 512, 1024])}}, kernel_constraint=maxnorm({{uniform(1, 3)}}))(dropout1)
    dropout2 = Dropout({{uniform(0.2, 0.7)}})(dense2)
    NN_body = Dense({{choice([64, 128, 256, 512, 1024, 2048])}}, kernel_constraint=maxnorm({{uniform(1, 3)}}))(dropout2)

    number_of_components = 2
    components = []
    for j in range(number_of_components):
        components.append(MultivariateNormalCholeskyOutputLayer(3))
    TopModel_obj = MixtureOutputLayer(components)

    NN = StochNeuralNetwork(input_tensor, NN_body, TopModel_obj)

    callbacks = []
    callbacks.append(EarlyStopping(monitor='val_loss', patience=3, verbose=1, mode='min'))
    result = NN.fit_generator(X_train, samples_per_epoch=10**5,
                    epochs={{choice([3, 6, 9, 12])}},
                    verbose=1,
                    callbacks=callbacks,
                    validation_generator=X_test)

    parameters = space
    val_loss = min(result.history['val_loss'])
    parameters["val_loss"] = val_loss
    print('Validation loss: {0}'.format(val_loss))

    if 'results' not in globals():
        global results
        results = []

    results.append(parameters)
    print(tabulate(results, headers="keys", tablefmt="fancy_grid", floatfmt=".8f"))
    with open('/home/lpalmier/workspace/model_tuning/SIR/SIR_model_tuning_MNC_01.json', 'w') as f:
        f.write(json.dumps(results))
    return {'loss': val_loss, 'status': STATUS_OK, 'model': NN.model}


if __name__ == '__main__':
    best_run, best_model = optim.minimize(model=model,
                                          data=data,
                                          algo=tpe.suggest,
                                          max_evals=20,
                                          trials=Trials())
    X_train, Y_train, X_test, Y_test = data()
