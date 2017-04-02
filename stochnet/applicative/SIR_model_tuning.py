import os
import csv
from tabulate import tabulate
from hyperopt import Trials, STATUS_OK, tpe
from hyperas import optim
from hyperas.distributions import choice, uniform
from keras.layers import Dense, Dropout, Input, LSTM
from keras.constraints import maxnorm
from keras.callbacks import EarlyStopping
from stochnet.classes.TimeSeriesDataset import TimeSeriesDataset
from stochnet.classes.NeuralNetworks import StochNeuralNetwork
from stochnet.classes.TopLayers import MultivariateNormalCholeskyOutputLayer, MixtureOutputLayer


def data():
    '''
    Data providing function:

    This function is separated from model() so that hyperopt
    won't reload data for each evaluation run.
    '''
    current = os.getcwd()
    working_path = os.path.dirname(current)
    basename = os.path.abspath(working_path)
    dataset_address = os.path.join(basename, 'dataset/SIR_dataset.npy')
    dataset = TimeSeriesDataset(dataset_address)

    nb_past_timesteps = 5
    dataset.format_dataset_for_ML(nb_past_timesteps=nb_past_timesteps, percentage_of_test_data=0.25)

    X_train = dataset.X_train
    X_test = dataset.X_test
    Y_train = dataset.y_train
    Y_test = dataset.y_test
    return X_train, Y_train, X_test, Y_test


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
    input_tensor = Input(shape=(5, 3))
    hidden1 = LSTM({{choice([64, 128, 256, 512, 1024])}}, kernel_constraint=maxnorm({{uniform(1, 5)}}),
                   recurrent_constraint=maxnorm({{uniform(1, 5)}}))(input_tensor)
    dropout1 = Dropout({{uniform(0.1, 0.8)}})(hidden1)
    NN_body = Dense({{choice([64, 128, 256, 512, 1024])}})(dropout1)

    number_of_components = 3
    components = []
    for j in range(number_of_components):
        components.append(MultivariateNormalCholeskyOutputLayer(3))
    TopModel_obj = MixtureOutputLayer(components)

    NN = StochNeuralNetwork(input_tensor, NN_body, TopModel_obj)

    callbacks = [EarlyStopping(monitor='val_loss', patience=4, verbose=1, mode='min')]
    result = NN.fit(X_train, Y_train,
                    batch_size={{choice([64, 128])}},
                    epochs=2,
                    verbose=2,
                    callbacks=callbacks,
                    validation_data=(X_test, Y_test))
    val_loss = result.history['val_loss'][-1]
    parameters = space
    parameters["val_loss"] = val_loss

    if 'results' not in globals():
        global results
        results = []

    results.append(parameters)
    print(tabulate(results, headers="keys", tablefmt="fancy_grid", floatfmt=".8f"))
    with open('SIR_model_tuning.csv', 'w') as f:
        writer = csv.writer(f, delimiter=',')
        writer.writerows(results)

    loss = NN.evaluate(X_test, Y_test, verbose=0)
    print('Test loss: {0}'.format(loss))
    return {'loss': loss, 'status': STATUS_OK, 'model': NN.model}


if __name__ == '__main__':
    best_run, best_model = optim.minimize(model=model,
                                          data=data,
                                          algo=tpe.suggest,
                                          max_evals=3,
                                          trials=Trials())
    X_train, Y_train, X_test, Y_test = data()
    print("Evalutation of best performing model:")
    print(best_model.evaluate(X_test, Y_test))
