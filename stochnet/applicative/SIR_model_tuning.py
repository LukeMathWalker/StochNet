import os
import json
from tabulate import tabulate
from hyperopt import Trials, STATUS_OK, tpe
from hyperas import optim
from hyperas.distributions import choice, uniform
from keras.layers import Dense, Dropout, Input, LSTM
from keras.constraints import maxnorm
from keras.callbacks import EarlyStopping
from stochnet.classes.TimeSeriesDataset import TimeSeriesDataset
from stochnet.classes.NeuralNetworks import StochNeuralNetwork
from stochnet.classes.TopLayers import MultivariateLogNormalOutputLayer, MixtureOutputLayer


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
    hidden1 = LSTM({{choice([64, 128, 256, 512, 1024, 1536, 2048])}}, kernel_constraint=maxnorm({{uniform(1, 3)}}),
                   recurrent_constraint=maxnorm({{uniform(1, 3)}}))(input_tensor)
    dropout1 = Dropout({{uniform(0.2, 0.7)}})(hidden1)
    NN_body = Dense({{choice([64, 128, 256, 512, 1024, 1536, 2048])}}, kernel_constraint=maxnorm({{uniform(1, 3)}}))(dropout1)
    dropout2 = Dropout({{uniform(0.2, 0.7)}})(NN_body)
    NN_body = Dense({{choice([64, 128, 256, 512, 1024])}}, kernel_constraint=maxnorm({{uniform(1, 3)}}))(dropout2)

    number_of_components = 2
    components = []
    for j in range(number_of_components):
        components.append(MultivariateLogNormalOutputLayer(3))
    TopModel_obj = MixtureOutputLayer(components)

    NN = StochNeuralNetwork(input_tensor, NN_body, TopModel_obj)

    callbacks = [EarlyStopping(monitor='val_loss', patience=3, verbose=1, mode='min')]
    result = NN.fit(X_train, Y_train,
                    batch_size={{choice([512, 1024, 2048, 3072, 4096])}},
                    epochs={{choice([10, 15, 20, 40])}},
                    verbose=2,
                    callbacks=callbacks,
                    validation_data=(X_test, Y_test))
    # TODO: avoid this additional computation: can't we find the index where EarlyStopping kicks in?
    # val_loss = result.history['val_loss'][-1]
    val_loss = NN.evaluate(X_data=X_test, y_data=Y_test, batch_size=512)
    parameters = space
    parameters["val_loss"] = val_loss
    print('Validation loss: {0}'.format(val_loss))

    if 'results' not in globals():
        global results
        results = []

    results.append(parameters)
    print(tabulate(results, headers="keys", tablefmt="fancy_grid", floatfmt=".8f"))
    with open('/home/lpalmier/workspace/output/SIR_model_tuning_05.json', 'w') as f:
        f.write(json.dumps(results))
    return {'loss': val_loss, 'status': STATUS_OK, 'model': NN.model}


if __name__ == '__main__':
    best_run, best_model = optim.minimize(model=model,
                                          data=data,
                                          algo=tpe.suggest,
                                          max_evals=25,
                                          trials=Trials())
    X_train, Y_train, X_test, Y_test = data()
    print("Evalutation of best performing model:")
    print(best_model.evaluate(X_test, Y_test))
