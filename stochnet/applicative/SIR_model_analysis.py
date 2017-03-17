import os
from stochnet.classes.TimeSeriesDataset import TimeSeriesDataset
from stochnet.classes.NeuralNetworks import StochNeuralNetwork
from stochnet.classes.TopLayers import MultivariateNormalCholeskyOutputLayer, MixtureOutputLayer
from keras.layers import Input, LSTM, Dense
from keras.callbacks import EarlyStopping

# We need to get to the proper directory first
# We are in ./stochnet/applicative
# We want to be in ./stochnet
# os.getcwd returns the directory we are working in and we use dirname to get its parent directory
current = os.getcwd()
working_path = os.path.dirname(current)
basename = os.path.abspath(working_path)
dataset_address = os.path.join(basename, 'dataset/SIR_dataset.npy')

dataset = TimeSeriesDataset(dataset_address)

nb_past_timesteps = 10
dataset.format_dataset_for_ML(nb_past_timesteps=nb_past_timesteps, percentage_of_test_data=0.25)

input_tensor = Input(shape=(nb_past_timesteps, dataset.nb_features))
hidden1 = LSTM(150, return_sequences=True)(input_tensor)
hidden2 = LSTM(150)(hidden1)
NN_body = Dense(75)(hidden2)

number_of_components = 2
components = []
for j in range(number_of_components):
    components.append(MultivariateNormalCholeskyOutputLayer(dataset.nb_features))
TopModel_obj = MixtureOutputLayer(components)

NN = StochNeuralNetwork(input_tensor, NN_body, TopModel_obj)
callbacks = [EarlyStopping(monitor='val_loss', patience=4, verbose=1, mode='min')]
NN.fit(dataset.X_train, dataset.y_train, nb_epoch=50, validation_split=0.2, callbacks=callbacks)

test_set_prediction = NN.predict(dataset.X_test)
NN.visualize_performance_by_sampling(dataset.X_test, dataset.y_test, test_set_prediction, max_display=2)
