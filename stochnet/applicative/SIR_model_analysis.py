import os
from stochnet.classes.TimeSeriesDataset import TimeSeriesDataset
from stochnet.classes.NeuralNetworks import StochNeuralNetwork
from stochnet.classes.TopLayers import MultivariateNormalCholeskyOutputLayer, MixtureOutputLayer
from keras.layers import Input, LSTM, Dense

# We need to get to the proper directory first
# We are in ./stochnet/applicative
# We want to be in ./stochnet
# os.getcwd returns the directory we are working in and we use dirname to get the parent directory
current = os.getcwd()
working_path = os.path.dirname(current)
basename = os.path.abspath(working_path)

dataset_address = os.path.join(basename, 'dataset/dataset_00.npy')
dataset = TimeSeriesDataset(dataset_address)
dataset.format_dataset_for_ML(nb_past_timesteps=3)
dataset.train_test_split(percentage_of_test_data=0.25)

input_tensor = Input(shape=(dataset.nb_timesteps, dataset.nb_features))
hidden1 = LSTM(150)(input_tensor)
NN_body = Dense(75)(hidden1)

number_of_components = 3
components = []
for j in range(number_of_components):
    components.append(MultivariateNormalCholeskyOutputLayer(dataset.nb_features))
TopModel_obj = MixtureOutputLayer(components)

NN = StochNeuralNetwork(NN_body, TopModel_obj)