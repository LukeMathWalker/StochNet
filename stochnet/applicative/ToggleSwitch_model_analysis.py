import os
from stochnet.classes.TimeSeriesDataset import TimeSeriesDataset
from stochnet.classes.NeuralNetworks import StochNeuralNetwork
from stochnet.classes.TopLayers import MultivariateNormalCholeskyOutputLayer, MixtureOutputLayer
from keras.layers import Input, LSTM, Dense, Dropout
from keras.callbacks import EarlyStopping


def generate_batches_from_array(X_data, y_data):
    while 1:
        nb_of_data = X_data.shape[0]
        for j in range(nb_of_data - 256):
            yield (X_data[j:j + 256], y_data[j:j + 256])


# We need to get to the proper directory first
# We are in ./stochnet/applicative
# We want to be in ./stochnet
# os.getcwd returns the directory we are working in and we use dirname to get its parent directory
current = os.getcwd()
working_path = os.path.dirname(current)
basename = os.path.abspath(working_path)
dataset_address = os.path.join(basename, 'dataset/ToggleSwitch_dataset.npy')

dataset = TimeSeriesDataset(dataset_address)

nb_past_timesteps = 10
dataset.format_dataset_for_ML(nb_past_timesteps=nb_past_timesteps, percentage_of_test_data=0.25)

input_tensor = Input(shape=(nb_past_timesteps, dataset.nb_features))
hidden1 = LSTM(150)(input_tensor)
dropout1 = Dropout(0.3)(hidden1)
NN_body = Dense(75)(dropout1)

number_of_components = 3
components = []
for j in range(number_of_components):
    components.append(MultivariateNormalCholeskyOutputLayer(dataset.nb_features))
TopModel_obj = MixtureOutputLayer(components)

print(dataset.X_data.shape)

NN = StochNeuralNetwork(input_tensor, NN_body, TopModel_obj)
callbacks = [EarlyStopping(monitor='val_loss', patience=2, verbose=1, mode='min')]
NN.fit(dataset.X_train, dataset.y_train, batch_size=1024, epochs=50, validation_split=0.2, callbacks=callbacks)


test_set_prediction = NN.predict(dataset.X_test)
NN.visualize_performance_by_sampling(dataset.X_test, dataset.y_test, test_set_prediction, max_display=2)
