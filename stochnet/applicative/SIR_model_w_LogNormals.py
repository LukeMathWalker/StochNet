import os
from stochnet.classes.TimeSeriesDataset import TimeSeriesDataset
from stochnet.classes.NeuralNetworks import StochNeuralNetwork
from stochnet.classes.TopLayers import MultivariateNormalCholeskyOutputLayer, MultivariateLogNormalOutputLayer, MixtureOutputLayer
from utils.iterator import NumpyArrayIterator
from keras.layers import Input, LSTM, Dense, Dropout
from keras.callbacks import EarlyStopping
from keras.constraints import maxnorm

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
dataset_address = os.path.join(basename, 'dataset/SIR_dataset_big.npy')
test_dataset_address = os.path.join(basename, 'dataset/SIR_dataset_upgraded.npy')

data_labels = {'Timestamps': 0, 'Susceptible': 1, 'Infected': 2, 'Removed': 3}

dataset = TimeSeriesDataset(dataset_address, labels=data_labels)
test_dataset = TimeSeriesDataset(test_dataset_address, labels=data_labels)

nb_past_timesteps = 5
dataset.format_dataset_for_ML(nb_past_timesteps=nb_past_timesteps, must_be_rescaled=True, percentage_of_test_data=0)
test_dataset.format_dataset_for_ML(nb_past_timesteps=nb_past_timesteps, must_be_rescaled=True, percentage_of_test_data=0)

training_generator = NumpyArrayIterator(dataset.X_train, dataset.y_train, batch_size=512, shuffle=False)
validation_generator = NumpyArrayIterator(test_dataset.X_train, test_dataset.y_train, batch_size=512, shuffle=False)

input_tensor = Input(shape=(nb_past_timesteps, dataset.nb_features))
hidden1 = LSTM(128, kernel_constraint=maxnorm(1.78998725), recurrent_constraint=maxnorm(2.95163704))(input_tensor)
dropout1 = Dropout(0.46178651)(hidden1)
dense1 = Dense(512, kernel_constraint=maxnorm(1.57732507))(dropout1)
dropout2 = Dropout(0.62220663)(dense1)
NN_body = Dense(128, kernel_constraint=maxnorm(1.67525276))(dropout2)

number_of_components = 2
components = []
for j in range(number_of_components):
    components.append(MultivariateNormalCholeskyOutputLayer(dataset.nb_features))

TopModel_obj = MixtureOutputLayer(components)

NN = StochNeuralNetwork(input_tensor, NN_body, TopModel_obj)
callbacks = [EarlyStopping(monitor='val_loss', patience=4, verbose=1, mode='min')]
# result = NN.fit(dataset.X_train, dataset.y_train, batch_size=512, epochs=20, callbacks=callbacks, validation_data=(test_dataset.X_train, test_dataset.y_train))
result = NN.fit_generator(training_generator=training_generator,
                          samples_per_epoch=10**5, epochs=20, verbose=1,
                          callbacks=callbacks, validation_generator=validation_generator,
                          nb_val_samples=10**3)
lowest_val_loss = min(result.history['val_loss'])
print(lowest_val_loss)

test_set_prediction = NN.predict(test_dataset.X_train)
NN.visualize_performance_by_sampling(test_dataset.X_train, test_dataset.y_train, test_set_prediction,
                                     max_display=4, fitted_scaler=dataset.scaler,
                                     feature_labels=dataset.labels)

filepath_for_saving = os.path.join(basename, 'models/SIR_' + str(lowest_val_loss) + '.h5')
NN.save(filepath_for_saving)
