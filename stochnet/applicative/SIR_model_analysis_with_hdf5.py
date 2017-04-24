import os
from stochnet.classes.TimeSeriesDataset import TimeSeriesDataset
from stochnet.classes.NeuralNetworks import StochNeuralNetwork
from stochnet.classes.TopLayers import MultivariateNormalCholeskyOutputLayer, MultivariateLogNormalOutputLayer, MixtureOutputLayer, MultivariateNormalDiagOutputLayer
from stochnet.utils.iterator import HDF5Iterator
from keras.layers import Input, LSTM, Dense, Dropout, Flatten
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.constraints import maxnorm
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

dataset_address = '/home/lucap/Documenti/Data storage/SIR/timestep_2-5_dataset_big_03.hdf5'
validation_dataset_address = '/home/lucap/Documenti/Data storage/SIR/timestep_2-5_dataset_big_04.hdf5'

nb_past_timesteps = 1

formatted_for_ML = False

if formatted_for_ML is True:
    training_filepath = dataset_address
    validation_filepath = validation_dataset_address
    nb_features = 3
else:
    dataset = TimeSeriesDataset(dataset_address=dataset_address, data_format='hdf5')
    validation_dataset = TimeSeriesDataset(dataset_address=validation_dataset_address, data_format='hdf5')

    filepath_for_saving_no_split = '/home/lucap/Documenti/Data storage/SIR/timestep_2-5_dataset_big_03_no_split.hdf5'
    filepath_for_saving_w_split = '/home/lucap/Documenti/Data storage/SIR/timestep_2-5_dataset_big_03_w_split.hdf5'
    dataset.format_dataset_for_ML(nb_past_timesteps=nb_past_timesteps, must_be_rescaled=True,
                                  percentage_of_test_data=0.0,
                                  filepath_for_saving_no_split=filepath_for_saving_no_split,
                                  filepath_for_saving_w_split=filepath_for_saving_w_split)

    filepath_for_saving_val_no_split = '/home/lucap/Documenti/Data storage/SIR/timestep_2-5_dataset_big_04_no_split.hdf5'
    filepath_for_saving_val_w_split = '/home/lucap/Documenti/Data storage/SIR/timestep_2-5_dataset_big_04_w_split.hdf5'
    validation_dataset.format_dataset_for_ML(nb_past_timesteps=nb_past_timesteps, must_be_rescaled=True,
                                  percentage_of_test_data=0.0,
                                  filepath_for_saving_no_split=filepath_for_saving_val_no_split,
                                  filepath_for_saving_w_split=filepath_for_saving_val_w_split)

    nb_features = dataset.nb_features

    training_filepath = filepath_for_saving_w_split
    validation_filepath = filepath_for_saving_val_w_split

batch_size = 64
training_generator = HDF5Iterator(training_filepath, batch_size=batch_size,
                                  shuffle=True, X_label='X_train', y_label='y_train')
validation_generator = HDF5Iterator(validation_filepath, batch_size=batch_size,
                                    shuffle=True, X_label='X_train', y_label='y_train')

input_tensor = Input(shape=(nb_past_timesteps, nb_features))
flatten1 = Flatten()(input_tensor)
dense1 = Dense(2048, kernel_constraint=maxnorm(3), activation='relu')(flatten1)
dense2 = Dense(1024, kernel_constraint=maxnorm(3), activation='relu')(dense1)
NN_body = Dense(1024, kernel_constraint=maxnorm(3), activation='relu')(dense2)

number_of_components = 2
components = []
for j in range(number_of_components):
    components.append(MultivariateNormalDiagOutputLayer(nb_features))

TopModel_obj = MixtureOutputLayer(components)

# TopModel_obj = MultivariateNormalDiagOutputLayer(nb_features)

NN = StochNeuralNetwork(input_tensor, NN_body, TopModel_obj)
NN.memorize_scaler(dataset.scaler)

callbacks = []
callbacks.append(EarlyStopping(monitor='val_loss', patience=6, verbose=1, mode='min'))
checkpoint_filepath = os.path.join(basename, 'models/model_08/best_weights.h5')
callbacks.append(ModelCheckpoint(checkpoint_filepath, monitor='val_loss',
                                 verbose=1, save_best_only=True,
                                 save_weights_only=True, mode='min'))
result = NN.fit_generator(training_generator=training_generator,
                          samples_per_epoch=10**5, epochs=80, verbose=1,
                          callbacks=callbacks, validation_generator=validation_generator,
                          nb_val_samples=10**3)
lowest_val_loss = min(result.history['val_loss'])
print(lowest_val_loss)

NN.load_weights(checkpoint_filepath)
model_filepath = os.path.join(basename, 'models/model_08/model.h5')
NN.save_model(model_filepath)

filepath = os.path.join(basename, 'models/model_08/dill_SIR_' + str(lowest_val_loss) + '.h5')
NN.save(filepath)

test_batch_x, test_batch_y = next(validation_generator)
test_batch_prediction = NN.predict_on_batch(test_batch_x)
NN.visualize_performance_by_sampling(test_batch_x, test_batch_y, test_batch_prediction,
                                     max_display=6)
