import os
import dill
import h5py
from tqdm import tqdm
from stochnet.classes.TimeSeriesDataset import TimeSeriesDataset
from stochnet.classes.NeuralNetworks import StochNeuralNetwork
from stochnet.classes.TopLayers import MultivariateNormalCholeskyOutputLayer, MultivariateLogNormalOutputLayer, MixtureOutputLayer, MultivariateNormalDiagOutputLayer
from stochnet.utils.iterator import HDF5Iterator
from keras.layers import Input, LSTM, Dense, Dropout, Flatten
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.constraints import maxnorm
from keras.regularizers import l2
import tensorflow as tf


def ensure_dir(file_path):
    if not os.path.exists(file_path):
        os.makedirs(file_path)

def prepare_dataset(dataset_folder, dataset_name, dataset_extension, nb_past_timesteps, positivity=None, scaler=None):
    dataset_address = os.path.join(dataset_folder,
                                   dataset_name + dataset_extension)
    dataset = TimeSeriesDataset(dataset_address=dataset_address,
                                data_format=dataset_extension[1:],
                                with_timestamps=False)
    dataset_no_split_filename = (dataset_name +
                                 '_no_split' +
                                 dataset_extension)
    dataset_w_split_filename = (dataset_name +
                                '_w_split' +
                                dataset_extension)
    path_no_split = os.path.join(dataset_folder, dataset_no_split_filename)
    path_w_split = os.path.join(dataset_folder, dataset_w_split_filename)
    if scaler is not None:
        dataset.set_scaler(scaler)
    dataset.format_dataset_for_ML(nb_past_timesteps=nb_past_timesteps,
                                  must_be_rescaled=True,
                                  positivity=positivity,
                                  percentage_of_test_data=0.0,
                                  filepath_for_saving_no_split=path_no_split,
                                  filepath_for_saving_w_split=path_w_split)
    return dataset


def rescale_hdf5(filepath_raw, filepath_rescaled, scaler, X_label='X_train', y_label='y_train'):
    chunk_size = 10**5
    f_raw = h5py.File(filepath_raw, 'r')
    X_data = f_raw[X_label]
    y_data = f_raw[y_label]
    f_rescaled = h5py.File(filepath_rescaled, 'a', libver='latest')
    X_data_rescaled = f_rescaled.create_dataset(X_label, shape=X_data.shape, chunks=True)
    y_data_rescaled = f_rescaled.create_dataset(y_label, shape=y_data.shape, chunks=True)

    nb_iteration = X_data.shape[0] // chunk_size
    for i in tqdm(range(nb_iteration)):
        X_data_slice = X_data[i * chunk_size: (i + 1) * chunk_size, ...]
        X_flat_data_slice = X_data_slice.reshape(-1, X_data.shape[-1])
        X_data_rescaled[i * chunk_size: (i + 1) * chunk_size, ...] = scaler.transform(X_flat_data_slice).reshape(X_data_slice.shape)

        y_data_slice = y_data[i * chunk_size: (i + 1) * chunk_size, ...]
        y_flat_data_slice = y_data_slice.reshape(-1, y_data.shape[-1])
        y_data_rescaled[i * chunk_size: (i + 1) * chunk_size, ...] = scaler.transform(y_flat_data_slice).reshape(y_data_slice.shape)
    if nb_iteration * chunk_size != X_data.shape[0]:
        X_data_slice = X_data[nb_iteration * chunk_size:]
        X_flat_data_slice = X_data_slice.reshape(-1, X_data.shape[-1])
        X_data_rescaled[nb_iteration * chunk_size:, ...] = scaler.transform(X_flat_data_slice).reshape(X_data_slice.shape)

        y_data_slice = y_data[nb_iteration * chunk_size:, ...]
        y_flat_data_slice = y_data_slice.reshape(-1, y_data.shape[-1])
        y_data_rescaled[nb_iteration * chunk_size:, ...] = scaler.transform(y_flat_data_slice).reshape(y_data_slice.shape)
    return


# sess = tf.Session()
# sess = tf_debug.LocalCLIDebugWrapperSession(sess)
# K.set_session(sess)

current = os.getcwd()
working_path = os.path.dirname(current)
basename = os.path.abspath(working_path)
dataset_folder = '/home/lucap/Documenti/Data storage/SIR'
training_dataset_name = 'SIR_dataset_timestep_2-1_dataset_01+02+03+04_no_timestamp'
validation_dataset_name = 'SIR_dataset_timestep_2-1_05_no_timestamp'
training_dataset_extension = '.hdf5'
validation_dataset_extension = '.hdf5'

training_dataset_address = os.path.join(dataset_folder,
                                        training_dataset_name + training_dataset_extension)
validation_dataset_address = os.path.join(dataset_folder,
                                          validation_dataset_name + validation_dataset_extension)

nb_past_timesteps = 1

training_dataset = prepare_dataset(dataset_folder,
                                   training_dataset_name,
                                   training_dataset_extension,
                                   nb_past_timesteps,
                                   positivity='needed',
                                   scaler=None)
training_filepath = training_dataset.path_w_split

scaler = training_dataset.scaler
scaler_address = os.path.join(dataset_folder,
                              'scaler_for_' + training_dataset_name + '.h5')
with open(scaler_address, 'wb') as f:
    dill.dump(scaler, f)

validation_dataset = prepare_dataset(dataset_folder,
                                     validation_dataset_name,
                                     validation_dataset_extension,
                                     nb_past_timesteps,
                                     positivity='needed',
                                     scaler=scaler)
validation_filepath = validation_dataset.path_w_split

nb_features = training_dataset.nb_features


batch_size = 64
training_generator = HDF5Iterator(training_filepath, batch_size=batch_size,
                                  shuffle=True, X_label='X_train',
                                  y_label='y_train')
validation_generator = HDF5Iterator(validation_filepath, batch_size=batch_size,
                                    shuffle=True, X_label='X_train',
                                    y_label='y_train')

input_tensor = Input(shape=(nb_past_timesteps, nb_features))
flatten1 = Flatten()(input_tensor)
dense1 = Dense(1700, kernel_constraint=maxnorm(3), activation='relu')(flatten1)
dropout1 = Dropout(0.3)(dense1)
dense2 = Dense(1000, kernel_constraint=maxnorm(3), activation='relu')(dropout1)
dropout2 = Dropout(0.4)(dense2)
NN_body = Dense(1600, kernel_constraint=maxnorm(3), activation='relu')(dropout2)

number_of_components = 2
components = []
components.append(MultivariateNormalDiagOutputLayer(nb_features))
components.append(MultivariateNormalDiagOutputLayer(nb_features))

TopModel_obj = MixtureOutputLayer(components)

# TopModel_obj = MultivariateNormalDiagOutputLayer(nb_features)

NN = StochNeuralNetwork(input_tensor, NN_body, TopModel_obj)
NN.memorize_scaler(scaler)

model_directory = os.path.join(basename, 'models/SIR_timestep_2-1/model_03')
ensure_dir(model_directory)

callbacks = []
callbacks.append(EarlyStopping(monitor='val_loss',
                               patience=5,
                               verbose=1,
                               mode='min'))

checkpoint_filepath = os.path.join(model_directory, 'best_weights.h5')
callbacks.append(ModelCheckpoint(checkpoint_filepath, monitor='val_loss',
                                 verbose=1, save_best_only=True,
                                 save_weights_only=True, mode='min'))
result = NN.fit_generator(training_generator=training_generator,
                          samples_per_epoch=3 * 10**5, epochs=80, verbose=1,
                          callbacks=callbacks,
                          validation_generator=validation_generator,
                          nb_val_samples=10**4)
lowest_val_loss = min(result.history['val_loss'])
print(lowest_val_loss)

NN.load_weights(checkpoint_filepath)
model_filepath = os.path.join(model_directory, 'model.h5')
NN.save_model(model_filepath)

filepath = os.path.join(model_directory, 'SIR_' + str(lowest_val_loss) + '.h5')
NN.save(filepath)

test_batch_x, test_batch_y = next(validation_generator)
test_batch_prediction = NN.predict_on_batch(test_batch_x)
NN.visualize_performance_by_sampling(test_batch_x, test_batch_y,
                                     test_batch_prediction,
                                     max_display=6)
