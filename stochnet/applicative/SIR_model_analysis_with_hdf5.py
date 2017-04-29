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

# We need to get to the proper directory first
# We are in ./stochnet/applicative
# We want to be in ./stochnet
# os.getcwd returns the directory we are working in and we use dirname to get its parent directory
current = os.getcwd()
working_path = os.path.dirname(current)
basename = os.path.abspath(working_path)

dataset_address = '/home/lucap/Documenti/Data storage/SIR/SIR_dataset_timestep_2-5_07.hdf5'
validation_dataset_address = '/home/lucap/Documenti/Data storage/SIR/SIR_dataset_timestep_2-5_06.hdf5'

nb_past_timesteps = 1

formatted_for_ML = False
to_be_scaled = True

if formatted_for_ML is True:
    nb_features = 3
    scaler_address = '/home/lucap/Documenti/Data storage/SIR/scaler_for_timestep_2-5_dataset_v_big_01.h5'
    with open(scaler_address, 'rb') as f:
        scaler = dill.load(f)
    if to_be_scaled is True:
        rescaled_dataset_address = '/home/lucap/Documenti/Data storage/SIR/timestep_2-5_dataset_v_big_01_w_split.hdf5'
        rescaled_validation_dataset_address = '/home/lucap/Documenti/Data storage/SIR/timestep_2-5_dataset_v_big_01_w_split.hdf5'
        rescale_hdf5(dataset_address, rescaled_dataset_address, scaler)
        # rescale_hdf5(validation_dataset_address, rescaled_validation_dataset_address, scaler)
        training_filepath = rescaled_dataset_address
        validation_filepath = rescaled_validation_dataset_address
    else:
        training_filepath = dataset_address
        validation_filepath = validation_dataset_address

else:
    dataset = TimeSeriesDataset(dataset_address=dataset_address, data_format='hdf5')
    validation_dataset = TimeSeriesDataset(dataset_address=validation_dataset_address, data_format='hdf5')

    filepath_for_saving_no_split = '/home/lucap/Documenti/Data storage/SIR/SIR_dataset_timestep_2-5_07_no_split.hdf5'
    filepath_for_saving_w_split = '/home/lucap/Documenti/Data storage/SIR/SIR_dataset_timestep_2-5_07_w_split.hdf5'
    dataset.format_dataset_for_ML(nb_past_timesteps=nb_past_timesteps, must_be_rescaled=True, positivity='needed',
                                  percentage_of_test_data=0.0,
                                  filepath_for_saving_no_split=filepath_for_saving_no_split,
                                  filepath_for_saving_w_split=filepath_for_saving_w_split)

    filepath_for_saving_val_no_split = '/home/lucap/Documenti/Data storage/SIR/SIR_dataset_timestep_2-5_06_no_split.hdf5'
    filepath_for_saving_val_w_split = '/home/lucap/Documenti/Data storage/SIR/SIR_dataset_timestep_2-5_06_w_split.hdf5'
    validation_dataset.format_dataset_for_ML(nb_past_timesteps=nb_past_timesteps, must_be_rescaled=True, positivity='needed',
                                  percentage_of_test_data=0.0,
                                  filepath_for_saving_no_split=filepath_for_saving_val_no_split,
                                  filepath_for_saving_w_split=filepath_for_saving_val_w_split)

    nb_features = dataset.nb_features

    training_filepath = filepath_for_saving_w_split
    validation_filepath = filepath_for_saving_val_w_split
    scaler = dataset.scaler

batch_size = 64
training_generator = HDF5Iterator(training_filepath, batch_size=batch_size,
                                  shuffle=True, X_label='X_train', y_label='y_train')
validation_generator = HDF5Iterator(validation_filepath, batch_size=batch_size,
                                    shuffle=True, X_label='X_train', y_label='y_train')

input_tensor = Input(shape=(nb_past_timesteps, nb_features))
flatten1 = Flatten()(input_tensor)
dense1 = Dense(2048, kernel_constraint=maxnorm(3), activation='relu')(flatten1)
dropout1 = Dropout(0.3)(dense1)
dense2 = Dense(1024, kernel_constraint=maxnorm(3), activation='relu')(dropout1)
dropout2 = Dropout(0.7)(dense2)
NN_body = Dense(2048, kernel_constraint=maxnorm(3), activation='relu')(dropout2)

number_of_components = 2
components = []
components.append(MultivariateNormalDiagOutputLayer(nb_features, diag_regularizer=None))
components.append(MultivariateNormalDiagOutputLayer(nb_features, diag_regularizer=None))

TopModel_obj = MixtureOutputLayer(components)

# TopModel_obj = MultivariateNormalDiagOutputLayer(nb_features)

NN = StochNeuralNetwork(input_tensor, NN_body, TopModel_obj)
NN.memorize_scaler(scaler)

model_directory = os.path.join(basename, 'models/model_14')
ensure_dir(model_directory)
print(model_directory)

callbacks = []
callbacks.append(EarlyStopping(monitor='val_loss', patience=6, verbose=1, mode='min'))
checkpoint_filepath = os.path.join(model_directory, 'best_weights.h5')
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
model_filepath = os.path.join(model_directory, 'model.h5')
NN.save_model(model_filepath)

filepath = os.path.join(model_directory, 'SIR_' + str(lowest_val_loss) + '.h5')
NN.save(filepath)

test_batch_x, test_batch_y = next(validation_generator)
test_batch_prediction = NN.predict_on_batch(test_batch_x)
NN.visualize_performance_by_sampling(test_batch_x, test_batch_y, test_batch_prediction,
                                     max_display=6)
