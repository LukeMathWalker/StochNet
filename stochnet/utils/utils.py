from stochnet.utils.iterator import NumpyArrayIterator
import numpy as np
import dill


def change_scaling(data, old_scaler, new_scaler):
    data_shape = data.shape
    flat_data = data.reshape((-1, data_shape[-1]))
    flat_data = new_scaler.transform(old_scaler.inverse_transform(flat_data))
    data = flat_data.reshape(data_shape)
    return data


def rescale(v, scaler):
    v_shape = v.shape
    flat_v = v.reshape(-1, v_shape[-1])
    flat_v_rescaled = scaler.transform(flat_v)
    v_rescaled = flat_v_rescaled.reshape(v_shape)
    return v_rescaled


def scale_back(v, scaler):
    v_shape = v.shape
    flat_v = v.reshape(-1, v_shape[-1])
    flat_v_rescaled = scaler.inverse_transform(flat_v)
    v_rescaled = flat_v_rescaled.reshape(v_shape)
    return v_rescaled

def get_train_and_validation_generator_w_scaler(train_explorer, val_explorer, batch_size=64):
    rescaled_x_train, rescaled_y_train, scaler_train = get_rescaled_dataset(train_explorer)
    rescaled_x_val, rescaled_y_val, scaler_val = get_rescaled_dataset(val_explorer)

    rescaled_x_val = change_scaling(rescaled_x_val, scaler_val, scaler_train)
    rescaled_y_val = change_scaling(rescaled_y_val, scaler_val, scaler_train)

    training_generator = NumpyArrayIterator(rescaled_x_train, rescaled_y_train,
                                            batch_size=batch_size,
                                            shuffle=True)
    validation_generator = NumpyArrayIterator(rescaled_x_val, rescaled_y_val,
                                              batch_size=batch_size,
                                              shuffle=True)
    return training_generator, validation_generator, scaler_train


def get_rescaled_dataset(dataset_explorer):
    with open(dataset_explorer.rescaled_x_fp, 'rb') as f:
        rescaled_x = np.load(f)
    with open(dataset_explorer.rescaled_y_fp, 'rb') as f:
        rescaled_y = np.load(f)
    with open(dataset_explorer.scaler_fp, 'rb') as f:
        scaler = dill.load(f)
    return rescaled_x, rescaled_y, scaler
