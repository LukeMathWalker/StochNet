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
