import sys
import os
import tensorflow as tf

import numpy as np
import matplotlib.pyplot as plt

from keras.utils.generic_utils import get_custom_objects

from stochnet.classes.NeuralNetworks import StochNeuralNetwork
from stochnet.utils.histograms import histogram_distance, get_histogram
from stochnet.utils.file_organization import ProjectFileExplorer
from stochnet.utils.utils import scale_back, rescale


def sample_from_distribution(NN, NN_prediction, nb_samples, sess=None):
    if sess is None:
        sess = tf.Session()
    samples = NN.TopLayer_obj.sample(NN_prediction, nb_samples, sess)
    return samples


def load_NN(model_explorer):
    NN = StochNeuralNetwork.load(model_explorer.StochNet_fp)
    get_custom_objects().update({"exp": lambda x: tf.exp(x),
                                 "loss_function": NN.TopLayer_obj.loss_function})
    NN.load_model(model_explorer.keras_fp)
    return NN


def evaluate_model_on_dataset(dataset_explorer, NN, sess, model_id, plot=False,
                              log_results=False):
    SSA_traj = get_SSA_traj(dataset_explorer)
    S_SSA_hist = get_SSA_hist(SSA_traj)
    nb_traj = SSA_traj.shape[1]
    return compute_histogram_distance(dataset_explorer, NN, S_SSA_hist, nb_traj,
                                      sess, model_id, plot=plot,
                                      log_results=log_results)


def compute_histogram_distance(dataset_explorer, NN, S_SSA_hist, nb_traj, sess,
                               model_id, plot=False, log_results=False):
    settings_rescaled = get_hist_settings_rescaled(dataset_explorer, NN.scaler)
    nb_settings = settings_rescaled.shape[0]

    histogram_explorer = dataset_explorer.get_HistogramFileExplorer(model_id)

    S_histogram_distance = np.zeros(nb_settings)
    for i in range(nb_settings):
        S_NN_hist = get_S_hist_from_NN(NN, settings_rescaled[i], nb_traj, sess)
        S_histogram_distance[i] = histogram_distance(S_NN_hist, S_SSA_hist[i], 1)

        if plot is True:
            make_and_save_plot(i, S_NN_hist, S_SSA_hist[i],
                               histogram_explorer.histogram_folder)

    S_mean_histogram_distance = np.mean(S_histogram_distance)
    if log_results is True:
        with open(histogram_explorer.log_fp, 'w') as f:
            f.write('The mean histogram distance, computed on {0} settings, is:'.format(nb_settings))
            f.write('{0}'.format(str(S_mean_histogram_distance)))

    return S_mean_histogram_distance


def get_hist_settings_rescaled(dataset_explorer, scaler):
    with open(dataset_explorer.histogram_settings_fp, 'rb') as f:
        settings = np.load(f)
    settings_rescaled = rescale(settings, scaler)
    return settings_rescaled


def get_SSA_traj(dataset_explorer):
    with open(dataset_explorer.histogram_dataset_fp, 'rb') as f:
        SSA_traj = np.load(f)
    return SSA_traj


def get_SSA_hist(SSA_traj):
    nb_settings = SSA_traj.shape[0]
    S_SSA_hist = []
    for i in range(nb_settings):
        S_samples_SSA = SSA_traj[i, :, -1, 1]
        S_SSA_hist.append(get_histogram(S_samples_SSA, -0.5, 200.5, 201))
    return S_SSA_hist


def get_S_hist_from_NN(NN, setting, nb_traj, sess):
    temp = setting.reshape(1, 1, -1)
    print(temp.shape)
    NN.model.summary()
    NN_prediction = NN.predict(temp, batch_size=1)
    NN_samples_rescaled = sample_from_distribution(NN, NN_prediction, nb_traj, sess)
    NN_samples = scale_back(NN_samples_rescaled, NN.scaler)
    S_samples_NN = NN_samples[:, 0, 0]
    S_NN_hist = get_histogram(S_samples_NN, -0.5, 200.5, 201)
    return S_NN_hist


def make_and_save_plot(figure_index, NN_hist, SSA_hist, folder):
    fig = plt.figure(figure_index)
    plt.plot(NN_hist, label='NN')
    plt.plot(SSA_hist, label='SSA')
    plt.legend()
    plot_filepath = os.path.join(folder, str(figure_index) + '.png')
    plt.savefig(plot_filepath, bbox_inches='tight')
    plt.close()
    return


if __name__ == '__main__':
    np.set_printoptions(suppress=True)
    sess = tf.Session()

    timestep = float(sys.argv[1])
    nb_past_timesteps = int(sys.argv[2])
    training_dataset_id = int(sys.argv[3])
    validation_dataset_id = int(sys.argv[4])
    model_id = int(sys.argv[5])
    project_folder = str(sys.argv[6])

    project_explorer = ProjectFileExplorer(project_folder)
    train_explorer = project_explorer.get_DatasetFileExplorer(timestep, training_dataset_id)
    val_explorer = project_explorer.get_DatasetFileExplorer(timestep, validation_dataset_id)
    model_explorer = project_explorer.get_ModelFileExplorer(timestep, model_id)
    NN = load_NN(model_explorer)

    mean_train_hist_dist = evaluate_model_on_dataset(train_explorer, NN, sess,
                                                     model_id, plot=True,
                                                     log_results=True)
    mean_val_hist_dist = evaluate_model_on_dataset(val_explorer, NN, sess,
                                                   model_id, plot=True,
                                                   log_results=True)
