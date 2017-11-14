import sys
import os
import tensorflow as tf

from importlib import import_module
import numpy as np
import matplotlib.pyplot as plt

import keras.backend as K
from keras.utils import CustomObjectScope

from stochnet.classes.NeuralNetworks import StochNeuralNetwork
from stochnet.utils.histograms import histogram_distance, get_histogram
from stochnet.utils.file_organization import ProjectFileExplorer
from stochnet.utils.utils import scale_back, rescale


def load_NN(model_explorer):
    NN = StochNeuralNetwork.load(model_explorer.StochNet_fp)
    with CustomObjectScope({"exp": lambda x: tf.exp(x), "loss_function": NN.TopLayer_obj.loss_function}):
        NN.load_model(model_explorer.keras_fp)
        NN.load_weights(model_explorer.weights_fp)
    return NN


def evaluate_model_on_dataset(dataset_explorer, nb_past_timesteps, NN, sess,
                              model_id, CRN_class, nb_steps, n_bins=200, plot=False,
                              log_results=False):
    hist_bounds = CRN_class.get_histogram_bounds()
    bin_lengths = [(b[1] - b[0]) / n_bins for b in hist_bounds]
    hist_species, hist_species_indexes = get_histogram_species_w_indexes(CRN_class)

    rescaled_settings = get_hist_settings_rescaled(dataset_explorer, NN.scaler)
    SSA_traj = get_SSA_traj(dataset_explorer)
    nb_traj = SSA_traj.shape[1]
    nb_settings = SSA_traj.shape[0]

    for nb_steps in steps:
        print('Number of future steps: {0}.'.format(nb_steps))
        for setting_id in range(nb_settings):
            SSA_hist_samples = SSA_traj[setting_id, :, nb_steps, hist_species_indexes].T
            NN_hist_samples = get_NN_hist_samples(NN, rescaled_settings[setting_id],
                                                  hist_species_indexes, nb_steps,
                                                  nb_past_timesteps, nb_traj, sess)

            hist_explorer = dataset_explorer.get_HistogramFileExplorer(model_id, nb_steps)
            hist_distance = compute_histogram_distance(hist_explorer, SSA_hist_samples, NN_hist_samples,
                                                       hist_bounds, bin_lengths, n_bins, hist_species,
                                                       setting_id, plot=plot)
            if setting_id == 0:
                mean_hist_distance = hist_distance
            else:
                tmp = np.array([mean_hist_distance, hist_distance])
                mean_hist_distance = np.average(tmp, weights=(setting_id, 1), axis=0)
        if log_results is True:
            _log_results(hist_explorer, nb_settings, mean_hist_distance, hist_species)
    return


def get_histogram_species_w_indexes(CRN_class):
    species = CRN_class.get_species()
    histogram_species = CRN_class.get_species_for_histogram()
    histogram_species_indexes = [species.index(s) for s in histogram_species]
    return (histogram_species, histogram_species_indexes)


def get_SSA_traj(dataset_explorer):
    # SSA_traj = [nb_settings, nb_traj, nb_past_timesteps+1, nb_species+1]
    with open(dataset_explorer.histogram_dataset_fp, 'rb') as f:
        SSA_traj = np.load(f)
    # Remove timestamps
    SSA_traj = SSA_traj[..., 1:]
    return SSA_traj


def get_hist_settings_rescaled(dataset_explorer, scaler):
    with open(dataset_explorer.histogram_settings_fp, 'rb') as f:
        settings = np.load(f)
    settings_rescaled = rescale(settings, scaler)
    return settings_rescaled


def get_NN_hist_samples(NN, rescaled_setting, hist_species_indexes, nb_steps,
                        nb_past_timesteps, nb_traj, sess):
    setting = rescaled_setting.reshape(1, nb_past_timesteps, -1)
    nb_species = setting.shape[-1]
    NN_prediction = NN.predict_on_batch(setting)
    rescaled_NN_samples = sample_from_distribution(NN, NN_prediction,
                                                   nb_traj, sess)
    for i in range(nb_steps - 1):
        NN_prediction = NN.predict(rescaled_NN_samples, batch_size=5096)
        rescaled_NN_samples = sample_from_distribution(NN, NN_prediction, 1, sess)
        rescaled_NN_samples = rescaled_NN_samples.reshape(nb_traj, nb_past_timesteps, nb_species)
    NN_samples = scale_back(rescaled_NN_samples, NN.scaler)
    NN_samples = NN_samples[:, 0, hist_species_indexes]
    return NN_samples


def sample_from_distribution(NN, NN_prediction, nb_samples, sess=None):
    if sess is None:
        sess = tf.Session()
    samples = NN.TopLayer_obj.sample(NN_prediction, nb_samples, sess)
    return samples


def compute_histogram_distance(hist_explorer, SSA_hist_samples, NN_hist_samples,
                               hist_bounds, bin_lengths, n_bins, hist_species,
                               setting_id, plot=False):
    md_bin_measure = np.prod(np.array(bin_lengths))
    hist_distance = []
    for species_index in range(len(hist_species)):
        hist_distance_1S = compute_hist_distance_1S(hist_bounds, bin_lengths,
                                                    SSA_hist_samples, NN_hist_samples,
                                                    n_bins, species_index,
                                                    hist_explorer, hist_species,
                                                    setting_id, plot)
        hist_distance.append(hist_distance_1S)

    SSA_md_hist = get_histogram(SSA_hist_samples, hist_bounds, n_bins, len(hist_bounds))
    NN_md_hist = get_histogram(NN_hist_samples, hist_bounds, n_bins, len(hist_bounds))
    md_hist_distance = histogram_distance(NN_md_hist, SSA_md_hist, md_bin_measure)
    hist_distance.append(md_hist_distance)
    return hist_distance


def compute_hist_distance_1S(hist_bounds, bin_lengths, SSA_hist_samples,
                             NN_hist_samples, n_bins, species_index,
                             hist_explorer, hist_species, i, plot):
    hist_bound = hist_bounds[species_index]
    bin_length = bin_lengths[species_index]

    SSA_1S_samples = SSA_hist_samples[:, species_index]
    SSA_1S_hist = get_histogram(SSA_1S_samples, hist_bound, n_bins, 1)

    NN_1S_samples = NN_hist_samples[:, species_index]
    NN_1S_hist = get_histogram(NN_1S_samples, hist_bound, n_bins, 1)

    hist_distance_1S = histogram_distance(NN_1S_hist,
                                          SSA_1S_hist,
                                          bin_length)
    if plot is True:
        make_and_save_plot(i, hist_species[species_index],
                           NN_1S_hist, SSA_1S_hist,
                           hist_explorer.histogram_folder)
    return hist_distance_1S


def make_and_save_plot(figure_index, species_name, NN_hist, SSA_hist, folder):
    fig = plt.figure(figure_index)
    plt.plot(NN_hist, label='NN - ' + species_name)
    plt.plot(SSA_hist, label='SSA - ' + species_name)
    plt.legend()
    plot_filepath = os.path.join(folder, str(figure_index) + '_' + species_name + '.png')
    plt.savefig(plot_filepath, bbox_inches='tight')
    plt.close()
    return


def _log_results(hist_explorer, nb_settings, mean_hist_distance, hist_species):
    with open(hist_explorer.log_fp, 'w') as f:
        f.write('The mean multidimensional histogram distance, computed on {0} settings, is: {1}.\n'.format(nb_settings,
                                                                                                            str(mean_hist_distance[-1])))
        for i, species_name in enumerate(hist_species):
            f.write('The mean 1d histogram distance for species {0}, computed on {1} settings, is: {2}.\n'.format(species_name,
                                                                                                                  nb_settings,
                                                                                                                  str(mean_hist_distance[i])))
    return


if __name__ == '__main__':
    np.set_printoptions(suppress=True)

    config = tf.ConfigProto()
    # config.gpu_options.per_process_gpu_memory_fraction = 0.3
    sess = tf.Session(config=config)
    K.set_session(sess)

    timestep = float(sys.argv[1])
    nb_past_timesteps = int(sys.argv[2])
    training_dataset_id = int(sys.argv[3])
    validation_dataset_id = int(sys.argv[4])
    model_id = int(sys.argv[5])
    project_folder = str(sys.argv[6])
    model_name = str(sys.argv[7])

    project_explorer = ProjectFileExplorer(project_folder)
    train_explorer = project_explorer.get_DatasetFileExplorer(timestep, training_dataset_id)
    val_explorer = project_explorer.get_DatasetFileExplorer(timestep, validation_dataset_id)
    model_explorer = project_explorer.get_ModelFileExplorer(timestep, model_id)
    NN = load_NN(model_explorer)
    CRN_module = import_module("stochnet.CRN_models." + model_name + '_py3')
    CRN_class = getattr(CRN_module, model_name)

    steps = [1, 5]
    mean_train_hist_dist = evaluate_model_on_dataset(train_explorer, nb_past_timesteps,
                                                     NN, sess, model_id, CRN_class,
                                                     steps, plot=True,
                                                     log_results=True)
    mean_val_hist_dist = evaluate_model_on_dataset(val_explorer, nb_past_timesteps,
                                                   NN, sess, model_id, CRN_class,
                                                   steps, plot=True,
                                                   log_results=True)
