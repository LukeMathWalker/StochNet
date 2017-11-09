import sys
import os
from functools import partial
from datetime import datetime
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
    # get_custom_objects().update({"exp": lambda x: tf.exp(x),
    #                             "loss_function": NN.TopLayer_obj.loss_function})
    with CustomObjectScope({"exp": lambda x: tf.exp(x), "loss_function": NN.TopLayer_obj.loss_function}):
        NN.load_model(model_explorer.keras_fp)
        NN.load_weights(model_explorer.weights_fp)
    return NN


def evaluate_model_on_dataset(dataset_explorer, nb_past_timesteps, NN, sess,
                              model_id, CRN_class, n_bins=200, plot=False,
                              log_results=False):
    hist_bounds = CRN_class.get_histogram_bounds()
    bin_lengths = [(b[1] - b[0]) / n_bins for b in hist_bounds]
    hist_species, hist_species_indexes = get_histogram_species_w_indexes(CRN_class)

    SSA_traj = get_SSA_traj(dataset_explorer)
    SSA_hists_samples = get_SSA_hists_samples(SSA_traj, hist_species_indexes)

    nb_traj = SSA_traj.shape[1]
    NN_hists_samples = get_NN_hists_samples(NN, dataset_explorer, nb_traj,
                                            nb_past_timesteps, sess,
                                            hist_species_indexes)

    hist_explorer = dataset_explorer.get_HistogramFileExplorer(model_id)
    return compute_histogram_distance(hist_explorer, SSA_hists_samples, NN_hists_samples,
                                      hist_bounds, bin_lengths, n_bins, hist_species,
                                      plot=plot, log_results=log_results)


def get_histogram_species_w_indexes(CRN_class):
    species = CRN_class.get_species()
    histogram_species = CRN_class.get_species_for_histogram()
    histogram_species_indexes = [species.index(s) for s in histogram_species]
    return (histogram_species, histogram_species_indexes)


def get_SSA_traj(dataset_explorer):
    with open(dataset_explorer.histogram_dataset_fp, 'rb') as f:
        SSA_traj = np.load(f)
    return SSA_traj


def get_SSA_hists_samples(SSA_traj, histogram_species_indexes):
    nb_settings = SSA_traj.shape[0]
    SSA_hists_samples = []
    for i in range(nb_settings):
        SSA_hist_samples = SSA_traj[i, :, -1, histogram_species_indexes]
        SSA_hists_samples.append(SSA_hist_samples)
    return SSA_hists_samples


def get_NN_hists_samples(NN, dataset_explorer, nb_traj, nb_past_timesteps, sess,
                         hist_species_indexes):
    rescaled_settings = get_hist_settings_rescaled(dataset_explorer, NN.scaler)
    nb_settings = rescaled_settings.shape[0]
    NN_hists_samples = []
    for i in range(nb_settings):
        NN_hist_samples = get_NN_hist_samples(NN, rescaled_settings[i],
                                              hist_species_indexes,
                                              nb_past_timesteps, nb_traj, sess)
        NN_hists_samples.append(NN_hist_samples)
    return NN_hists_samples


def get_hist_settings_rescaled(dataset_explorer, scaler):
    with open(dataset_explorer.histogram_settings_fp, 'rb') as f:
        settings = np.load(f)
    settings_rescaled = rescale(settings, scaler)
    return settings_rescaled


def get_NN_hist_samples(NN, rescaled_setting, hist_species_indexes,
                        nb_past_timesteps, nb_traj, sess):
    setting = rescaled_setting.reshape(1, nb_past_timesteps, -1)
    NN_prediction = NN.predict_on_batch(setting)
    rescaled_NN_samples = sample_from_distribution(NN, NN_prediction,
                                                   nb_traj, sess)
    NN_samples = scale_back(rescaled_NN_samples, NN.scaler)
    NN_samples = NN_samples[:, 0, hist_species_indexes]
    return NN_samples


def sample_from_distribution(NN, NN_prediction, nb_samples, sess=None):
    if sess is None:
        sess = tf.Session()
    samples = NN.TopLayer_obj.sample(NN_prediction, nb_samples, sess)
    return samples


def compute_histogram_distance(hist_explorer, SSA_hists_samples, NN_hists_samples,
                               hist_bounds, bin_lengths, n_bins, hist_species,
                               plot=False, log_results=False):
    md_bin_measure = np.prod(np.array(bin_lengths))
    hist_distances = []
    for i, _ in enumerate(zip(SSA_hists_samples, NN_hists_samples)):
        SSA_hist_samples, NN_hist_samples = _
        hist_distance = []
        for species_index in range(len(hist_species)):
            print(hist_bounds)
            hist_bound = hist_bounds[species_index]
            print(hist_bound)
            bin_length = bin_lengths[species_index]

            SSA_1S_samples = SSA_hist_samples[:, species_index]
            SSA_1S_hist = get_histogram(SSA_1S_samples, hist_bound, n_bins)

            NN_1S_samples = NN_hist_samples[:, species_index]
            NN_1S_hist = get_histogram(NN_1S_samples, hist_bound, n_bins)

            hist_distance_1S = histogram_distance(NN_1S_hist,
                                                  SSA_1S_hist,
                                                  bin_length)
            hist_distance.append(hist_distance_1S)

            if plot is True:
                make_and_save_plot(i, hist_species[species_index],
                                   NN_1S_hist, SSA_1S_hist,
                                   hist_explorer.histogram_folder)

        SSA_md_hist = get_histogram(SSA_hist_samples, hist_bounds, n_bins)
        NN_md_hist = get_histogram(NN_hist_samples, hist_bounds, n_bins)
        md_hist_distance = histogram_distance(NN_md_hist, SSA_md_hist, md_bin_measure)
        hist_distance.append(md_hist_distance)

        hist_distances.append(hist_distance)

    nb_settings = len(SSA_hists_samples)
    hist_distances = np.array(hist_distances)
    mean_hist_distance = np.mean(hist_distances, axis=0)
    if log_results is True:
        with open(hist_explorer.log_fp, 'w') as f:
            f.write('The mean multidimensional histogram distance, computed on {0} settings, is:'.format(nb_settings))
            f.write('{0}'.format(str(mean_hist_distance[-1])))
            for i, species_name in enumerate(hist_species):
                f.write('The mean 1d histogram distance for species {0}, computed on {1} settings, is:'.format(species_name, nb_settings))
                f.write('{0}'.format(str(mean_hist_distance[i])))

    return mean_hist_distance


def compute_one_d_histogram_distance(NN_hist_samples, SSA_hist_samples, bin_lengths,
                                     hist_species, species_index, setting_index,
                                     hist_explorer, plot):
    other_species_indexes = tuple(set(range(len(NN_hist.shape))).difference(set([species_index])))
    one_d_NN_hist = np.add.reduce(NN_hist, other_species_indexes)
    one_d_SSA_hist = np.add.reduce(SSA_hist, other_species_indexes)
    bin_length = bin_lengths[species_index]
    one_d_hist_distance = histogram_distance(one_d_NN_hist,
                                             one_d_SSA_hist,
                                             bin_length)
    if plot is True:
        make_and_save_plot(setting_index, hist_species[species_index],
                           one_d_NN_hist, one_d_SSA_hist,
                           hist_explorer.histogram_folder)
    return one_d_hist_distance


def make_and_save_plot(figure_index, species_name, NN_hist, SSA_hist, folder):
    fig = plt.figure(figure_index)
    plt.plot(NN_hist, label='NN - ' + species_name)
    plt.plot(SSA_hist, label='SSA - ' + species_name)
    plt.legend()
    plot_filepath = os.path.join(folder, str(figure_index) + '_' + species_name + '.png')
    plt.savefig(plot_filepath, bbox_inches='tight')
    plt.close()
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

    mean_train_hist_dist = evaluate_model_on_dataset(train_explorer, nb_past_timesteps,
                                                     NN, sess, model_id, CRN_class,
                                                     plot=True, log_results=True)
    mean_val_hist_dist = evaluate_model_on_dataset(val_explorer, nb_past_timesteps,
                                                   NN, sess, model_id, CRN_class,
                                                   plot=True, log_results=True)
