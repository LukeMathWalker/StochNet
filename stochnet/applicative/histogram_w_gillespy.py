import sys
import os
import tensorflow as tf
from functools import partial

from importlib import import_module
import numpy as np
import matplotlib.pyplot as plt

from keras.utils.generic_utils import get_custom_objects
from keras import backend as K
from keras.utils import CustomObjectScope

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
    hist_species, hist_species_indexes = get_histogram_species_w_indexes(CRN_class)

    SSA_traj = get_SSA_traj(dataset_explorer)
    SSA_hists = get_SSA_hists(SSA_traj, hist_bounds,
                              hist_species_indexes, n_bins)

    nb_traj = SSA_traj.shape[1]
    NN_hists = get_NN_hists(NN, dataset_explorer, nb_traj, nb_past_timesteps,
                            sess, hist_bounds, hist_species_indexes,
                            n_bins)

    hist_explorer = dataset_explorer.get_HistogramFileExplorer(model_id)
    bin_lengths = [(b[1] - b[0]) / n_bins for b in hist_bounds]
    return compute_histogram_distance(hist_explorer, SSA_hists, NN_hists,
                                      bin_lengths, hist_species,
                                      plot=plot, log_results=log_results)


def compute_histogram_distance(hist_explorer, SSA_hists, NN_hists,
                               bin_lengths, hist_species, plot=False,
                               log_results=False):
    md_bin_measure = np.prod(np.array(bin_lengths))
    hist_distances = []
    for SSA_hist, NN_hist, i in enumerate(zip(SSA_hists, NN_hists)):
        hist_distance = []
        for species_index in range(hist_species):
            one_d_hist_distance = compute_one_d_histogram_distance(NN_hist, SSA_hist, bin_lengths,
                                                                   hist_species, species_index, i,
                                                                   hist_explorer, plot)
            hist_distance.append(one_d_hist_distance)

        md_hist_distance = histogram_distance(NN_hist, SSA_hist, md_bin_measure)
        hist_distance.append(md_hist_distance)

        hist_distances.append(hist_distance)

    nb_settings = len(SSA_hist)
    hist_distances = np.array(hist_distances)
    mean_hist_distance = np.mean(hist_distances, axis=0)
    if log_results is True:
        with open(hist_explorer.log_fp, 'w') as f:
            f.write('The mean multidimensional histogram distance, computed on {0} settings, is:'.format(nb_settings))
            f.write('{0}'.format(str(mean_hist_distance[-1])))
            for species_name, i in enumerate(hist_species):
                f.write('The mean 1d histogram distance for species {0}, computed on {1} settings, is:'.format(species_name, nb_settings))
                f.write('{0}'.format(str(mean_hist_distance[i])))

    return mean_hist_distance


def compute_one_d_histogram_distance(NN_hist, SSA_hist, bin_lengths,
                                     hist_species, species_index, setting_index,
                                     hist_explorer, plot):
    one_d_NN_hist = NN_hist[species_index]
    one_d_SSA_hist = SSA_hist[species_index]
    bin_length = bin_lengths[species_index]
    one_d_hist_distance = histogram_distance(one_d_NN_hist,
                                             one_d_SSA_hist,
                                             bin_length)
    if plot is True:
        make_and_save_plot(setting_index, hist_species[species_index],
                           one_d_NN_hist, one_d_SSA_hist,
                           hist_explorer.histogram_folder)
    return one_d_hist_distance


def get_hist_settings_rescaled(dataset_explorer, scaler):
    with open(dataset_explorer.histogram_settings_fp, 'rb') as f:
        settings = np.load(f)
    settings_rescaled = rescale(settings, scaler)
    return settings_rescaled


def get_SSA_traj(dataset_explorer):
    with open(dataset_explorer.histogram_dataset_fp, 'rb') as f:
        SSA_traj = np.load(f)
    return SSA_traj


def get_SSA_hists(SSA_traj, histogram_bounds, histogram_species_indexes, n_bins):
    nb_settings = SSA_traj.shape[0]
    SSA_hists = []
    for i in range(nb_settings):
        SSA_hist_samples = SSA_traj[i, :, -1, histogram_species_indexes]
        SSA_hist = get_histogram(SSA_hist_samples.T, histogram_bounds, n_bins)
        SSA_hists.append(SSA_hist)
    return SSA_hists


def get_NN_hists(NN, dataset_explorer, nb_traj, nb_past_timesteps, sess,
                 histogram_bounds, histogram_species_indexes, n_bins):
    compute_histogram = partial(get_histogram,
                                histogram_bounds=histogram_bounds,
                                n_bins=n_bins)

    rescaled_settings = get_hist_settings_rescaled(dataset_explorer, NN.scaler)
    nb_settings = rescaled_settings.shape[0]
    NN_hists = []
    for i in range(nb_settings):
        NN_hist = get_NN_hist(NN, rescaled_settings[i],
                              compute_histogram, histogram_species_indexes,
                              nb_past_timesteps, nb_traj, sess)
        NN_hists.append(NN_hist)
    return NN_hists


def get_NN_hist(NN, rescaled_setting, compute_histogram, histogram_species_indexes,
                nb_past_timesteps, nb_traj, sess):
    setting = rescaled_setting.reshape(1, nb_past_timesteps, -1)
    print(setting.shape)
    NN.model.summary()
    NN_prediction = NN.predict_on_batch(setting)
    rescaled_NN_samples = sample_from_distribution(NN, NN_prediction,
                                                   nb_traj, sess)
    NN_samples = scale_back(rescaled_NN_samples, NN.scaler)
    NN_samples = NN_samples[:, 0, histogram_species_indexes]
    NN_hist = compute_histogram(NN_samples.T)
    return NN_hist


def get_histogram_species_w_indexes(CRN_class):
    species = CRN_class.get_species()
    histogram_species = CRN_class.get_species_for_histogram()
    histogram_species_indexes = [species.index(s) for s in histogram_species]
    return (histogram_species, histogram_species_indexes)


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

    print(locals())

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
