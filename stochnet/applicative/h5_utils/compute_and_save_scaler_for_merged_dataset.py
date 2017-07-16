from sklearn.preprocessing import StandardScaler
import dill
import numpy as np


def _incremental_mean_and_var(new_mean, new_variance, new_sample_count, last_mean=.0, last_variance=None, last_sample_count=0):
    """Calculate mean update and a Youngs and Cramer variance update.
    last_mean and last_variance are statistics computed at the last step by the
    function. Both must be initialized to 0.0. In case no scaling is required
    last_variance can be None. The mean is always required and returned because
    necessary for the calculation of the variance. last_n_samples_seen is the
    number of samples encountered until now.
    From the paper "Algorithms for computing the sample variance: analysis and
    recommendations", by Chan, Golub, and LeVeque.
    Parameters
    ----------
    X : array-like, shape (n_samples, n_features)
        Data to use for variance update
    last_mean : array-like, shape: (n_features,)
    last_variance : array-like, shape: (n_features,)
    last_sample_count : int
    Returns
    -------
    updated_mean : array, shape (n_features,)
    updated_variance : array, shape (n_features,)
        If None, only mean is computed
    updated_sample_count : int
    References
    ----------
    T. Chan, G. Golub, R. LeVeque. Algorithms for computing the sample
        variance: recommendations, The American Statistician, Vol. 37, No. 3,
        pp. 242-247
    Also, see the sparse implementation of this in
    `utils.sparsefuncs.incr_mean_variance_axis` and
    `utils.sparsefuncs_fast.incr_mean_variance_axis0`
    """
    # old = stats until now
    # new = the current increment
    # updated = the aggregated stats
    last_sum = last_mean * last_sample_count
    new_sum = new_mean * new_sample_count

    updated_sample_count = last_sample_count + new_sample_count

    updated_mean = (last_sum + new_sum) / updated_sample_count

    if last_variance is None:
        updated_variance = None
    else:
        new_unnormalized_variance = new_variance * new_sample_count
        if last_sample_count == 0:  # Avoid division by 0
            updated_unnormalized_variance = new_unnormalized_variance
        else:
            last_over_new_count = last_sample_count / new_sample_count
            last_unnormalized_variance = last_variance * last_sample_count
            updated_unnormalized_variance = (
                last_unnormalized_variance +
                new_unnormalized_variance +
                last_over_new_count / updated_sample_count *
                (last_sum / last_over_new_count - new_sum) ** 2)
        updated_variance = updated_unnormalized_variance / updated_sample_count

    return updated_mean, updated_variance, updated_sample_count


scaler_for_merged_data = StandardScaler()
scaler_for_merged_data.mean_ = 0
scaler_for_merged_data.var_ = 0
scaler_for_merged_data.n_samples_seen_ = 0

for i in range(2, 7):
    scaler_filepath = '/home/lucap/Documenti/Data storage/SIR/scaler_for_timestep_2-5_dataset_big_0' + str(i) + '.h5'
    with open(scaler_filepath, mode='rb') as f:
        scaler = dill.load(f)
    print(scaler.n_samples_seen_)
    scaler_for_merged_data.mean_, scaler_for_merged_data.var_, scaler_for_merged_data.n_samples_seen_ = _incremental_mean_and_var(scaler.mean_, scaler.var_, scaler.n_samples_seen_, scaler_for_merged_data.mean_, scaler_for_merged_data.var_, scaler_for_merged_data.n_samples_seen_)

scaler_for_merged_data.scale_ = np.sqrt(scaler_for_merged_data.var_)
merged_scaler_filepath = '/home/lucap/Documenti/Data storage/SIR/scaler_for_timestep_2-5_dataset_v_big_01.h5'
with open(merged_scaler_filepath, mode='wb') as f:
    dill.dump(scaler_for_merged_data, f)
