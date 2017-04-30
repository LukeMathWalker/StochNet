import tensorflow as tf
import numpy as np
from tqdm import tqdm
# We use the First Reaction Method! It's easier to vectorize


def prop_infection(system_state):
    return 3. * system_state[:, 0] * system_state[:, 1] / N


def prop_recover(system_state):
    return 1. * system_state[:, 1]


nb_trajectories = 1000
nb_reactions = 2
endtime = tf.constant(1.)


initial_setting = tf.placeholder(tf.float32, shape=(1, 3))
N = tf.cast(tf.reduce_sum(initial_setting), dtype=tf.float32)
# system_state_init = np.broadcast_to(initial_setting, (nb_trajectories, 3))
system_state_init = tf.tile(initial_setting, tf.stack([nb_trajectories, 1]))

timestamp = tf.Variable(tf.zeros(nb_trajectories), name='timestamp')
system_state = tf.Variable(system_state_init, name='system_state', dtype=tf.float32)
tau = tf.Variable(tf.zeros((nb_trajectories,)), name='tau')
reaction_index = tf.Variable(tf.zeros((nb_trajectories,), dtype=tf.int64), name='reaction_index')
random = tf.Variable(tf.random_uniform([nb_trajectories, 2], minval=0, maxval=1, dtype=tf.float32), name='random')
tau_candidates = tf.Variable(tf.zeros(nb_reactions), name='random')

null_update = tf.constant([0., 0., 0.])
infection_update = tf.constant([-1., 1., 0.])
recover_update = tf.constant([0., -1., 1.])
updates = tf.stack([null_update, infection_update, recover_update], axis=0)


propensity_infection = prop_infection(system_state)
propensity_recover = prop_recover(system_state)
propensity = tf.stack([propensity_infection, propensity_recover], axis=1)


def null_update(x):
    return [tf.constant(0.), tf.constant(0, dtype=tf.int64)]


def eff_update(x):
    y = tau_candidates.assign(-tf.log(x[0, :]) / x[1, :])
    return [tf.reduce_min(y), tf.argmin(y, axis=0) + tf.constant(1, dtype=tf.int64)]


def map_function(x):
    return tf.cond(tf.equal(tf.reduce_sum(x[1, :]), 0.), lambda: null_update(x), lambda: eff_update(x))


tau, reaction_index = tf.map_fn(map_function, tf.stack([random, propensity], axis=1), dtype=[tf.float32, tf.int64])

update_to_be_done = tf.map_fn(lambda x: tf.gather(updates, indices=x), reaction_index, dtype=tf.float32, parallel_iterations=10000, back_prop=False, swap_memory=True)

_timestamp = timestamp + tau


def simulation_end():
    return [endtime, tf.constant([0., 0., 0.])]


def simulation_update(x):
    return [x[0], x[1:]]


def map_function_2(x):
    return tf.cond(tf.greater_equal(x[0], endtime), lambda: simulation_end(), lambda: simulation_update(x))


timestamp_1, _system_state = tf.map_fn(map_function_2, tf.concat([tf.reshape(_timestamp, [-1, 1]), update_to_be_done], axis=1), dtype=[tf.float32, tf.float32], parallel_iterations=10000, back_prop=False, swap_memory=True)

timestamp_update = timestamp.assign(timestamp_1)
system_update = system_state.assign_add(_system_state)
step = tf.group(timestamp_update, system_update)

init_op = tf.global_variables_initializer()

with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
    nb_settings = 20
    initial_settings = np.random.randint(low=2, high=200, size=(1, nb_settings, 3))
    for j in tqdm(range(nb_settings)):
        sess.run(init_op, feed_dict={initial_setting: initial_settings[:, j]})
        for i in range(100):
            sess.run(step, feed_dict={initial_setting: initial_settings[:, j]})
        print('Done!')
