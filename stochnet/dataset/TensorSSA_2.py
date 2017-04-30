import tensorflow as tf
# We use the First Reaction Method! It's easier to vectorize


def prop_infection(system_state):
    return 3. * system_state[0] * system_state[1] / (system_state[0] + system_state[1] + system_state[2])


def prop_recover(system_state):
    return 1. * system_state[1]


endtime = 1.

initial_setting = tf.Variable(tf.constant([120., 25., 43.]), name='initial_setting')
timestamp = tf.Variable(tf.constant(0.), name='timestamp')
system_state = tf.Variable(initial_setting.initialized_value(), name='system_state')
tau = tf.Variable(0., name='tau')
reaction_index = tf.Variable(0, name='reaction_index', dtype=tf.int64)
random = tf.Variable([0., 0.], name='random')
tau_candidates = tf.Variable([0., 0.], name='random')

null_update = tf.constant([0., 0., 0.])
infection_update = tf.constant([-1., 1., 0.])
recover_update = tf.constant([0., -1., 1.])
updates = tf.stack([null_update, infection_update, recover_update], axis=0)

propensity_infection = prop_infection(system_state)
propensity_recover = prop_recover(system_state)
total_propensity = propensity_infection + propensity_recover


def null_update():
    with tf.control_dependencies([tau.assign(0.)]):
        return reaction_index.assign(0)


def eff_update():
    with tf.control_dependencies([tau.assign(tf.reduce_min(tau_candidates.assign(-tf.log(random.assign(tf.random_uniform([2], minval=0, maxval=1))) / total_propensity)))]):
        return reaction_index.assign(tf.argmin(tau_candidates, axis=0) + tf.constant(1, dtype=tf.int64))


result = tf.cond(tf.equal(total_propensity, 0.),
                 null_update,
                 eff_update
                 )

update_to_be_done = tf.gather(updates, indices=reaction_index)

_timestamp = timestamp + tau


def simulation_end():
    return timestamp.assign(endtime)


def simulation_update():
    with tf.control_dependencies([system_state.assign(system_state + update_to_be_done)]):
        return timestamp.assign(_timestamp)


result_2 = tf.cond(tf.greater(_timestamp, endtime),
                   simulation_end,
                   simulation_update)

init_op = tf.global_variables_initializer()

with tf.Session() as sess:
    # Run the init operation.
    sess.run(tf.variables_initializer([initial_setting, system_state]))
    sess.run(init_op)
    for i in range(150):
        sess.run(result)
        sess.run(result_2)
    print(system_state.eval())
    print(timestamp.eval())
