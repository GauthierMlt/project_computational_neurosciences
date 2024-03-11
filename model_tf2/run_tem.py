#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: James Whittington
"""

import data_utils as d_u
import model_utils as m_u
import parameters

import numpy as np
import tensorflow as tf

import glob
import os
import shutil
import tem_model as tem
import time
import importlib

run_start_time = time.time()

importlib.reload(tem)

# Create directories for storing all information about the current run
run_path, train_path, model_path, save_path, script_path, envs_path = d_u.make_directories()
# Save all python files in current directory to script directory
files = glob.iglob(os.path.join('', '*.py'))
for file in files:
    if os.path.isfile(file):
        shutil.copy2(file, os.path.join(script_path, file))

# Initialise hyper-parameters for model
params = parameters.default_params()
# Save parameters
np.save(os.path.join(save_path, 'params'), dict(params))

tf.keras.backend.set_floatx('float32')
params.precision = m_u.precision

# Create instance of TEM with those parameters
model = tem.TEM(params)

# Create a logger to write log output to file
logger_sums = d_u.make_logger(run_path, 'summaries')
logger_envs = d_u.make_logger(run_path, 'env_details')
# Create a tensor board to stay updated on training progress. Start tensorboard with tensorboard --logdir=runs
summary_writer = tf.summary.create_file_writer(train_path)

# Make an ADAM optimizer for TEM
optimizer = tf.keras.optimizers.Adam(learning_rate=params.learning_rate_max)

# Specify training setup
debug, debug_data, profile = False, False, False
optimise = True
tf.config.run_functions_eagerly(debug or debug_data or not params.graph_mode)
if profile or debug:
    params.train_iters = 11
    params.save_walk = 1


@tf.function  # (experimental_compile=True)
def train_step(model_, model_inputs_):
    with tf.GradientTape() as tape:
        print('start_model')
        variables_, re_input_ = model_(model_inputs_, training=True)
        # collate inputs for model
        losses_ = tem.compute_losses(model_inputs_, variables_, model_.trainable_variables, params)
    if optimise:
        gradients_ = tape.gradient(losses_.train_loss, model_.trainable_variables)
        capped_grads = [(tf.clip_by_norm(grad, 2), var) if grad is not None else (grad, var) for grad, var in
                        zip(gradients_, model_.trainable_variables)]
        optimizer.apply_gradients(capped_grads)
    print('train_step_optimiser_done')
    return variables_, re_input_, losses_


@tf.function  # (experimental_compile=True)
def test_step(model_, model_inputs_):
    variables_, re_input_ = model_(model_inputs_, training=False)
    return variables_, re_input_


# initialise dictionary to contain environments and data info
train_dict = d_u.get_initial_data_dict(params)

msg = 'Training Started'
logger_sums.info(msg)
logger_envs.info(msg)
train_i = 0
for train_i in range(params.train_iters):

    # INITIALISE ENVIRONMENT AND INPUT VARIABLES
    if sum(train_dict.env_steps == 0) > 0:
        msg = str(sum(train_dict.env_steps == 0)) + ' New Environments ' + str(train_i) + ' ' + str(
            train_i * params.seq_len)
        logger_envs.info(msg)

    # Get scaling parameters
    scalings = parameters.get_scaling_parameters(train_i, params)
    optimizer.lr = scalings.l_r

    data_start_time = time.time()
    # collect batch-specific environment data
    train_dict = d_u.data_step(train_dict, params)
    # convert all inputs to tensors - otherwise re-build graph every time
    inputs_tf = m_u.inputs_2_tf(train_dict.inputs, train_dict.hebb, scalings)

    model_start_time = time.time()
    if debug_data:
        continue
    elif profile and train_i > 0:
        # start profiler after graph set-up
        if train_i == 1:
            tf.profiler.experimental.start(train_path)
        # profile
        with tf.profiler.experimental.Trace('train', step_num=train_i, _r=1):
            variables, re_input, losses = train_step(model, inputs_tf)
    else:
        variables, re_input, losses = train_step(model, inputs_tf)
    stop_time = time.time()
    # feeding in correct initial states
    re_input = m_u.tf2numpy(re_input)
    train_dict.variables.gs, memories_dict = re_input.g, re_input.memories_dict

    if not params.graph_mode:
        print('n_states_visited', np.sum(np.sum(train_dict.variables.edge_visits, axis=2) > 0, axis=1))

    # reorganise memories
    train_dict.hebb = d_u.new2stored_memories(memories_dict, params)

    # Update logging info
    train_dict.curric_env.data_step_time = model_start_time - data_start_time
    train_dict.curric_env.model_step_time = stop_time - model_start_time
    msg = 'Step {:.0f}, data time : {:.4f} , model time {:.4f}'.format(train_i, model_start_time - data_start_time,
                                                                       stop_time - model_start_time)
    logger_envs.info(msg)

    # try to find nans / infs in the weightings
    if m_u.is_any_nan_inf([m_u.nested_isnan_inf(re_input), m_u.nested_isnan_inf(model.trainable_weights),
                           m_u.nested_isnan_inf(variables), m_u.nested_isnan_inf(inputs_tf)]):
        print('NANS/INFS somewhere')
        breakpoint()

    # Log training progress summary statistics
    if train_i % params.sum_int == 0:
        accuracies = m_u.compute_accuracies(inputs_tf.x, variables.pred, train_dict.inputs.s_visited, params)
        summaries = m_u.make_summaries(losses, accuracies, scalings, variables, train_dict.curric_env,
                                       train_dict.inputs.seq_index, model.trainable_variables, params)

        for key_, val_ in summaries.items():
            with summary_writer.as_default():
                tf.summary.scalar(key_, val_, step=train_i)
        summary_writer.flush()

        msg = "T={:.2f}, train_i={:.2f}, total_steps={:.2f}".format(scalings.temp, train_i, train_i * params.seq_len)
        logger_sums.info(msg)
        msg = "it={:.5f}, lxG={:.2f}, lxGt={:.2f}, lg={:.2f}, aGinf={:.2f}, aGt={:.2f}".format(train_i, losses.lx_g,
                                                                                               losses.lx_gt, losses.lg,
                                                                                               accuracies.g,
                                                                                               accuracies.gt)
        logger_sums.info(msg)

    # Log current model performance summaries - requires running full environments from scratch
    if params.summary_inference and train_i % params.sum_int_inferences == 0:
        start_time = time.time()
        d_u.summary_inferences(train_i, model, test_step, summary_writer, params)
        logger_envs.info(
            "summary inference time {:.2f}, train_i={:.2f}, total_steps={:.2f}".format(time.time() - start_time,
                                                                                       train_i,
                                                                                       train_i * params.seq_len))

    # Save model parameters which can be loaded later to analyse model
    if train_i % params.save_interval == 0 and train_i > 0:
        start_time = time.time()

        # save model checkpoint
        model.save_weights(model_path + '/tem_' + str(train_i))
        logger_sums.info("save data time {:.2f}, train_i={:.2f}, total_steps={:.2f}".format(time.time() - start_time,
                                                                                            train_i,
                                                                                            train_i * params.seq_len))
if profile:
    tf.profiler.experimental.stop()
print('Finished training')

# save final copy of model
d_u.save_model_outputs(model, m_u, train_i, save_path, params, test_step=test_step)
model.save_weights(model_path + '/tem_' + str(train_i))
