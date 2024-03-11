#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: James Whittington
"""

import environments
import model_utils as m_u
import parameters

import tensorflow as tf
import numpy as np
import copy as cp
import os
import datetime
import logging
import time
from distutils.dir_util import copy_tree


def cell_norm_online(cells, positions, current_cell_mat, pars):
    # for separate environments within each batch
    n_states = pars.n_states_world
    n_envs_save = pars.n_envs_save

    num_cells = np.shape(cells)[1]
    n_trials = np.shape(cells)[2]

    cell_mat = [np.zeros((n_states[env], num_cells)) for env in range(n_envs_save)]

    new_cell_mat = []

    for env in range(n_envs_save):
        for ii in range(n_trials):
            position = int(positions[env, ii])
            cell_mat[env][position, :] += cells[env, :, ii]
        if current_cell_mat is None:
            new_cell_mat.append(cell_mat[env])
        else:
            new_cell_mat.append(cell_mat[env] + current_cell_mat[env])

    return new_cell_mat


def accuracy_positions_online(cells, positions, current_cell_mat, pars):
    # for separate environments within each batch
    n_states = pars.n_states_world
    n_envs_save = pars.n_envs_save
    n_trials = np.shape(cells)[1]

    cell_mat = [np.zeros(n_states[env]) for env in range(n_envs_save)]

    new_cell_mat = []

    for env in range(n_envs_save):
        for ii in range(n_trials):
            position = int(positions[env, ii])
            cell_mat[env][position] += cells[env, ii]
        if current_cell_mat is None:
            new_cell_mat.append(cell_mat[env])
        else:
            new_cell_mat.append(cell_mat[env] + current_cell_mat[env])

    return new_cell_mat


def correct_action_pred_np(real, pred):
    correct_prediction = np.equal(real, np.round(pred)).astype(np.float32)
    accuracy = np.floor(np.mean(correct_prediction, 1, keepdims=True))
    return accuracy.astype(np.int32)


def positions_online(position, positions, n_envs_save):
    new_positions = []
    for env in range(n_envs_save):
        if positions is None:
            new_positions.append(cp.deepcopy(position[env]))
        else:
            new_positions.append(np.concatenate((positions[env], position[env]), axis=0))

    return new_positions


def sense_online(sense, senses, n_seqs):
    senses_ = []
    for seq in range(n_seqs):
        senses_.append(np.argmax(sense[seq], 1))
    senses_ = np.transpose(np.squeeze(np.array(senses_)), [1, 0])

    if senses is None:
        return cp.deepcopy(senses_)
    else:
        return np.concatenate((senses, senses_), axis=1)


def accuracy_online(accs, acc_fn, real, pred, n_seqs):
    acc = []
    for seq in range(n_seqs):
        acc.append(acc_fn(real[:, :, seq], pred[seq]))
    acc = np.transpose(np.squeeze(np.array(acc)), [1, 0])

    if accs is None:
        accs = cp.deepcopy(acc)
    else:
        accs = np.concatenate((accs, acc), axis=1)

    return acc, accs


def inference_opportunity_online(inference_opportunitys, inference_opportunity):
    if inference_opportunitys is None:
        accs = cp.deepcopy(inference_opportunity)
    else:
        accs = np.concatenate((inference_opportunitys, inference_opportunity), axis=1)

    return accs


def acc_sense(real, pred):
    accuracy = np.equal(np.argmax(real, 1), np.argmax(pred, 1))
    accuracy = np.expand_dims(accuracy, 1)
    return accuracy.astype(np.int32)


def make_directories(base_path='../Summaries/'):
    """
    Creates directories for storing data during a model training run
    """

    # Get current date for saving folder
    date = datetime.datetime.today().strftime('%Y-%m-%d')
    # Initialise the run and dir_check to create a new run folder within the current date
    run = 0
    dir_check = True
    # Initialise all pahts
    train_path, model_path, save_path, script_path, run_path, envs_path = None, None, None, None, None, None
    # Find the current run: the first run that doesn't exist yet
    while dir_check:
        # Construct new paths
        run_path = base_path + date + '/run' + str(run) + '/'
        train_path = run_path + 'train'
        model_path = run_path + 'model'
        save_path = run_path + 'save'
        script_path = run_path + 'script'
        envs_path = script_path + '/envs'
        run += 1
        # And once a path doesn't exist yet: create new folders
        if not os.path.exists(train_path) and not os.path.exists(model_path) and not os.path.exists(save_path):
            os.makedirs(train_path)
            os.makedirs(model_path)
            os.makedirs(save_path)
            os.makedirs(script_path)
            os.makedirs(envs_path)
            dir_check = False
    # Return folders to new path
    return run_path, train_path, model_path, save_path, script_path, envs_path


def set_directories(date, run, base_path='../Summaries/'):
    """
    Returns directories for storing data during a model training run from a given previous training run
    """

    # Initialise all paths
    run_path = base_path + date + '/run' + str(run) + '/'
    train_path = run_path + 'train'
    model_path = run_path + 'model'
    save_path = run_path + 'save'
    script_path = run_path + 'script'
    envs_path = script_path + '/envs'
    # Return folders to new path
    return run_path, train_path, model_path, save_path, script_path, envs_path


def make_logger(run_path, name):
    """
    Creates logger so output during training can be stored to file in a consistent way
    """

    # Create new logger    
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    # Remove anly existing handlers so you don't output to old files, or to new files twice
    # - important when resuming training existing model
    logger.handlers = []
    # Create a file handler, but only if the handler does
    handler = logging.FileHandler(run_path + name + '.log')
    handler.setLevel(logging.INFO)
    # Create a logging format
    formatter = logging.Formatter('%(asctime)s: %(message)s')
    handler.setFormatter(formatter)
    # Add the handlers to the logger
    logger.addHandler(handler)
    # Return the logger object
    return logger


def save_params(pars, save_path, script_path):
    np.save(save_path + '/params', dict(pars))
    copy_tree('./', script_path)
    return


def get_next_batch(position, direction, edge_visits, pars, envs_class):
    xs = environments.get_new_data_diff_envs(position, pars, envs_class)

    # s_visited is for each bptt, saying whether each the state at current timestep has been visited before
    s_visited = np.ones((pars.batch_size, pars.seq_len))
    inference_opportunity = np.zeros((pars.batch_size, pars.seq_len))

    for seq in range(pars.seq_len):
        pos = position[:, seq].astype(int)
        direc = np.argmax(direction[:, :, seq], axis=1)
        # account for 'stay still' directions input all 0's
        direc += pars.n_actions * (np.sum(direction[:, :, seq], axis=1) == 0).astype(int)

        current_node_visits = np.sum(edge_visits, axis=2)[np.arange(pars.batch_size), pos]
        current_edge_visits = edge_visits[np.arange(pars.batch_size), pos, direc]
        # have I visited this position before - 1 if yes, 0 if no
        s_visited[:, seq] = (current_node_visits > 0).astype(int)
        # inference available if arrive at old state from a new direction!
        inference_opportunity[:, seq] = (np.logical_and(current_edge_visits == 0, current_node_visits > 0)).astype(int)
        # add position to places I've been
        edge_visits[np.arange(pars.batch_size), pos, direc] += 1

    new_data = (xs, edge_visits, s_visited, inference_opportunity)

    return new_data


def initialise_variables(env_steps, data_dict):
    for env, env_step in enumerate(env_steps):
        # only do if just entered environment
        if env_step > 0:
            continue

        data_dict.gs[env, ...] = 0
        data_dict.edge_visits[env, ...] = 0
    return data_dict


def save_data_maps(positions, data_list, save_path, n_envs_save, index, states, names):
    pos_count = [0] * n_envs_save

    for env in range(n_envs_save):
        pos_count[env] = np.bincount(positions[env].astype(np.int32).flatten(), minlength=states[env]) + 0.001
    np.save(save_path + '/pos_count_' + str(index), pos_count)

    for data, name in zip(data_list, names):
        data_map = []
        for env in range(n_envs_save):
            try:
                data_map.append(np.matmul(np.diag(1 / pos_count[env]), data[env]))
            except ZeroDivisionError:
                pass

        np.save(save_path + '/' + name + '_' + str(index), data_map)

        del data_map

    return


def initialise_hebb(env_steps, data_dict, pars):
    for env, env_step in enumerate(env_steps):
        # only do if just entered environment
        if env_step > 0:
            continue
        data_dict.x[env, :, :] = np.zeros((pars.s_size_project, pars.hebb_mems_keep))
        data_dict.g[env, :, :] = np.zeros((pars.phase_size, pars.hebb_mems_keep))
        data_dict.in_use[env, :] = np.zeros(pars.hebb_mems_keep)

    return data_dict


def prepare_data_maps(data, prev_cell_maps, prev_acc_maps, positions, pars):
    gs, position, acc_st = data
    gs_all = prev_cell_maps
    accs_x_to, accs_x_from = prev_acc_maps

    g1s = np.transpose(np.array(cp.deepcopy(gs)), [1, 2, 0])

    pos_to = position[:, :pars.seq_len]
    pos_from = position[:, :pars.seq_len - 1]
    acc_st_from = acc_st[:, 1:pars.seq_len]

    gs_all = cell_norm_online(g1s, pos_to, gs_all, pars)
    accs_x_to = accuracy_positions_online(acc_st, pos_to, accs_x_to, pars)
    accs_x_from = accuracy_positions_online(acc_st_from, pos_from, accs_x_from, pars)

    positions = positions_online(position, positions, pars.n_envs_save)

    cell_list = [gs_all]
    acc_list = [accs_x_to, accs_x_from]

    return acc_list, cell_list, positions


def prepare_cell_timeseries(data, prev_data, pars):
    gs, pos, xs, xs_gt = data
    gs_, pos_, xs_, xs_gt_ = prev_data
    # convert to batch_size x cells x timesteps
    g1s = np.transpose(np.array(cp.deepcopy(gs)), [1, 2, 0])
    g1s = g1s[:pars.n_envs_save, :, :]

    xgt1s = np.transpose(np.array(cp.deepcopy(xs_gt)), [1, 2, 0])
    xgt1s = xgt1s[:pars.n_envs_save, :, :]
    x1s = xs[:pars.n_envs_save, :, :]

    grids, places, positions, senses, senses_pred = [], [], [], [], []

    for env in range(pars.n_envs_save):
        if gs_ is None:
            grids.append(cp.deepcopy(g1s[env]))
            positions.append(cp.deepcopy(pos[env]))
            senses.append(cp.deepcopy(x1s[env]))
            senses_pred.append(cp.deepcopy(xgt1s[env]))
        else:
            grids.append(np.concatenate((gs_[env], g1s[env]), axis=1))
            positions.append(np.concatenate((pos_[env], pos[env]), axis=0))
            senses.append(np.concatenate((xs_[env], x1s[env]), axis=1))
            senses_pred.append(np.concatenate((xs_gt_[env], xgt1s[env]), axis=1))

    return [grids, positions, senses, senses_pred]


def prepare_input(data_dict, pars, start_i=None):
    """
    Select section of walk sequences that gets fed into model, and prepare model input dictionary
    """
    # select right bit of data to send to model
    i1 = data_dict.env_steps * pars.seq_len if start_i is None else start_i
    i2 = i1 + pars.seq_len
    for batch in range(pars.batch_size):
        data_dict.bptt_data.position[batch, :] = data_dict.walk_data.position[batch][i1[batch]:i2[batch]]
        data_dict.bptt_data.direc[batch, ...] = data_dict.walk_data.direc[batch][:, i1[batch]:i2[batch]]
    # convert positions to sensory observations, and get node/edge visit info
    new_data = get_next_batch(data_dict.bptt_data.position, data_dict.bptt_data.direc,
                              data_dict.variables.edge_visits, pars, data_dict.curric_env.envs)
    xs, edge_visits, s_visited, inference_opportunity = new_data

    # model input data
    data_dict.inputs = m_u.DotDict({'xs': xs,
                                    'gs': data_dict.variables.gs,
                                    'ds': data_dict.bptt_data.direc,
                                    'seq_index': np.array(data_dict.env_steps),
                                    's_visited': s_visited,
                                    'positions': data_dict.bptt_data.position,
                                    })
    data_dict.variables.edge_visits = edge_visits
    data_dict.variables.inference_opportunity = inference_opportunity

    # update env_steps
    data_dict.env_steps += 1
    # new environment if finished all data from walk
    data_dict.env_steps[i2 >= [x.shape[1] for x in data_dict.walk_data.direc]] = 0

    return data_dict


def get_initial_data_dict(pars):
    # prepare_environment_data
    data_dict = m_u.DotDict({'env_steps': np.zeros(pars.batch_size).astype(int),
                             'curric_env':
                                 {'envs': [None for _ in range(pars.batch_size)],
                                  'n_restart': pars.env.restart_max,
                                  'walk_len': np.zeros(pars.batch_size).astype(int),
                                  'states_mat': [0 for _ in range(pars.batch_size)],
                                  'adjs': [0 for _ in range(pars.batch_size)],
                                  'trans': [0 for _ in range(pars.batch_size)],
                                  'g_maps': [None for _ in range(pars.batch_size)],
                                  },
                             'hebb': {
                                 'g': np.zeros((pars.batch_size, pars.phase_size, pars.hebb_mems_keep)),
                                 'x': np.zeros((pars.batch_size, pars.s_size_project, pars.hebb_mems_keep)),
                                 'in_use': np.zeros((pars.batch_size, pars.hebb_mems_keep)),
                             },
                             'variables':
                                 {'gs': np.zeros((pars.batch_size, pars.g_size)),
                                  # Visited x state by y action (pars.n_actions + 1 as
                                  # 'stay still' is the extra action)
                                  'edge_visits': np.zeros((pars.batch_size, pars.max_states, pars.env.n_actions + 1)),
                                  'start_state': np.zeros(pars.batch_size),
                                  },
                             'walk_data':
                                 {'position': [0 for _ in range(pars.batch_size)],
                                  'direc': [0 for _ in range(pars.batch_size)],
                                  },
                             'bptt_data':
                                 {'position': np.zeros((pars.batch_size, pars.seq_len)),
                                  'direc': np.zeros((pars.batch_size, pars.env.n_actions, pars.seq_len)),
                                  },
                             })
    return data_dict


def initialise_environments(curric_env, env_steps, pars, test=False):
    for b, (env, env_step) in enumerate(zip(curric_env.envs, env_steps)):
        # only do if just entered environment
        if env_step > 0:
            continue

        if pars.world_type == 'rectangle':
            curric_env.envs[b] = environments.Rectangle(pars, pars.env.widths[b], pars.env.heights[b])
        if pars.world_type == 'hexagonal':
            curric_env.envs[b] = environments.Hexagonal(pars, pars.env.widths[b])
        if pars.world_type == 'family_tree':
            curric_env.envs[b] = environments.FamilyTree(pars, pars.env.widths[b])
        if pars.world_type == 'line_ti':
            curric_env.envs[b] = environments.LineTI(pars, pars.env.widths[b])
        if pars.world_type == 'wood2000':
            curric_env.envs[b] = environments.Wood2000(pars, pars.env.widths[b], pars.env.heights[b])
        if pars.world_type == 'frank2000':
            curric_env.envs[b] = environments.Frank2000(pars, pars.env.widths[b], pars.env.heights[b])
        if pars.world_type == 'grieves2016':
            curric_env.envs[b] = environments.Grieves2016(pars, pars.env.widths[b])
        if pars.world_type == 'sun2020':
            curric_env.envs[b] = environments.Sun2020(pars, pars.env.widths[b])
        if pars.world_type == 'nieh2021':
            curric_env.envs[b] = environments.Nieh2021(pars, pars.env.widths[b])

        curric_env.envs[b].world()
        curric_env.envs[b].state_data()

    for b, (env, env_step) in enumerate(zip(curric_env.envs, env_steps)):
        # only do if just entered environment
        if env_step > 0:
            continue
        # redoing this loop so that the max operation below makes sense (i.e. env.n_states might be meaningless...)
        # choose walk length
        if test:
            walk_len = max(pars.seq_len, pars.env.save_walk * max([env_.n_states for env_ in curric_env.envs]))
        else:
            # asyncrounous environment walks - each env will have different walk length
            # shorter walks for smaller environments
            probs = np.ones(pars.env.seq_jitter)
            batch_rn = curric_env.n_restart + np.random.choice(np.arange(pars.env.seq_jitter), p=probs / sum(probs))
            walk_len = int(batch_rn * env.n_states)
        # needs to be a multiple of pars.seq_len
        walk_len -= walk_len % pars.seq_len
        curric_env.envs[b].walk_len = walk_len
        curric_env.walk_len[b] = walk_len

    return curric_env


def get_walk_data_class(data_dict, envs, env_steps):
    for b, (env, env_step) in enumerate(zip(envs.envs, env_steps)):
        # only do if just entered environment
        if env_step > 0:
            continue
        # should really just do this in class de, i.e. walk() ends with self.position, self.direc = ...
        data_dict.position[b], data_dict.direc[b] = env.walk()
    return data_dict


def data_step(data_dict, pars, test=False):
    """
    could do env step loop here, with curriculum etc only for one env at a time
    """
    curriculum_steps = (pars.env.restart_max - pars.env.restart_min) / (
            400 * (pars.env.restart_max + pars.env.restart_min + pars.env.seq_jitter) / 2)
    data_dict.curric_env.n_restart = np.maximum(data_dict.curric_env.n_restart - curriculum_steps, pars.env.restart_min)

    # make environments
    data_dict.curric_env = initialise_environments(data_dict.curric_env, data_dict.env_steps, pars, test=test)
    # initialise Hebbian matrices
    data_dict.hebb = initialise_hebb(data_dict.env_steps, data_dict.hebb, pars)
    # initialise all other variables
    data_dict.variables = initialise_variables(data_dict.env_steps, data_dict.variables)
    # Collect full sequence of data
    data_dict.walk_data = get_walk_data_class(data_dict.walk_data, data_dict.curric_env, data_dict.env_steps)
    # Select section of walk sequences that gets fed into model, and prepare model input dictionary
    data_dict = prepare_input(data_dict, pars)

    return data_dict


def save_model_outputs(model, m_u_, train_i, iter_path, pars, test_step=None):
    """
    Takes a model and collects cell and environment timeseries from a forward pass
    """
    # Initialise timeseries data to collect
    gs_timeseries, pos_timeseries, xs_timeseries, xs_gt_timeseries, variables_test, memories_dict = None, None, None, \
                                                                                                    None, None, None
    # Initialise model input data
    test_dict = get_initial_data_dict(pars)
    # Run forward pass
    ii, data_continue = 0, True
    while data_continue:
        # Update input
        test_dict = data_step(test_dict, pars, test=True)
        scalings = parameters.get_scaling_parameters(train_i, pars)
        inputs_test_tf = m_u_.inputs_2_tf(test_dict.inputs, test_dict.hebb, scalings)
        # Do model forward pass step
        if test_step:
            variables_test, re_input_test = test_step(model, inputs_test_tf)
        else:
            variables_test, re_input_test = model(inputs_test_tf, training=False)
        re_input_test = m_u_.tf2numpy(re_input_test)
        test_dict.variables.gs, memories_dict = re_input_test.g, re_input_test.memories_dict

        # prune memories
        test_dict.hebb = new2stored_memories(memories_dict, pars)

        # Collect environment step data: position and observation
        position = test_dict.bptt_data.position
        xs = test_dict.inputs.xs
        # Collect model step data: cell activity (converted to numpy)
        gs_numpy = [x.numpy() for x in variables_test.g.g]
        x_gt_numpy = [x.numpy() for x in variables_test.pred.x_gt]

        # Update timeseries
        prev_cell_timeseries = [gs_timeseries, pos_timeseries, xs_timeseries, xs_gt_timeseries]
        save_data_timeseries = [gs_numpy, position, xs, x_gt_numpy]
        cell_timeseries = prepare_cell_timeseries(save_data_timeseries, prev_cell_timeseries, pars)
        gs_timeseries, pos_timeseries, xs_timeseries, xs_gt_timeseries = cell_timeseries

        ii += 1
        print(str(ii) + '/' + str(int(len(test_dict.walk_data.position[0]) / pars.seq_len)), end=' ')
        if sum(test_dict.env_steps) == 0:
            data_continue = False

    # save all final variables
    if not os.path.exists(iter_path):
        os.makedirs(iter_path)

    np.save(iter_path + '/final_variables' + str(train_i),
            m_u_.DotDict.to_dict(m_u_.tf2numpy(variables_test)), allow_pickle=True)
    np.save(iter_path + '/final_memories' + str(train_i),
            m_u_.DotDict.to_dict(m_u_.tf2numpy(memories_dict)), allow_pickle=True)

    # Save all timeseries to file
    np.save(iter_path + '/gs_timeseries_' + str(train_i), gs_timeseries)
    np.save(iter_path + '/pos_timeseries_' + str(train_i), pos_timeseries)
    np.save(iter_path + '/xs_timeseries_' + str(train_i), xs_timeseries)
    np.save(iter_path + '/xs_gt_timeseries_' + str(train_i), xs_gt_timeseries)

    # Convert test_dict, which is DotDicts, to a normal python dictionary - don't want any DotDicts remaining
    final_dict = m_u_.DotDict.to_dict(test_dict)

    # convert class params to dict
    for i, env in enumerate(final_dict['curric_env']['envs']):
        final_dict['curric_env']['envs'][i].par = m_u_.DotDict.to_dict(env.par)

    # Save final test_dict to file, which contains all environment info
    np.save(iter_path + '/final_dict_' + str(train_i), final_dict, allow_pickle=True)

    return


def summary_inferences(train_i, model_, test_step, summary_writer, pars):
    # collect time-series to do link inference summaries.
    # Needs to be separate from training loop so all envs start at same time.
    inf_opps, correct_link, inputs_test_tf = None, None, None
    test_dict = get_initial_data_dict(pars)
    data_continue, num_passes, forward_pass_time = True, 0, 0
    while data_continue:
        test_dict = data_step(test_dict, pars, test=True)
        scalings = parameters.get_scaling_parameters(train_i, pars)
        inputs_test_tf = m_u.inputs_2_tf(test_dict.inputs, test_dict.hebb, scalings)
        start_time = time.time()
        variables_test, re_input_test = test_step(model_, inputs_test_tf)
        forward_pass_time += time.time() - start_time
        re_input_test = m_u.tf2numpy(re_input_test)
        test_dict.variables.gs, memories_dict = re_input_test.g, re_input_test.memories_dict

        # prune memories
        test_dict.hebb = new2stored_memories(memories_dict, pars)

        xs = test_dict.inputs.xs
        x_gt_numpy = [x.numpy() for x in variables_test.pred.x_gt]

        acc_st, correct_link = accuracy_online(correct_link, acc_sense, xs, x_gt_numpy, pars.seq_len)
        inf_opps = inference_opportunity_online(inf_opps, test_dict.variables.inference_opportunity)

        num_passes += 1
        if sum(test_dict.env_steps) == 0:
            data_continue = False

    with summary_writer.as_default():
        # summary of average forward time
        tf.summary.scalar('extras/time_per_forward_pass', forward_pass_time / num_passes, step=train_i)
        # summary of link inferences
        tf.summary.scalar('accuracies/inferences_', 100 * np.sum(inf_opps * correct_link) / np.sum(inf_opps),
                          step=train_i)
        # summaries performance percentage pars.sum_inf_walk
        prop = [[0, 1], [1, 2], [3, 5], [5, 10], [10, 20], [40, 50], [70, 80], [80, 90], [90, 99]]
        length = np.shape(correct_link)[1]
        for (p1, p2) in prop:
            seq_pos1 = int(length * p1 / 100)
            seq_pos2 = int(length * p2 / 100)
            tf.summary.scalar('accuracies/percent {:.1f}-{:.1f}'.format(p1, p2),
                              np.mean(correct_link[:, seq_pos1:seq_pos2]) * 100, step=train_i)
    summary_writer.flush()

    return


def new2stored_memories(memories_dict_, pars):
    """
    Takes 'new' memories and puts them into 'stored' memories.
    Only keeps memories around that have non-zero weighting

    :param memories_dict_:
    :param pars:
    :return:
    """

    memories_dict = m_u.DotDict(cp.deepcopy(m_u.DotDict.to_dict(memories_dict_)))

    for b in range(pars.batch_size):

        # remove memories that were deleted -  - i.e. with zero in 'weighting' (min val is zero I hope)
        indices = np.where(memories_dict.stored.in_use[b, :] == pars.prune_mems_corr_threshold)[0]
        memories_dict.stored.x[b, :, indices] = 0.0
        memories_dict.stored.g[b, :, indices] = 0.0
        # re-order mems so that all memories to keep are at the 'front'
        idx = np.argsort(np.abs(memories_dict.stored.in_use[b, :]))[::-1]
        memories_dict.stored.x[b, :, :] = memories_dict.stored.x[b, :, idx].T
        memories_dict.stored.g[b, :, :] = memories_dict.stored.g[b, :, idx].T
        memories_dict.stored.in_use[b, :] = memories_dict.stored.in_use[b, idx]

        # Remove 'new' memories that were deleted - i.e. with zero in 'weighting' (min val is zero I hope)
        indices = np.where(memories_dict.new.in_use[b, :] != pars.prune_mems_corr_threshold)[0]
        n = len(indices)

        if n > 0:
            memories_dict.stored.x[b, :, n:] = memories_dict.stored.x[b, :, :-n]
            memories_dict.stored.x[b, :, :n] = memories_dict.new.x[b, :, indices].T

            memories_dict.stored.g[b, :, n:] = memories_dict.stored.g[b, :, :-n]
            memories_dict.stored.g[b, :, :n] = memories_dict.new.g[b, :, indices].T

            memories_dict.stored.in_use[b, n:] = memories_dict.stored.in_use[b, :-n]
            memories_dict.stored.in_use[b, :n] = memories_dict.new.in_use[b, indices]

    # return stored memories
    new_dict = m_u.DotDict({'x': memories_dict.stored.x,
                            'g': memories_dict.stored.g,
                            'in_use': memories_dict.stored.in_use,
                            })

    return new_dict
