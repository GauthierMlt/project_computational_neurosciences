#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: James Whittington
"""

import numpy as np
import environments
import os
import gzip
from model_utils import DotDict as Dd
from itertools import cycle, islice


def default_params(width=None, height=None, world_type=None, batch_size=None):
    params = Dd()

    params.graph_mode = True  # graph mode or eager mode - set False for debugging
    params.tf_range = True  # tf_range=True compiles graph quicker but is slower when running.
    params.batch_size = 16 if not batch_size else batch_size
    params.seq_len = 300  # 75  # 50.   we truncate BPTT to sequences of this length

    """
    ----------------------------------------------------------------
    ENVIRONMENT / DATA / SAVE / SUMMARY
    ----------------------------------------------------------------
    """
    # 'rectangle', 'hexagonal', 'family_tree', 'line_ti', 'wood2000', 'frank2000', 'grieves2016', 'sun2020', 'nieh2021'
    params.world_type = 'rectangle' if not world_type else world_type
    params.n_envs = params.batch_size
    params.s_size = 45
    params = get_env_params(params, width, height=height)
    params.use_reward = True

    # only save date from first X of batch
    params.n_envs_save = 16
    params.summary_inference = False
    # num gradient updates between summaries
    params.sum_int = 5
    # num gradient updates between detailed accuracy summaries
    params.sum_int_inferences = 400
    # number of gradient steps between saving data
    params.save_interval = int(50000 / params.seq_len)
    # number of gradient steps between saving model
    params.save_model = 5 * params.save_interval

    """
    ----------------------------------------------------------------
    MODEL
    ----------------------------------------------------------------
    """
    params.rnn_type = 'TEM'  # 'CANN'  'TEM'    ?? need more neurons for CANN ??
    params.infer_g_type = 'g_mem'  # 'g'
    params.project_x = False
    params.s_size_project = 20 if params.project_x else params.s_size

    # numbers of variables for each frequency
    params.g_size = 128
    params.grid2phase = 1.0
    params.phase_size = int(params.g_size / params.grid2phase)
    params.s_size_hidden = 20 * params.s_size_project

    # initialisations
    params.g_init = 0.5
    params.p2g_init = 0.1

    # position encoding
    params.g_norm = 'none'  # 'layer_norm','none' --> keep this as none!!
    params.g_act = 'none'  # 'relu', 'leaky_relu', 'none', 'tanh' - KEEP THIS AS LEAKY RELU AND JUST CHANGE ALPHA
    params.leaky_relu_alpha = 0.01  # 1.0
    params.g_act_after_inference = True
    if params.rnn_type == 'CANN':
        params.g_act = 'tanh'
    params.g_thresh_max = 10.0
    params.g_thresh_min = -10.0
    params.d_mixed_size = 15

    # keys/queries
    params.g_projection_init = 'orthogonal'  # 'glorot_uniform' 'orthogonal' 'identity'
    params.g_projection_learn = True  # old 'subsection' is same as identity ans no learning
    params.g2p_norm = 'layer_norm'  # 'layer_norm', 'none'
    params.norm_before_downsample = True
    params.g2p_thresh_max = 5.0
    params.g2p_thresh_min = -5.0

    # memory retrieval
    params.memory_order = {'gen': {'in': 'g',
                                   'out': 'x'
                                   },
                           'inf': {'in': 'x',
                                   'out': 'g'},
                           }
    params.similarity_measure = 'dot_product'
    params.inf_mem_use_g = True
    params.inf_mem_helper = True
    params.extra_mem_inf_type = 'multiplicative'  # 'additive', 'multiplicative'
    params.kernel_norm = 'sqrt'
    params.kernel_thresh_max = 2.0
    params.kernel_thresh_min = -2.0
    params.softmax_target = 0.9
    params.softmax_beta = 0.4  # 0.1 a low value here leads to inf model hacking...
    # need to make sure the inner prods of self attention are not too high initially

    # pruning memories
    params.prune_mems_corr_threshold = 0.0
    params.hebb_mems_keep = int(50 * np.ceil(np.maximum(params.max_states, params.seq_len) / 50))  # to nearest 50
    params.key_mem_thresh = 0.7
    params.perfect_memory_add_remove = False
    params.overwrite_bptt_mems = True
    """
    ----------------------------------------------------------------
    TRAINING
    ----------------------------------------------------------------
    """

    params.train_iters = 50000  # 2000000  # just a large number - does not require this many steps
    params.train_on_visited_states_only = True  # not necessary
    params.learning_rate_max = 4.5e-4
    params.learning_rate_min = 2e-4

    # losses
    # 'lx_g', 'lx_gt', 'lg', 'lg_reg', 'weight_reg' 'keys_non_neg', 'lg_non_neg', 'keys_non_neg',
    # 'orthog_transformer', 'keys_l2'
    params.which_costs = ['lx_g', 'lx_gt', 'lg', 'weight_reg', 'lg_reg', 'lg_non_neg']
    if not params.g_projection_learn or not params.norm_before_downsample:
        params.which_costs = [x for x in params.which_costs if
                              x not in ['orthog_transformer', 'keys_l2', 'keys_non_neg']]
    if params.g_norm == 'layer_norm':
        params.which_costs = [x for x in params.which_costs if x not in ['lg_reg', 'lg_non_neg']]

    # regularisation values
    params.lg_val = 1.0
    params.lx_gt_val = 1.0 if 'lx_g' in params.which_costs else 2.0
    params.g_reg_pen = 1e-3  # * 256.0 / params.g_size
    params.lg_temp = 1.0  # * 256.0 / params.g_size
    params.weight_reg_val = 1e-7
    params.g_non_neg_pen = 16 * 4e-3  # * 256.0 / params.g_size
    params.keys_sparsity_pen = 2e-4
    params.keys_l2_pen = 2e-3
    params.orthog_transformer_reg_val = 1e-3

    # Number gradient updates for annealing (in number of gradient updates)
    # they should be >= 1
    params.temp_it = 2000  # 2000  # 2000  # 400   # CANNOT BE 0
    params.l_r_decay_steps = 4000
    params.l_r_decay_rate = 0.5  # 0.5
    params.p2g_start = -5  # 100  # iteration p2g kicks in
    params.p2g_warmup = 1  # 200
    params.g_gt_it = 20000000000  # 2000
    params.g_gt_bias = 0.0  # between -1 and 1
    params.g_reg_it = 20000000000

    if params.g_act in ['leaky_relu', 'relu']:
        params.temp_it = 2000
        # params.keys_l2_pen = 2e-3
        # params.lg_val = 1.0
        # params.g_reg_pen = 1e-4

    return params


def get_env_params(par, width, height):
    if par.world_type == 'rectangle':
        par_env = Dd({'stay_still': True,
                      'bias_type': 'angle',
                      'direc_bias': 0.25,
                      'angle_bias_change': 0.4,
                      'restart_max': 40,
                      'restart_min': 5,
                      'seq_jitter': 30,
                      'save_walk': 30,
                      'sum_inf_walk': 30,
                      'widths': [10, 10, 11, 11, 8, 9, 10, 11, 8, 9, 10, 11, 8, 8, 9, 9] if not width else
                      [width] * par.batch_size,
                      'heights': [10, 10, 11, 11, 8, 9, 10, 11, 8, 9, 10, 11, 8, 8, 9, 9] if not height else
                      [height] * par.batch_size,
                      'rels': ['down', 'up', 'left', 'right', 'stay still'],
                      })
        size_increase = 6
        par_env.widths = [width + size_increase for width in par_env.widths]
        par_env.heights = [height + size_increase for height in par_env.heights]
        n_states = [environments.Rectangle.get_n_states(width, height) for width, height in
                    zip(par_env.widths, par_env.heights)]

    elif par.world_type == 'hexagonal':
        par_env = Dd({'stay_still': True,
                      'bias_type': 'angle',
                      'direc_bias': 0.2,
                      'angle_bias_change': 0.4,
                      'hex_boundary': True,
                      'restart_max': 40,
                      'restart_min': 5,
                      'seq_jitter': 30,
                      'save_walk': 30,
                      'sum_inf_walk': 30,
                      'widths': [6, 6, 7, 7, 5, 5, 6, 7, 5, 6, 6, 7, 5, 5, 6, 6] if not width else
                      [width] * par.batch_size,
                      'rels': ['down left', 'down right', 'up left', 'up right', 'left', 'right', 'stay still'],
                      })
        par_env.widths = [2 * x - 1 for x in par_env.widths]
        n_states = [environments.Hexagonal.get_n_states(width)[1] for width in par_env.widths]

    elif par.world_type == 'family_tree':
        par_env = Dd({'restart_max': 30,
                      'restart_min': 10,
                      'seq_jitter': 10,
                      'save_walk': 30,
                      'sum_inf_walk': 30,
                      'widths': [4, 4, 5, 5, 3, 3, 4, 4, 3, 5, 5, 4, 3, 4, 3, 5] if not width else
                      [width] * par.batch_size,
                      'rels': ['parent', 'child 1', 'child 2', 'sibling', 'grand parent', 'uncle/aunt',
                               'niece/nephew 1', 'niece/nephew 2', 'cousin 1', 'cousin 2'],
                      })
        n_states = [environments.FamilyTree.get_n_states(width) for width in par_env.widths]

    elif par.world_type == 'line_ti':
        par_env = Dd({'jump_length': 4,
                      'restart_max': 30,
                      'restart_min': 10,
                      'seq_jitter': 10,
                      'save_walk': 30,
                      'sum_inf_walk': 30,
                      'widths': [5, 5, 6, 6, 4, 4, 5, 6, 4, 4, 5, 6, 4, 5, 6, 5] if not width else
                      [width] * par.batch_size,
                      })

        par_env.rels = [str(x) for x in range(-par_env.jump_length, par_env.jump_length + 1)]
        n_states = [environments.LineTI.get_n_states(width) for width in par_env.widths]

    elif par.world_type == 'wood2000':
        par_env = Dd({'error_prob': 0.15,
                      'restart_max': 40,
                      'restart_min': 10,
                      'seq_jitter': 30,
                      'save_walk': 40,
                      'sum_inf_walk': 40,
                      'heights': [6 if width is None else height] * par.batch_size,
                      'widths': [4 if width is None else width] * par.batch_size,
                      'rels': ['down', 'up', 'left', 'right'],
                      })
        n_states = [environments.Wood2000.get_n_states(width, height) for width, height in
                    zip(par_env.widths, par_env.heights)]

    elif par.world_type == 'frank2000':
        par_env = Dd({'error_prob': 0.1,
                      'restart_max': 40,
                      'restart_min': 10,
                      'seq_jitter': 30,
                      'save_walk': 40,
                      'sum_inf_walk': 40,
                      'heights': [5 if width is None else height] * par.batch_size,
                      'widths': [1 if width is None else width] * par.batch_size,
                      'rels': ['down', 'up', 'left', 'right'],
                      })
        n_states = [environments.Frank2000.get_n_states(width, height)[0] for width, height in
                    zip(par_env.widths, par_env.heights)]

    elif par.world_type == 'grieves2016':
        par_env = Dd({'error_prob': 0.1,
                      'switch_prob': 0.2,
                      'exploration_bias': 0.80,
                      'restart_max': 30,
                      'restart_min': 10,
                      'seq_jitter': 30,
                      'save_walk': 30,
                      'sum_inf_walk': 30,
                      'widths': [4 if width is None else width] * par.batch_size,
                      'simplified': True,
                      'rels': ['down', 'up', 'up-left', 'down-right', 'down-left', 'up-right', 'reward', 'no-reward',
                               'try reward']
                      })
        par_env.rels += ['0', '1', '2', '3'] if par_env.simplified else []
        n_states = [environments.Grieves2016.get_n_states(width)[0] for width in par_env.widths]

    elif par.world_type == 'sun2020':
        par_env = Dd({'restart_max': 40,
                      'restart_min': 10,
                      'seq_jitter': 30,
                      'save_walk': 40,
                      'sum_inf_walk': 40,
                      'widths': [3 if width is None else width] * par.batch_size,
                      'rels': ['down', 'up', 'left', 'right'],
                      'n_laps': 4,
                      })
        n_states = [environments.Sun2020.get_n_states(width, par_env.n_laps) for width in par_env.widths]

    elif par.world_type == 'nieh2021':
        par_env = Dd({'error_beta': 0.5,
                      'bias': 4.5,
                      'restart_max': 10,
                      'restart_min': 2,
                      'seq_jitter': 4,
                      'save_walk': 10,
                      'sum_inf_walk': 10,
                      'widths': [8, 8, 10, 10, 12, 12, 6, 6, 12, 12, 10, 10, 8, 8, 6, 6] if not width else
                      [width] * par.batch_size,
                      'rels': ['proceed', 'pillar_left', 'pillar_right', 'left', 'right'],
                      })
        n_states = [environments.Nieh2021.get_n_states(width) for width in par_env.widths]

    else:
        raise ValueError('incorrect world specified')

    # repeat widths and height
    par_env.widths = list(islice(cycle(par_env.widths), par.batch_size))
    par_env.heights = list(islice(cycle(par_env.heights), par.batch_size))

    par.max_states = np.max(n_states)
    par_env.n_actions = len(par_env.rels) if 'stay still' not in par_env.rels else len(par_env.rels) - 1
    par.n_actions = par_env.n_actions
    par.env = par_env
    return par


def get_scaling_parameters(index, par):
    # these scale with number of gradient updates
    temp = np.maximum(np.minimum((index + 1) / par.temp_it, 1.0), 0.0)
    l_r = (par.learning_rate_max - par.learning_rate_min) * (par.l_r_decay_rate ** (
            index / par.l_r_decay_steps)) + par.learning_rate_min
    l_r = np.maximum(l_r, par.learning_rate_min)
    p2g_scale = 0.0 if index <= par.p2g_start else np.minimum((index - par.p2g_start) / par.p2g_warmup, 1.0)
    g_gt = par.g_gt_bias + np.maximum(np.minimum((index + 1) / par.g_gt_it, 1.0), -1.0)
    g_cell_reg = 1 - np.minimum((index + 1) / par.g_reg_it, 1.0)

    scalings = Dd({'temp': temp,
                   'l_r': l_r,
                   'iteration': index,
                   'p2g_scale': p2g_scale,
                   'g_gt': g_gt,
                   'g_cell_reg': g_cell_reg,
                   })

    return scalings


def old2new(world_type):
    old2new_name_convert = Dd({'hex': 'hexagonal',
                               'splitter': 'wood2000',
                               'in_out_bound': 'frank2000',
                               'splitter_grieves': 'grieves2016',
                               'loop_laps': 'sun2020',
                               'tank': 'nieh2021',
                               'rectangle': 'rectangle',
                               'square': 'rectangle',
                               'hexagonal': 'hexagonal',
                               'wood2000': 'wood2000',
                               'frank2000': 'frank2000',
                               'grieves2016': 'grieves2016',
                               'sun2020': 'sun2020',
                               'nieh2021': 'nieh2021',
                               'yves': 'yves'
                               })
    try:
        return old2new_name_convert[world_type]
    except KeyError:
        return world_type


def load_params_wrapper(save_dir, date, run):
    try:
        saved_path = save_dir + date + '/run' + str(run) + '/save'
        pars = load_params(saved_path)
    except FileNotFoundError:
        saved_path = save_dir + date + '/save/' + 'run' + str(run)
        pars = load_params(saved_path)
    return pars


def load_params(saved_path):
    try:
        pars = load_numpy_gz(saved_path + '/params.npy')
    except FileNotFoundError:
        pars = load_numpy_gz(saved_path + '/pars.npy')

    return pars


def load_numpy_gz(file_name):
    try:
        return np.load(file_name, allow_pickle=True)
    except FileNotFoundError:
        f = gzip.GzipFile(file_name + '.gz', "r")
        return np.load(f, allow_pickle=True)


def get_params(save_dirs, date, run, not_this_dir=False, print_where=True):
    savedir1, params_1 = None, None
    if type(save_dirs) != list:
        save_dirs = [save_dirs]
    for save_dir in save_dirs:
        savedir1 = save_dir + date + '/run' + str(run)
        if savedir1 != not_this_dir:
            try:
                params_1 = load_params_wrapper(save_dir, date, run).item()
                if print_where:
                    print('params yes: ' + savedir1)
                break
            except FileNotFoundError:
                print('params not: ' + savedir1)
                pass
        else:
            print('params not: ' + savedir1)

    if print_where:
        print('')

    if params_1 is None:
        raise ValueError('NO PARAMS FOUND: ' + savedir1)
    else:
        saved_dir_1 = savedir1[:]

        return Dd(params_1), saved_dir_1


def compare_params(params_1, params_2):
    messages = []
    for p1 in params_1:
        message = compare_param(params_1, params_2, p1, which=1)
        if type(message) == list:
            for m in message:
                messages.append(m)
        else:
            messages.append(message)

    for p2 in params_2:
        message = compare_param(params_1, params_2, p2, which=2)
        if type(message) == list:
            for m in message:
                messages.append(m)
        else:
            messages.append(message)

    messages = list(set(messages))

    return messages


def compare_param(params_1, params_2, param, which=1):
    try:
        p1 = params_1[param]
        p2 = params_2[param]
    except KeyError:
        if which == 1:
            message = str(['missing param 2 : ', param, params_1[param]])
        else:
            message = str(['missing param 1 : ', param, params_2[param]])
        return message

    different = str(['different : ', param, p1, p2])
    same = str(['same : ', param, p1, p2])
    same_too_big = str(['same : ', param, ' too big to show'])

    if not isinstance(p1, type(p2)):
        message = different
    elif type(p1) == dict or type(p2) == Dd:
        message = compare_params(params_1[param], params_2[param])
    elif type(p1) == list:
        if sorted(p1) == sorted(p2):
            message = same
        else:
            message = different
    elif type(p1) == np.ndarray:
        if np.array_equal(p1, p2):
            if np.max(np.shape(p1)) > 10:
                message = same_too_big
            else:
                message = same
        else:
            message = different
    else:
        if p1 == p2:
            message = same
        else:
            message = different

    return message


def find_model_with_params(save_dirs, keys, vals_desired):
    key = None
    for dirs in save_dirs:
        list_of_save_paths = [x[0] for x in os.walk(dirs) if
                              'save' in x[0][-20:] and 'run' in x[0] and 'iter' not in x[0]]
        list_of_save_paths.sort()
        for s_p in sorted(list_of_save_paths, reverse=True):
            print_bool = True
            try:
                pars = load_params(s_p)
                pars = pars.item()

                for key, val_desired in zip(keys, vals_desired):
                    if val_desired:
                        if pars[key] != val_desired:
                            print_bool = False
                    else:
                        # print(str([pars[key] for key in keys]) + ': ' + s_p)
                        pass
                if print_bool:
                    print(s_p)
                    print(str([pars[key] for key in keys]))

            except FileNotFoundError:
                print('file not found: ' + s_p)
            except KeyError:
                print(key, 'Key Error: ' + s_p)
    return
