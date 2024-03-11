#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: James Whittington
"""

import tensorflow as tf
import numpy as np
import functools

eps = 1e-8
precision = tf.float32


class DotDict(dict):
    def __init__(self, *args, **kwargs):
        # We trust the dict to init itself better than we can.
        dict.__init__(self, *args, **kwargs)
        # Because of that, we do duplicate work, but it's worth it.
        for k, v in self.items():
            self.__setitem__(k, v)

    def __getattr__(self, k):
        try:
            return dict.__getitem__(self, k)
        except KeyError:
            # Maintain consistent syntactical behaviour.
            raise AttributeError(
                "'DotDict' object has no attribute '" + str(k) + "'"
            )

    def __setitem__(self, k, v):
        dict.__setitem__(self, k, DotDict.__convert(v))

    __setattr__ = __setitem__

    def __delattr__(self, k):
        try:
            dict.__delitem__(self, k)
        except KeyError:
            raise AttributeError(
                "'DotDict' object has no attribute '" + str(k) + "'"
            )

    @staticmethod
    def __convert(o):
        """
        Recursively convert `dict` objects in `dict`, `list`, `set`, and
        `tuple` objects to `attrdict` objects.
        """
        if isinstance(o, dict):
            o = DotDict(o)
        elif isinstance(o, list):
            o = list(DotDict.__convert(v) for v in o)
        elif isinstance(o, set):
            o = set(DotDict.__convert(v) for v in o)
        elif isinstance(o, tuple):
            o = tuple(DotDict.__convert(v) for v in o)
        return o

    @staticmethod
    def to_dict(data):
        """
        Recursively transforms a dotted dictionary into a dict
        """
        if isinstance(data, dict):
            data_new = {}
            for k, v in data.items():
                data_new[k] = DotDict.to_dict(v)
            return data_new
        elif isinstance(data, list):
            return [DotDict.to_dict(i) for i in data]
        elif isinstance(data, set):
            return [DotDict.to_dict(i) for i in data]
        elif isinstance(data, tuple):
            return [DotDict.to_dict(i) for i in data]
        else:
            return data


def threshold(x, thresh_min, thresh_max, thresh_slope=0.01):
    # make this a softer threshold - i.e. shallow gradient past threshold?
    between_thresh = tf.minimum(tf.maximum(x, thresh_min), thresh_max)
    above_thresh = tf.maximum(x, thresh_max) - thresh_max
    below_thresh = tf.minimum(x, thresh_min) - thresh_min

    return between_thresh + thresh_slope * (above_thresh + below_thresh)


def combine2(mu1, mu2, sigma1, sigma2, batch_size):
    out_size = tf.shape(input=mu1)[1]
    inv_sigma_sq1 = tf.truediv(1.0, tf.square(sigma1))
    inv_sigma_sq2 = tf.truediv(1.0, tf.square(sigma2))

    logsigma = -0.5 * tf.math.log(inv_sigma_sq1 + inv_sigma_sq2)
    sigma = tf.exp(logsigma)

    mu = tf.square(sigma) * (mu1 * inv_sigma_sq1 + mu2 * inv_sigma_sq2)
    e = tf.random.normal((batch_size, out_size), mean=0, stddev=1)
    return mu + sigma * e, mu, logsigma, sigma


def squared_error(t, o, keepdims=False):
    return 0.5 * tf.reduce_sum(input_tensor=tf.square(t - o), axis=1, keepdims=keepdims)


def sparse_softmax_cross_entropy_with_logits(labels, logits):
    labels = tf.argmax(input=labels, axis=1)
    return tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=logits)


def acc_tf(real, pred):
    correct_prediction = tf.equal(tf.argmax(input=real, axis=1), tf.argmax(input=pred, axis=1))
    accuracy = tf.reduce_mean(input_tensor=tf.cast(correct_prediction, precision))
    return accuracy * 100


def tf_repeat_axis_1(tensor, repeat, dim1):
    dim0 = tf.shape(input=tensor)[0]
    return tf.reshape(tf.tile(tf.reshape(tensor, (-1, 1)), (1, repeat)), (dim0, dim1))


def inputs_2_tf(input_vars, input_hebb, scalings, batch_size=None):
    scalings_tf = {}
    for key, val in scalings.items():
        scalings_tf[key] = tf.constant(val, dtype=precision, name=key)

    # for some reason tf is not happy and we need to use DotDict and not DotMap the inputs
    inputs_dict = DotDict({'x': tf.transpose(tf.constant(input_vars.xs, dtype=precision, name='x'), (2, 0, 1)),
                           'g': tf.constant(input_vars.gs, dtype=precision, name='g'),
                           'd': tf.transpose(tf.constant(input_vars.ds, dtype=precision, name='d'), (2, 0, 1)),
                           'stored_mems': {
                               'x': tf.constant(input_hebb.x, dtype=precision, name='stored_mems_x'),
                               'g': tf.constant(input_hebb.g, dtype=precision, name='stored_mems_g'),
                               'in_use': tf.constant(input_hebb.in_use, dtype=precision, name='stored_mems_in_use'),
                           },
                           'seq_i': tf.constant(input_vars.seq_index, dtype=precision, name='seq_i'),
                           's_visited': tf.constant(input_vars.s_visited, dtype=precision, name='s_visited'),
                           'scalings': scalings_tf,
                           'positions': tf.transpose(tf.constant(input_vars.positions, dtype=precision, name='pos'),
                                                     (1, 0)),
                           })

    return inputs_dict


def compute_accuracies(xs, preds, inf_opportunity, pars):
    accuracy_g = tf.constant(0.0, dtype=precision)
    accuracy_gt = tf.constant(0.0, dtype=precision)
    inference = tf.constant(0.0, dtype=precision)

    for i in range(pars.seq_len):
        accuracy_g += acc_tf(xs[i], preds.x_g[i])  # acc of generated
        accuracy_gt += acc_tf(xs[i], preds.x_gt[i])  # acc of generated

        corr = tf.cast(tf.equal(tf.argmax(input=xs[i], axis=1), tf.argmax(input=preds.x_gt[i], axis=1)), precision)
        inference += tf.reduce_sum(corr * inf_opportunity[:, i])

    num_inf_ops = tf.cast(tf.reduce_sum(inf_opportunity), dtype=precision) + eps
    inference = 100.0 * inference / num_inf_ops
    accuracies = DotDict({'g': accuracy_g.numpy() / pars.seq_len,
                          'gt': accuracy_gt.numpy() / pars.seq_len,
                          'diff': (accuracy_g.numpy() - accuracy_gt.numpy()) / pars.seq_len,
                          'inferences': inference.numpy(),
                          })

    return accuracies


def make_summaries(losses, accuracies, scalings, variables, curric_env, env_steps, trainable_variables, pars):
    summaries = {}
    for key, val in losses.items():
        summaries['losses/' + key] = val.numpy()

    for key, val in accuracies.items():
        summaries['accuracies/' + key] = val

    for key, val in scalings.items():
        summaries['scalings/' + key] = val

    for variable in trainable_variables:
        key = variable.name
        val = variable.value()
        mean = tf.reduce_mean(val).numpy()
        sq = tf.reduce_mean(val ** 2).numpy()
        summaries['weights/' + key + '_mean'] = mean
        summaries['weights/' + key + '_sq'] = sq
        summaries['weights/' + key + '_std'] = (sq - mean ** 2) ** 0.5

    var_dict = {'g': variables.g.g,
                'g_gen': variables.g.g_gen,
                'g_mem_input': variables.g.g_mem_input,
                'delta': variables.p2g.delta,
                'p2g': variables.p2g.p2g}
    for key, var in var_dict.items():
        # we can look at mean, sq, std across population vector or across time/space. Averaging over batch for both
        # across population
        e_x = [tf.reduce_mean(x, axis=1) for x in var]
        e_x2 = [tf.reduce_mean(x ** 2, axis=1) for x in var]
        std = [(b - a ** 2) ** 0.5 for a, b in zip(e_x, e_x2)]
        summaries['vars/pop_' + key + '_mu'] = tf.reduce_mean(tf.add_n(e_x)).numpy() / pars.seq_len
        summaries['vars/pop_' + key + '_sq'] = tf.reduce_mean(tf.add_n(e_x2)).numpy() / pars.seq_len
        summaries['vars/pop_' + key + '_std'] = tf.reduce_mean(tf.add_n(std)).numpy() / pars.seq_len
        # across time-steps
        e_x = tf.add_n(var) / pars.seq_len
        e_x2 = tf.add_n([v ** 2 for v in var]) / pars.seq_len
        std = (e_x2 - e_x ** 2) ** 0.5
        summaries['vars/time_' + key + '_mu'] = tf.reduce_mean(e_x).numpy()
        summaries['vars/time_' + key + '_sq'] = tf.reduce_mean(e_x2).numpy()
        summaries['vars/time_' + key + '_std'] = tf.reduce_mean(std).numpy()
    """
    summaries['vars/g_mu'] = tf.reduce_mean(tf.add_n(variables.g.g)).numpy() / pars.seq_len
    summaries['vars/g_gen_mu'] = tf.reduce_mean(tf.add_n(variables.g.g_gen)).numpy() / pars.seq_len
    summaries['vars/g_mem_input_mu'] = tf.reduce_mean(tf.add_n(variables.g.g_mem_input)).numpy() / pars.seq_len

    summaries['vars/g_sq'] = tf.reduce_mean(tf.add_n([x ** 2 for x in variables.g.g])).numpy() / pars.seq_len
    summaries['vars/g_gen_sq'] = tf.reduce_mean(tf.add_n([x ** 2 for x in variables.g.g_gen])).numpy() / pars.seq_len
    summaries['vars/g_mem_input_sq'] = tf.reduce_mean(
        tf.add_n([x ** 2 for x in variables.g.g_mem_input])).numpy() / pars.seq_len

    summaries['vars/g_std'] = (summaries['vars/g_sq'] - summaries['vars/g_mu'] ** 2) ** 0.5
    summaries['vars/g_gen_std'] = (summaries['vars/g_gen_sq'] - summaries['vars/g_gen_mu'] ** 2) ** 0.5
    summaries['vars/g_mem_input_std'] = (summaries['vars/g_mem_input_sq'] - summaries[
        'vars/g_mem_input_mu'] ** 2) ** 0.5
    """

    a = variables['mem_dist']['inner_prods'][-1].numpy()
    a[a < -50] = np.nan
    a[a == 0] = np.nan
    summaries['vars/inner_prod_mu'] = np.nanmean(a.flatten())
    summaries['vars/inner_prod_std'] = np.nanstd(a.flatten())

    summaries['vars/x_g_mu'] = tf.reduce_mean(tf.add_n(variables.pred.x_g)).numpy() / pars.seq_len
    summaries['vars/x_gt_mu'] = tf.reduce_mean(tf.add_n(variables.pred.x_gt)).numpy() / pars.seq_len
    summaries['vars/logits_g_mu'] = tf.reduce_mean(tf.add_n(variables.logits.x_g)).numpy() / pars.seq_len
    summaries['vars/logits_gt_mu'] = tf.reduce_mean(tf.add_n(variables.logits.x_gt)).numpy() / pars.seq_len

    summaries['extras/new_envs'] = sum(env_steps == 0)
    summaries['extras/min_walk_length'] = np.min(curric_env.n_restart)
    summaries['extras/av_walk_length'] = np.mean(curric_env.walk_len)
    summaries['extras/model_step_time'] = curric_env.model_step_time
    summaries['extras/data_step_time'] = curric_env.data_step_time

    return summaries


def define_scope(func):
    """Creates a name_scope that contains all ops created by the function.
    The scope will default to the provided name or to the name of the function
    in CamelCase. If the function is a class constructor, it will default to
    the class name. It can also be specified with name='Name' at call time.

    Is helpful for debugging!
    """

    name_func = func.__name__
    if name_func == '__init__':
        name_func = func.__class__.__name__
    name_func = camel_case(name_func)

    @functools.wraps(func)
    def _wrapper(*args, **kwargs):
        # Local, mutable copy of `name`.
        name_to_use = name_func

        with tf.name_scope(name_to_use):
            return func(*args, **kwargs)

    return _wrapper


def camel_case(name):
    """Converts the given name in snake_case or lowerCamelCase to CamelCase."""
    words = name.split('_')
    return ''.join(word.capitalize() for word in words)


def get_all_keys(value, key_=None):
    """
    Build list of keys for a nested dictionary, so that each value has its own list of nested keys
    """
    key_ = [] if key_ is None else key_
    if not (isinstance(value, dict) or isinstance(value, DotDict)):
        return [[]]
    return [[key] + path for key, val in value.items() for path in get_all_keys(val, key_)]


def nested_set(dic, keys, value):
    """
    Set value of dictionary for a list of nested keys
    """
    for key in keys[:-1]:
        dic = dic.setdefault(key, {})
    dic[keys[-1]] = value


def nested_get(dic, keys):
    """
    Get value of dictionary for a list of nested keys
    """
    for key in keys:
        dic = dic[key]
    return dic


def tf2numpy(d):
    if isinstance(d, dict):
        new_dict = DotDict()
        for k, v in d.items():
            new_dict[k] = tf2numpy(v)
        return new_dict
    elif isinstance(d, list):
        return [tf2numpy(x) for x in d]
    elif isinstance(d, tf.Tensor):
        return d.numpy()
    else:
        return d


def copy_tensor(x):
    if isinstance(x, tf.Tensor):
        return tf.identity(x)
    elif isinstance(x, DotDict):
        return DotDict({key: copy_tensor(value) for key, value in x.items()})
    elif isinstance(x, dict):
        return {key: copy_tensor(value) for key, value in x.items()}
    elif isinstance(x, list):
        return [copy_tensor(y) for y in x]
    else:
        raise ValueError('unsopported type: ' + str(type(x)))


def nested_isnan_inf(x, key=None):
    if isinstance(x, np.ndarray) or isinstance(x, int) or isinstance(x, float) or isinstance(x, np.int64) or isinstance(
            x, np.int32) or isinstance(x, np.float32):
        if np.isnan(x).any() or np.isinf(x).any():
            return True
        else:
            return False
    elif isinstance(x, tf.Tensor) or isinstance(x, tf.Variable):
        return nested_isnan_inf(x.numpy(), key)
    elif isinstance(x, DotDict) or isinstance(x, dict):
        return {key: nested_isnan_inf(value, key) for key, value in x.items()}
    elif isinstance(x, list):
        return [nested_isnan_inf(y, key) for y in x]
    elif x is None:
        return False
    else:
        raise ValueError('unsopported type: ' + str(type(x)), key)


def is_any_nan_inf(x, current=False):
    if isinstance(x, bool):
        if current:
            return current
        else:
            return x
    elif isinstance(x, DotDict) or isinstance(x, dict):
        return np.asarray([is_any_nan_inf(value) for value in x.values()]).any()
    elif isinstance(x, list):
        return np.asarray([is_any_nan_inf(y) for y in x]).any()
    elif x is None:
        return False
    else:
        raise ValueError('unsopported type: ' + str(type(x)))


def check_inputs_modified(x, y):
    if isinstance(x, tf.Tensor):
        print(tf.reduce_sum((x - y) ** 2))
    elif isinstance(x, dict) or isinstance(x, DotDict):
        for key, value in x.items():
            check_inputs_modified(value, y[key])
    elif isinstance(x, list):
        for i, a in enumerate(x):
            check_inputs_modified(a, y[i])
    else:
        print(type(x), type(y))
    return
