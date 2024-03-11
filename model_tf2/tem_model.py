#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: James Whittington
"""

import model_utils as m_u
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense

eps = m_u.eps


class TEM(tf.keras.Model):
    def __init__(self, par):
        super(TEM, self).__init__()

        self.par = par
        self.precision = tf.float32 if 'precision' not in self.par else self.par.precision
        self.batch_size = self.par.batch_size
        self.scalings = None  # JW: probs need to change this
        self.seq_pos = tf.zeros(self.batch_size, dtype=self.precision, name='seq_pos')
        self.softmax_beta_0 = np.log(self.par.softmax_target / (1.0 - self.par.softmax_target)) * self.par.softmax_beta

        # Create trainable parameters
        glorot_uniform = tf.keras.initializers.GlorotUniform()
        init_p2g = tf.initializers.TruncatedNormal(stddev=self.par.p2g_init)
        init_g = tf.initializers.TruncatedNormal(stddev=self.par.g_init)
        # if self.par.g_act in ['leaky_relu', 'relu']:
        #    init_g = tf.keras.initializers.RandomUniform(minval=0, maxval=2 * self.par.g_init)

        # entorhinal layer norm
        if self.par.g_norm == 'layer_norm':
            self.layernorm_g = tf.keras.layers.LayerNormalization(epsilon=1e-6, axis=1, trainable=False,
                                                                  name='layer_norm_g')

        # g2p norm
        if self.par.g2p_norm == 'layer_norm':
            self.layernorm_g2p = tf.keras.layers.LayerNormalization(epsilon=1e-6, axis=1, trainable=False,
                                                                    name='layer_norm_g2p')
        elif self.par.g2p_norm == 'unit_norm':
            # JW: perhaps make this non-negative??
            self.unitnorm_scale_g2p = tf.Variable(1.0, dtype=self.precision, trainable=False, name='g2p_unit_norm')

        # g_prior mu
        self.g_prior_mu = tf.Variable(init_g(shape=(1, self.par.g_size), dtype=self.precision), trainable=True,
                                      name='g_prior_mu')
        self.g_init = None

        # MLP for transition weights
        if self.par.rnn_type == 'TEM':
            self.t_vec = tf.keras.Sequential([Dense(self.par.d_mixed_size, input_shape=(self.par.n_actions,),
                                                    activation=tf.tanh, kernel_initializer=glorot_uniform,
                                                    name='t_vec_1', use_bias=False),
                                              Dense(self.par.g_size ** 2, use_bias=False,
                                                    kernel_initializer=tf.zeros_initializer, name='t_vec_2')])
        elif self.par.rnn_type == 'CANN':
            if self.par.world_type == 'rectangle':
                self.d_index2space = tf.Variable([[0, -1], [0, 1], [-1, 0], [1, 0]], dtype=self.precision,
                                                 trainable=False, name='d_index2space')
            elif self.par.world_type == 'hexagonal':
                self.d_index2space = tf.Variable(
                    [[-0.5, -(3 / 4) ** 0.5], [0.5, -(3 / 4) ** 0.5], [-0.5, (3 / 4) ** 0.5], [0.5, (3 / 4) ** 0.5],
                     [-1, 0], [0, 1]], dtype=self.precision, trainable=False, name='d_index2space')
            else:
                raise ValueError('CANN not implemented for ' + self.par.world_type + ' world')

            self.cann_B = tf.keras.Sequential(
                Dense(self.par.g_size, use_bias=True, kernel_initializer=tf.zeros_initializer, name='CANN_B'))
            self.cann_W = tf.keras.Sequential(
                Dense(self.par.g_size, use_bias=False, kernel_initializer=tf.zeros_initializer, name='CANN_W'))
        else:
            raise ValueError(self.par.rnn_type + ' rnn variant not implemented yet')

        if self.par.g_projection_init == 'identity':
            g_projection_init = tf.keras.initializers.Identity(gain=1.0)
        elif self.par.g_projection_init == 'orthogonal':
            g_projection_init = tf.keras.initializers.Orthogonal(gain=1.0, seed=None)
        elif self.par.g_projection_init == 'glorot_uniform':
            g_projection_init = tf.keras.initializers.GlorotUniform()
        else:
            raise ValueError('unknown g_projection init')
        self.projection_g = tf.keras.Sequential(
            Dense(self.par.phase_size, input_shape=(self.par.g_size,), name='projection_g',
                  kernel_initializer=g_projection_init, trainable=self.par.g_projection_learn,
                  use_bias=False))

        if self.par.project_x:
            self.projection_x = tf.keras.Sequential(
                Dense(self.par.s_size_project, input_shape=(self.par.s_size,), name='projection_x'))
            raise ValueError('Not implemented x projection yet')

        # p2g
        if 'mem' in self.par.infer_g_type:
            self.p2g_mu = tf.keras.Sequential(
                [Dense(2 * self.par.g_size, input_shape=(self.par.phase_size,), activation=tf.nn.elu, name='p2g_mu_1',
                       kernel_initializer=glorot_uniform),
                 Dense(self.par.g_size, name='p2g_mu_2', kernel_initializer=init_p2g)])

            self.p2g_delta = tf.keras.Sequential(
                [Dense(10 * self.par.g_size, input_shape=(2 * self.par.g_size + 2,), activation=tf.nn.elu,
                       kernel_initializer=glorot_uniform, name='p2g_alpha_1'),
                 Dense(self.par.g_size, kernel_initializer=glorot_uniform, activation=None, name='p2g_alpha_2')])

            if self.par.extra_mem_inf_type == 'additive':
                self.alpha_g_x = tf.Variable(0.0, dtype=self.precision, trainable=False, name='alpha_g_x')

        # predict x MLP
        self.MLP_x_pred = tf.keras.Sequential([Dense(self.par.s_size_hidden, input_shape=(self.par.s_size_project,),
                                                     activation=tf.nn.elu, kernel_initializer=glorot_uniform,
                                                     name='MLP_c_star_1'),
                                               Dense(self.par.s_size, kernel_initializer=glorot_uniform,
                                                     name='MLP_c_star_2')])

    @m_u.define_scope
    def call(self, inputs, training=None, mask=None):

        # inputs = m_u.copy_tensor(inputs_)
        # Setup member variables and get dictionaries from input
        memories_dict, variable_dict = self.init_input(inputs)

        # Precompute transitions
        ta_mat = self.precomp_trans(inputs.d)

        # book-keeping
        g_t = inputs.g
        for i in tf.range(self.par.seq_len, name='iteration') if self.par.tf_range else range(self.par.seq_len):
            # tf.range turns everything into tensors. Be careful with that! E.g. in mem_step use 'gen', 'inf'
            # tf.range (and tf in general) is slow with conditionals. Don't use where possible
            # using and appending to lists is slow with tf.range
            # tf.range version slower (+30%) than range version, though faster compilation (1000s to 80s for bptt=75)
            # tf.range version uses much less RAM

            # single step
            g_t, variable_dict, memories_dict = self.step(inputs, g_t, variable_dict, memories_dict, i, ta_mat.read(i))

        # convert tensorarray to list
        variable_dict = self.tensorarray_2_list(variable_dict)

        # Collate g for re-input to model
        re_input_dict = {'memories_dict': memories_dict,
                         'g': g_t,
                         }

        return variable_dict, re_input_dict

    # WRAPPER FUNCTIONS

    @m_u.define_scope
    def step(self, inputs, g_t, variable_dict, memories_dict, i, t_mat, mem_offset=0):
        # with tf.range and in graph mode, can't make the 'i' variable a global. So pass seq_pos, and i
        seq_pos = inputs.seq_i * self.par.seq_len + tf.cast(i, self.precision)

        # generative transition
        g_gen, g2g = self.gen_g(g_t, t_mat, seq_pos)

        # infer hippocampus (p) and entorhinal (g)
        mem_step = self.mem_step(memories_dict, i + mem_offset)
        g, hidden_x2g, delta_p2g, mu_p2g = self.inference(g2g, inputs.x[i], mem_step)

        # generate sensory
        x_all, x_logits_all, retrieved_g2x, g_mem_input, hidden_g2x = self.generation(g, g_gen, mem_step)

        # Hebbian update - equivalent to the matrix updates, but implemented differently for computational ease
        # x, g, retrieved_g2x, x_hat, h_g2x, h_x2g, mems, mem_i
        memories_dict = self.hebbian(inputs.x[i], g_mem_input, hidden_g2x, hidden_x2g, memories_dict, i + mem_offset,
                                     inputs.s_visited[:, i])

        # Collate all variables for losses and saving representations
        var_updates = [[['g', 'g'], g],
                       [['g', 'g_gen'], g_gen],
                       [['g', 'g_mem_input'], g_mem_input],
                       [['p2g', 'delta'], delta_p2g],
                       [['p2g', 'p2g'], mu_p2g],
                       [['pred', 'x_g'], x_all['x_g']],
                       [['pred', 'x_gt'], x_all['x_gt']],
                       [['logits', 'x_g'], x_logits_all['x_g']],
                       [['logits', 'x_gt'], x_logits_all['x_gt']],
                       [['mem_dist', 'probabilities'],
                        tf.concat([hidden_g2x['stored']['prob'], tf.concat([hidden_g2x['new']['prob'], tf.zeros(
                            (self.par.batch_size, self.par.seq_len - i + mem_offset))], axis=1)], axis=1)],
                       [['mem_dist', 'inner_prods'],
                        tf.concat([hidden_g2x['stored']['scal'], tf.concat([hidden_g2x['new']['scal'], tf.zeros(
                            (self.par.batch_size, self.par.seq_len - i + mem_offset))], axis=1)], axis=1)]
                       ]

        # And write all variables to tensorarrays
        variable_dict = self.update_vars(variable_dict, var_updates, i)

        return g, variable_dict, memories_dict

    @m_u.define_scope
    def inference(self, g2g, x, memories):
        """
        Infer all variables
        """

        # infer entorhinal
        g, hidden_x2g, delta_p2g, mu_p2g = self.infer_g(g2g, x, memories)

        return g, hidden_x2g, delta_p2g, mu_p2g

    @m_u.define_scope
    def generation(self, g, g_gen, memories):
        """
        Generate all variabes
        """
        retrieved_g2x, g_mem_input, hidden_g2x = self.gen_p(g, memories)
        x_g, x_g_logits = self.f_x(retrieved_g2x)

        retrieved_gt2x, _, _ = self.gen_p(g_gen, memories)
        x_gt, x_gt_logits = self.f_x(retrieved_gt2x)

        x = m_u.DotDict({'x_g': x_g,
                         'x_gt': x_gt})
        x_logits = m_u.DotDict({'x_g': x_g_logits,
                                'x_gt': x_gt_logits})

        return x, x_logits, retrieved_g2x, g_mem_input, hidden_g2x

    # INFERENCE FUNCTIONS

    @m_u.define_scope
    def infer_g(self, mu, mu_x2p, memories):
        """
        Infer grids cells
        :param mu: mean from grids on previous time step
        :param mu_x2p: input to attractor from sensory data
        :param memories: memory dict
        :return: inference grid cells
        """

        delta_p2g, mu_p2g, hidden_x2g = None, None, None

        # Inference - factorised posteriors
        if 'mem' in self.par.infer_g_type:
            delta_p2g, mu_p2g, hidden_x2g = self.p2g(mu_x2p, memories, mu)
            mu = mu + self.scalings.p2g_scale * delta_p2g * (mu_p2g - mu)

        if self.par.g_act_after_inference:
            # apply activation
            mu = self.norm(mu, 'g')
            mu = self.activation(mu, 'g')
            mu = m_u.threshold(mu, self.par.g_thresh_min, self.par.g_thresh_max)

        return mu, hidden_x2g, delta_p2g, mu_p2g

    @m_u.define_scope
    def p2g(self, x, memories, g2g_mu):
        """
        Pattern completion - can we aid our inference of where we are based on sensory data that we may have seen before
        :param x: input to place cells from data
        :param g2g_mu: ---
        :param memories: memory dict
        :return: parameters of distributions, as well as terms for Hebbian update
        """

        # extract inverse memory - do softmax(gG^T + alpha*xX^T) G
        # this should attend to both x and g (init alpha as 0.5) - therefore can deal with aliasing!

        retrieved_x2g, hidden_x2g = self.retrieve_memory(x, memories, 'inf', average='none',
                                                         inp_extra=g2g_mu if self.par.inf_mem_use_g else None)

        # now find delta - i.e. how much to change g from the g2g prediction. inputs to deltas:
        # -> g = g2g + delta. delta is a function of g2g, g_retreved, hidden_x * (p2g - g2g)

        # 1. Entropy of retrieved memory distribution
        entropy_hidden = tf.reduce_sum(-hidden_x2g['stored']['prob'] * tf.math.log(hidden_x2g['stored']['prob'] + 1e-8),
                                       axis=1, keepdims=True) + tf.reduce_sum(
            -hidden_x2g['new']['prob'] * tf.math.log(hidden_x2g['new']['prob'] + 1e-8), axis=1, keepdims=True)

        # 2. Is the memory any good? I.e. does retrieved g, re-predict x_comp
        x_retrieved, _ = self.retrieve_memory(retrieved_x2g, memories, 'gen', average=self.par.kernel_norm)
        # compare x_comp with x_comp_retrieved
        err = m_u.squared_error(x, x_retrieved, keepdims=True)
        err = tf.stop_gradient(err)

        # 3. estimated g from inferred mems
        x_mem_g = self.p2g_mu(retrieved_x2g)

        if not self.par.g_act_after_inference:
            # apply activation
            x_mem_g = self.norm(x_mem_g, 'g')
            x_mem_g = self.activation(x_mem_g, 'g')
            x_mem_g = m_u.threshold(x_mem_g, self.par.g_thresh_min, self.par.g_thresh_max)

        delta_input = tf.concat([x_mem_g, g2g_mu, entropy_hidden, err], axis=1)
        delta = tf.nn.sigmoid(self.p2g_delta(delta_input))  # delta should be between 0 and 1

        return delta, x_mem_g, hidden_x2g

    @m_u.define_scope
    def g2p(self, g):
        """
        input from grid cells to place cell layer
        :param g: grid cells
        :return: input to place cell layer
        """
        if self.par.norm_before_downsample:
            g = self.norm(g, 'g2p')

        g2p = self.projection_g(g)

        if not self.par.norm_before_downsample:
            g2p = self.norm(g2p, 'g2p')

        g2p = self.activation(g2p, 'g2p')
        g2p = m_u.threshold(g2p, self.par.g2p_thresh_min, self.par.g2p_thresh_max)

        return g2p

    # GENERATIVE FUNCTIONS
    @m_u.define_scope
    def gen_p(self, g, memories):
        """
        generate place cell based on grids
        :param g: grids
        :param memories: dictionary of memory stuff
        :return:
        """

        # grid input to hippocampus
        g_mem_input = self.g2p(g)

        # retrieve memory via attractor network
        retrieved_g2x, hidden = self.retrieve_memory(g_mem_input, memories, 'gen', average=self.par.kernel_norm)

        return retrieved_g2x, g_mem_input, hidden

    @m_u.define_scope
    def gen_g(self, g, t_mat, seq_pos):
        """
        Get entorhinal transiton. This is g_prior for 1st step in environment.
        :param g:
        :param t_mat:
        :param seq_pos:
        :return:
        """

        seq_pos_ = tf.expand_dims(seq_pos, axis=1)

        # get g transition for generative + apply activation
        mu_gen = self.g2g(g, t_mat)
        mu_gen = self.norm(mu_gen, 'g')
        mu_gen = self.activation(mu_gen, 'g')
        mu_gen = m_u.threshold(mu_gen, self.par.g_thresh_min, self.par.g_thresh_max)

        # get g transition for inference (activation for for inference is at end of self.infer_g)
        mu_inf_ = self.g2g(g, t_mat)
        if not self.par.g_act_after_inference:
            # apply activation (otherwise activation is at end of self.infer_g)
            mu_inf_ = self.norm(mu_inf_, 'g')
            mu_inf_ = self.activation(mu_inf_, 'g')
            mu_inf_ = m_u.threshold(mu_inf_, self.par.g_thresh_min, self.par.g_thresh_max)

        # get prior on g - only used for initial step in environment
        mu_prior = self.g_prior()

        return tf.where(seq_pos_ > 0, mu_gen, mu_prior), tf.where(seq_pos_ > 0, mu_inf_, mu_prior)

    @m_u.define_scope
    def g2g(self, g, t_mat):
        """
        make grid to grid transition
        :param g: grid from previous time-step
        :param t_mat: direction of travel
        :return:
        """

        # transition update
        update = self.get_g2g_update(g, t_mat)
        # add on update to current representation
        mu = update + g

        return mu

    @m_u.define_scope
    def g_prior(self):
        """
        Gives prior distribution for grid cells
        :return:
        """

        mu = self.g_init if self.g_init is not None else tf.tile(self.g_prior_mu, [self.batch_size, 1])

        mu = self.norm(mu, 'g')
        mu = self.activation(mu, 'g')
        mu = m_u.threshold(mu, self.par.g_thresh_min, self.par.g_thresh_max)

        return mu

    @m_u.define_scope
    def get_transition(self, d):
        if self.par.rnn_type == 'TEM':
            # get transition matrix based on relationship / action
            t_vec = self.t_vec(d)
            # turn vector into matrix
            trans_all = tf.reshape(t_vec, [self.batch_size, self.par.g_size, self.par.g_size])
        elif self.par.rnn_type == 'CANN':
            d_translated = tf.matmul(d, self.d_index2space)
            trans_all = self.cann_B(d_translated)
        else:
            raise ValueError(self.par.rnn_type + ' rnn variant not implemented yet')
        return trans_all

    @m_u.define_scope
    def get_g2g_update(self, g_p, t_mat):
        if self.par.rnn_type == 'TEM':
            # multiply current entorhinal representation by transition matrix
            update = tf.squeeze(tf.matmul(t_mat, tf.expand_dims(g_p, axis=2)))
        elif self.par.rnn_type == 'CANN':
            update = self.cann_W(g_p) + t_mat
        else:
            raise ValueError(self.par.rnn_type + ' rnn variant not implemented yet')

        return update

    @m_u.define_scope
    def f_x(self, retrieved_g2x):
        """
        :param retrieved_g2x: retrieved_g2x cells
        :return: sensory predictions
        """
        x_logits = self.MLP_x_pred(retrieved_g2x)

        x = tf.nn.softmax(x_logits)

        # other option would be to use cross entropy on normalised retrieved_g2x ?

        return x, x_logits

    # MEMORY RETRIEVAL FUNCTIONS

    @m_u.define_scope
    def retrieve_memory(self, inp, memories, gen_or_inf, average='sqrt', inp_extra=None):
        """
        Uses scalar products instead of explicit matrix calculations. Makes everything faster.
        Note that this 'efficient implementation' will be costly if our sequence length is greater than the hidden
        state dimensionality
        Wrapper function for actual computation of scalar products
        :param inp: current state of attractor
        :param memories: memory stuff
        :param gen_or_inf:
        :param average
        :param inp_extra
        :return:
        """
        inp = tf.expand_dims(inp, axis=1)

        # 1. Inner products with each memory
        hid_stored, hid_new = self.mem_vis2hid_wrap(memories, inp, gen_or_inf, average=average)

        hid_stored_copy, hid_new_copy = None, None
        if gen_or_inf == 'inf' and self.par.inf_mem_helper:
            # for later use!
            hid_stored_copy, hid_new_copy = tf.identity(hid_stored), tf.identity(hid_new)

        # in inference we use softmax(gG + xX)G
        if gen_or_inf == 'inf' and inp_extra is not None:
            # inp extra is g2g_mu
            inp_extra_ = self.g2p(inp_extra)
            inp_extra_ = tf.expand_dims(inp_extra_, axis=1)
            hid_stored_, hid_new_ = self.mem_vis2hid_wrap(memories, inp_extra_, 'gen', average=self.par.kernel_norm)
            if self.par.extra_mem_inf_type == 'additive':
                hid_stored = hid_stored + 2 * tf.nn.sigmoid(self.alpha_g_x) * hid_stored_
                hid_new = hid_new + 2 * tf.nn.sigmoid(self.alpha_g_x) * hid_new_
            elif self.par.extra_mem_inf_type == 'multiplicative':
                hid_stored = tf.abs(hid_stored) * hid_stored_
                hid_new = tf.abs(hid_new) * hid_new_
            else:
                raise ValueError('Not implemented')
            # set unused memories to -100
            hid_stored = tf.where(memories['stored']['in_use'] == 0, -100 * tf.ones_like(hid_stored), hid_stored)
            hid_new = tf.where(memories['new']['in_use'] == 0, -100 * tf.ones_like(hid_new), hid_new)

        if gen_or_inf == 'inf' and self.par.inf_mem_helper:
            # only keep info from observations seen
            hid_stored = tf.where(hid_stored_copy == 0, -100 * tf.ones_like(hid_stored), hid_stored)
            hid_new = tf.where(hid_new_copy == 0, -100 * tf.ones_like(hid_new), hid_new)

        # 2. Softmax distribution over memories
        f_hid_stored, f_hid_new = self.f_hid_mem(hid_stored, hid_new, memories)

        # 3. Retrieve memories
        vis_stored, vis_new = self.mem_hid2vis_wrap(memories, gen_or_inf, f_hid_stored, f_hid_new)

        poss_updates = vis_stored + vis_new

        return poss_updates, {'stored': {'scal': hid_stored,
                                         'prob': f_hid_stored,
                                         },
                              'new': {'scal': hid_new,
                                      'prob': f_hid_new,
                                      },
                              }

    @m_u.define_scope
    def kernel_similarity(self, p, mem, average='average', kernel=None):
        """
        :param p: b_s x 1 x n_p
        :param mem: b_s x n_p x n_mems
        :param average
        :param kernel
        :return:
        """
        if kernel is None:
            kernel = self.par.similarity_measure

        if kernel == 'dot_product':
            p_z = p
            mem_z = mem
        elif kernel == 'correlation':
            # z-score mems
            mem_std = tf.math.reduce_std(mem, axis=1, keepdims=True) + eps
            mem_z = (mem - tf.reduce_mean(mem, axis=1, keepdims=True)) / mem_std
            # z-score p
            p_std = tf.math.reduce_std(p, axis=2, keepdims=True) + eps
            p_z = (p - tf.reduce_mean(p, axis=2, keepdims=True)) / p_std
        elif kernel == 'cosine_similarity':
            # l2 norm mem
            mem_sq = tf.sqrt(tf.reduce_mean(mem ** 2, axis=1, keepdims=True)) + eps
            mem_z = mem / mem_sq
            # l2 norm mem
            p_sq = tf.sqrt(tf.reduce_mean(p ** 2, axis=2, keepdims=True)) + eps
            p_z = p / p_sq
        else:
            raise ValueError('Unallowed similarity measure')

        # dot_prod
        dot_prod = tf.squeeze(tf.matmul(p_z, mem_z), axis=1)

        # threshold similarities? if layer_normed before the threshold is p.shape[2]
        n_k = tf.cast(p.shape[2], tf.float32)
        dot_prod = m_u.threshold(dot_prod, n_k * self.par.kernel_thresh_min, n_k * self.par.kernel_thresh_max)

        # normalise
        if average == 'average':
            return dot_prod / n_k
        elif average == 'sqrt':
            return dot_prod / tf.sqrt(n_k)
        elif average == 'none':
            return dot_prod
        else:
            raise ValueError('Unallowed average type')

    @m_u.define_scope
    def mem_vis2hid_wrap(self, memories, inp, gen_or_inf, average='sqrt', kernel=None):
        order = self.par.memory_order[gen_or_inf]['in']
        scal_prods_stored = self.mem_vis2hid(memories['stored'][order], memories['stored']['in_use'], inp,
                                             average=average, kernel=kernel)
        scal_prods_new = self.mem_vis2hid(memories['new'][order], memories['new']['in_use'], inp, average=average,
                                          kernel=kernel)

        return scal_prods_stored, scal_prods_new

    @m_u.define_scope
    def mem_vis2hid(self, memories, in_use, inp, average='sqrt', kernel=None):
        """
        :param memories:
        :param in_use
        :param inp: [b_s x 1 x n_p]
        :param average
        :param kernel
        :return:
        """

        # Compute inner products of ps with past memories
        scal_prods = self.kernel_similarity(inp, memories, average=average, kernel=kernel)

        # where weighting exactly zero (i.e. no memory there) set to very low number for softmax
        scal_prods = tf.where(in_use == 0, -100 * tf.ones_like(scal_prods), scal_prods)

        return scal_prods  # b_s x n_mem

    @m_u.define_scope
    def mem_hid2vis_wrap(self, memories, gen_or_inf, f_hid_stored, f_hid_new):
        order = self.par.memory_order[gen_or_inf]['out']
        vis_stored = self.mem_hid2vis(memories['stored'][order], f_hid_stored)
        vis_new = self.mem_hid2vis(memories['new'][order], f_hid_new)

        return vis_stored, vis_new

    @m_u.define_scope
    def mem_hid2vis(self, memories, hid):
        """
        :param memories: stored and new are b_s x n_p x n_mems
        :param hid: b_s x n_stored_mems
        :return:
        """

        updates = tf.squeeze(tf.matmul(memories, tf.expand_dims(hid, axis=2)))

        return updates

    @m_u.define_scope
    def f_hid_mem(self, stored_scal_prods, new_scal_prods, memories):
        # stored_scal_prods: b_s x num_mems
        # new_scal_prods: b_s x num_mems

        # make softmax numerically stable
        max_stored = tf.reduce_max(stored_scal_prods, axis=1, keepdims=True)
        max_new = tf.reduce_max(new_scal_prods, axis=1, keepdims=True)
        max_both = tf.maximum(max_stored, max_new)

        # find effective number of current memories - more memories requires a larger beta weight
        num_mem = tf.reduce_sum(tf.cast(memories['stored']['in_use'] != 0.0, self.precision), axis=1,
                                keepdims=True) + tf.reduce_sum(
            tf.cast(memories['new']['in_use'] != 0.0, self.precision), axis=1, keepdims=True)
        num_mem2 = tf.reduce_sum(tf.cast(stored_scal_prods != -100.0, self.precision), axis=1,
                                 keepdims=True) + tf.reduce_sum(
            tf.cast(new_scal_prods != -100.0, self.precision), axis=1, keepdims=True)
        num_mem = tf.math.minimum(num_mem, num_mem2)

        # scale beta depending on number of current memories - ?plus scale with number of modules?
        beta_weights = self.softmax_beta_0 + tf.math.log(tf.maximum(num_mem - 1.0, 1.0)) * self.par.softmax_beta

        softmax_stored = tf.exp(beta_weights * (stored_scal_prods - max_both))  # b_s x n_stored_mems
        softmax_new = tf.exp(beta_weights * (new_scal_prods - max_both))  # b_s x n_new_mems

        softmax_norm = tf.reduce_sum(softmax_stored, axis=1, keepdims=True) + tf.reduce_sum(softmax_new, axis=1,
                                                                                            keepdims=True)
        # softmax normalised
        hid_stored = softmax_stored / softmax_norm
        hid_new = softmax_new / softmax_norm

        return hid_stored, hid_new

    # Memory functions
    @m_u.define_scope
    def hebbian(self, x, g_projection, h_g2x, h_x2g, mems, mem_i, vis):
        """
        :param x: x
        :param g_projection: g
        :param h_g2x
        :param h_x2g:
        :param mems:
        :param mem_i: memory number
        :param vis:
        :return:

        This process is equivalent to updating Hebbian matrices, though it is more computationally efficient.
        See Ba et al 2016.
        """

        # Remove past memory that was good at prediction. Do this so BPTT has something to work with.
        # returns the weighting - now if a weighting is zero, this means memory has been deleted!
        mems, add_in_use = self.keep_single_copy_of_mem(x, g_projection, h_g2x, h_x2g, mems, vis, mem_i)

        # add new memory
        mems['new']['g'] = self.mem_update(g_projection, mems['new']['g'], mem_i)
        mems['new']['x'] = self.mem_update(x, mems['new']['x'], mem_i)

        # update 'in_use' i.e. 1 if just added memory!
        mems['new']['in_use'] = self.mem_weight_update(add_in_use, mems['new']['in_use'], mem_i)

        return mems

    @m_u.define_scope
    def keep_single_copy_of_mem(self, x, g_projection, h_g2x, h_x2g, mems, vis, mem_i):
        """
        We want to always add a new memory and delete the old one, so we can backprop well.
        Equivalent to only keeping a single memory around (assuming it learns how to use memories effectively)
        Weight memories - 1 weighting if memory was unused. 0 weighting if memory used.
        :param x:
        :param g_projection:
        :param h_g2x:
        :param h_x2g:
        :param mems:
        :param vis
        :param mem_i
        :return:
        """

        # if came from stored -> remove from stored,
        # if came from new -> remove *current* memory : this is so path integration learns long range stuff

        # -->
        # If we have seen this g-x combination before, and that the memory we used, then remove it
        # Find maximal memory. Do cosine of inferred g to memory g. Cosine of observed x to memory x.
        # If both above some threshold then remove memory

        # --> The most preferentially used memory
        zeros_new_add = tf.zeros((self.par.batch_size, self.par.seq_len - mem_i))
        h_g = tf.concat([h_g2x['new']['prob'], zeros_new_add], axis=1)
        max_hidden = tf.maximum(tf.reduce_max(h_g2x['stored']['prob'], axis=1), tf.reduce_max(h_g, axis=1))
        preferential_stored = h_g2x['stored']['prob'] >= tf.expand_dims(max_hidden, axis=1)
        preferential_new = h_g >= tf.expand_dims(max_hidden, axis=1)

        # --> Have we seen this sensory observation before. No need for cosine similarity as x is one-hot!
        if h_x2g is None or self.par.inf_mem_use_g:
            # note: not using inp_extra here...
            _, h_x2g = self.retrieve_memory(x, mems, 'inf', average='none')
            large_similarties_x2g_new = h_x2g['new']['scal'] >= 0.9
        else:
            large_similarties_x2g_new = tf.concat([h_x2g['new']['scal'], zeros_new_add], axis=1) >= 0.9
        large_similarties_x2g_stored = h_x2g['stored']['scal'] >= 0.9

        # --> have we seen this g before? Need cosine similarity as g can be anything.
        hid_stored, hid_new = self.mem_vis2hid_wrap(mems, tf.expand_dims(g_projection, axis=1), 'gen',
                                                    average='average', kernel='correlation')
        large_similarties_stored = hid_stored >= self.par.key_mem_thresh
        large_similarties_new = hid_new >= self.par.key_mem_thresh  # tf.concat([hid_new, zeros_new_add], axis=1)

        used_s = tf.logical_and(large_similarties_stored, large_similarties_x2g_stored)
        used_n = tf.logical_and(large_similarties_new, large_similarties_x2g_new)
        used_s = tf.logical_and(used_s, preferential_stored)
        used_n = tf.logical_and(used_n, preferential_new)

        if self.par.perfect_memory_add_remove:
            """	
            Perhaps while sorting out other params, delete memory if vis = 1	
            delete mem with highest scal_prod?	
            """
            used_s = tf.logical_and(preferential_stored, tf.cast(tf.expand_dims(vis, axis=1), dtype=tf.bool))
            used_n = tf.logical_and(preferential_new, tf.cast(tf.expand_dims(vis, axis=1), dtype=tf.bool))

        # Give 0 weighting if memory used - i.e. we overwrite it
        # Either multiply by the mask (i.e. tf.cast(used_s, tf.float32)), or use tf.where.
        mems['stored']['in_use'] = tf.where(used_s, tf.zeros_like(mems['stored']['in_use']), mems['stored']['in_use'])
        if self.par.overwrite_bptt_mems:
            # don't add new memory if new got used -> makes path-int have to learn long range -> stop grid drift
            mems['new']['in_use'] = tf.where(used_n, tf.zeros_like(mems['new']['in_use']), mems['new']['in_use'])
            add_in_use = tf.ones(self.par.batch_size, dtype=self.precision)
        else:
            add_in_use = 1.0 - tf.minimum(tf.reduce_sum(tf.cast(used_n, dtype=self.precision), axis=1), 1.0)

        return mems, add_in_use

    @m_u.define_scope
    def mem_update(self, mem_update, mems, mem_num):
        """
        Update bank of memories (for scalar product computations)
        :param mem_update: memory to add
        :param mems: current memories
        :param mem_num:
        :return:
        """
        indices = tf.expand_dims(tf.expand_dims(mem_num, axis=0), axis=0)

        # add new memory - clearly shouldn't have to do two transposes
        mems = tf.transpose(mems, [2, 0, 1])
        mems = tf.tensor_scatter_nd_update(mems, indices, tf.expand_dims(mem_update, axis=0))
        mems = tf.transpose(mems, [1, 2, 0])

        return mems

    @m_u.define_scope
    def mem_weight_update(self, weight_update, weights, mem_num):
        """
        Update bank of memories (for scalar product computations)
        :param weights: memory to add
        :param weight_update: current memories
        :param mem_num:
        :return:
        """

        indices = tf.expand_dims(tf.expand_dims(mem_num, axis=0), axis=0)

        weights = tf.transpose(weights, [1, 0])
        weights = tf.tensor_scatter_nd_update(weights, indices, tf.expand_dims(weight_update, axis=0))
        weights = tf.transpose(weights, [1, 0])

        return weights

    # Activation functions

    @m_u.define_scope
    def activation(self, x, name):
        if name == 'g':
            act = self.f_g
        elif name == 'g2p':
            act = self.f_g2p
        else:
            raise ValueError('Name <' + name + '> not supported')

        return act(x)

    @m_u.define_scope
    def f_g(self, g):
        if self.par.g_act == 'none':
            return g
        elif self.par.g_act == 'orig_tem':
            return tf.minimum(tf.maximum(g, -1), 1)
        elif self.par.g_act == 'tanh':
            return tf.nn.tanh(g)
        elif self.par.g_act == 'leaky_relu':
            return tf.nn.leaky_relu(g, alpha=self.par.leaky_relu_alpha)
        elif self.par.g_act == 'relu':
            return tf.nn.relu(g)
        else:
            raise ValueError('g activation not implemented')

    @m_u.define_scope
    def f_g2p(self, g2p):
        return g2p  # tf.nn.leaky_relu(g2p, alpha=0.01) # tf.nn.leaky_relu(tf.minimum(tf.maximum(p, -1), 1), alpha=0.01)

    @m_u.define_scope
    def norm(self, x, name):
        if name == 'g':
            norm = self.norm_g
        elif name == 'g2p':
            norm = self.norm_g2p
        else:
            raise ValueError('Name <' + name + '> not supported')

        return norm(x)

    @m_u.define_scope
    def norm_g(self, g):
        if self.par.g_norm == 'layer_norm':
            return self.layernorm_g(g)
        elif self.par.g_norm == 'none':
            return g
        else:
            raise ValueError('norm type not allowed')

    @m_u.define_scope
    def norm_g2p(self, g2p):
        if self.par.g2p_norm == 'unit_norm':
            return self.unitnorm_scale_g2p * tf.math.l2_normalize(g2p, axis=1)
        elif self.par.g2p_norm == 'layer_norm':
            return self.layernorm_g2p(g2p)
        elif self.par.g2p_norm == 'none':
            return g2p
        else:
            raise ValueError('norm type not allowed')

    @m_u.define_scope
    def init_mems(self, stored_mems, new_mems):

        memories_dict = {
            'stored': stored_mems,
            'new': {'x': tf.zeros((self.batch_size, self.par.s_size_project, new_mems), dtype=self.precision),
                    'g': tf.zeros((self.batch_size, self.par.phase_size, new_mems), dtype=self.precision),
                    'in_use': tf.zeros((self.batch_size, new_mems), dtype=self.precision)},
        }

        return memories_dict

    @m_u.define_scope
    def mem_step(self, mems, itnum):

        mem_s = {
            'stored': mems['stored'],
            'new': {'x': mems['new']['x'][:, :, :itnum],
                    'g': mems['new']['g'][:, :, :itnum],
                    'in_use': mems['new']['in_use'][:, :itnum]
                    }
        }
        return mem_s

    @m_u.define_scope
    def init_input(self, inputs, new_mems=None):
        """
        Set model member variables from inputs and prepare memory and variable dictionaries
        """
        # Set member variables from input
        self.batch_size = inputs.x[0].shape[0]
        self.scalings = inputs.scalings

        # Find how many new memories will be created in this forward pass - length of input sequence by default
        new_mems = self.par.seq_len if new_mems is None else new_mems
        # Create memory and data dictionaries
        memories_dict = self.init_mems(inputs.stored_mems, new_mems)
        variable_dict = self.init_vars()
        # Return dicts
        return memories_dict, variable_dict

    @m_u.define_scope
    def init_vars(self, seq_len=None):
        """
        Collecting variables for losses, accuracies and saving. Start with all fields that can possibly be collected.
        Then when generating output in tensorarray_2_list, only stack those fields that were actually written to.
        Tensorflow annoying any wont deal with list appends with tf.range, so using TensorArray instead
        """
        # Total number of variables collected: if not provided, default to the length of the backprop sequence
        seq_len = self.par.seq_len if seq_len is None else seq_len

        # Create dictionary with all possible data for saving
        vars_dict = m_u.DotDict(
            {'g': {'g': tf.TensorArray(self.precision, size=seq_len, clear_after_read=False, name='ta_g'),
                   'g_gen': tf.TensorArray(self.precision, size=seq_len, clear_after_read=False, name='ta_g_gen'),
                   'g_mem_input': tf.TensorArray(self.precision, size=seq_len, clear_after_read=False,
                                                 name='ta_g_mem_input'),
                   },
             'p2g': {'p2g': tf.TensorArray(self.precision, size=seq_len, clear_after_read=False, name='ta_p2g'),
                     'delta': tf.TensorArray(self.precision, size=seq_len, clear_after_read=False, name='ta_delta'),
                     },
             'pred': {'x_g': tf.TensorArray(self.precision, size=seq_len, clear_after_read=False, name='ta_x_g'),
                      'x_gt': tf.TensorArray(self.precision, size=seq_len, clear_after_read=False, name='ta_x_gt'),
                      'd': tf.TensorArray(self.precision, size=seq_len - 1, clear_after_read=False, name='ta_d')
                      },
             'logits': {
                 'x_g': tf.TensorArray(self.precision, size=seq_len, clear_after_read=False, name='ta_logit_x_g'),
                 'x_gt': tf.TensorArray(self.precision, size=seq_len, clear_after_read=False, name='ta_logit_x_gt'),
                 'd': tf.TensorArray(self.precision, size=seq_len - 1, clear_after_read=False, name='ta_logit_d')
             },
             'mem_dist': {'inner_prods': tf.TensorArray(self.precision, size=seq_len, clear_after_read=False,
                                                        name='ta_inner_prods'),
                          'probabilities': tf.TensorArray(self.precision, size=seq_len, clear_after_read=False,
                                                          name='ta_probabilities')
                          },
             })

        return vars_dict

    @m_u.define_scope
    def update_vars(self, input_dict, updates, i):
        """
        Select specific fields for writing to an output array, or by default write all used values
        """
        # Create output vars_dict, which will have the requested fields
        output_dict = {}

        # Find all keys in input dict
        all_keys = m_u.get_all_keys(input_dict)

        # Get updated keys and values by 'transposing' updates input
        update_keys, update_vals = [list(field) for field in zip(*updates)]

        # Run through all keys. Simply copy from input dict, unless new value was specified as update
        for key in all_keys:
            # Get tensorarray from input dict
            input_val = m_u.nested_get(input_dict, key)
            if key in update_keys:
                # If an update was provided: set update to corresponding value
                m_u.nested_set(output_dict, key, input_val.write(i, update_vals[update_keys.index(key)]))
            else:
                # If no update was provided: simply copy the original value
                m_u.nested_set(output_dict, key, input_val)

        # Return output dict
        return m_u.DotDict(output_dict)

    @m_u.define_scope
    def precomp_trans(self, dirs, seq_len=None, name=None):
        """
        Precompute transitions for provided tensor of directions
        """
        # If sequence length is not specified: use full sequence length from parameters
        seq_len = self.par.seq_len if seq_len is None else seq_len
        # alternatively could pre-compute all types of actions and then use control flow
        ta_mat = tf.TensorArray(self.precision, size=seq_len, clear_after_read=False,
                                name='t_mat' + ('' if name is None else name))
        ds = tf.unstack(dirs, axis=0)
        for j, d in enumerate(ds):
            # Get transition matrix from action/relation
            new_ta = self.get_transition(d)
            # And write transitions for this iteration to ta_mat
            ta_mat = ta_mat.write(j, new_ta)
        return ta_mat

    @m_u.define_scope
    def tensorarray_2_list_old(self, variable_dict):
        # likely not the best way to do this...
        vars_dict = m_u.DotDict({'g': {'g': tf.unstack(variable_dict.g.g.stack(), axis=0, name='g_unstack'),
                                       'g_gen': tf.unstack(variable_dict.g.g_gen.stack(), axis=0,
                                                           name='g_gen_unstack'),
                                       },
                                 'p2g': {'delta': tf.unstack(variable_dict.p2g.delta.stack(), axis=0, name='delta'),
                                         'p2g': tf.unstack(variable_dict.p2g.p2g.stack(), axis=0, name='p2g'),
                                         },
                                 'pred': {'x_g': tf.unstack(variable_dict.pred.x_g.stack(), axis=0,
                                                            name='x_g_unstack'),
                                          'x_gt': tf.unstack(variable_dict.pred.x_gt.stack(), axis=0,
                                                             name='x_gt_unstack'),
                                          },
                                 'logits': {'x_g': tf.unstack(variable_dict.logits.x_g.stack(), axis=0,
                                                              name='x_g_unstack'),
                                            'x_gt': tf.unstack(variable_dict.logits.x_gt.stack(), axis=0,
                                                               name='x_gt_unstack'),
                                            },
                                 })

        # Add action predictions, if they exist
        if 'd' in variable_dict.pred:
            # Note ['pred']['d'] instead of .pred.d: DotDict nested assignment doesn't work
            vars_dict['pred']['d'] = tf.unstack(variable_dict.pred.d.stack(), axis=0, name='d_unstack')
            vars_dict['logits']['d'] = tf.unstack(variable_dict.logits.d.stack(), axis=0, name='d_unstack')

        return vars_dict

    @m_u.define_scope
    def tensorarray_2_list(self, variable_dict):
        """
        Select specific fields for writing to an output array, or by default write all used values
        """
        # If no selection of keys to write was provided: simply select all
        keys_to_write = m_u.get_all_keys(variable_dict)

        # Create output vars_dict, which will have the requested fields
        vars_dict = {}

        # Then set the values of vars_dict according to fields to write from the input variable dict
        for key in keys_to_write:
            # Retrieve the value of the nested key from input variable dict and stack
            value = m_u.nested_get(variable_dict, key)
            # Convert value to list if it is a tensorarray
            if isinstance(value, tf.TensorArray):
                # Convert tensorarray to list if it was written to at least once
                value = None if value.element_shape == tf.TensorShape(None) \
                    else tf.unstack(value.stack(), axis=0, name=key[-1] + '_unstack')
            # Set the value of the nested key in the output dict
            m_u.nested_set(vars_dict, key, value)

        # Return output dict
        return m_u.DotDict(vars_dict)


@m_u.define_scope
def compute_losses(model_inputs, data, trainable_variables, par):
    lx_g = 0.0
    lx_gt = 0.0
    lg = 0.0
    lg_reg = 0.0
    lg_non_neg = 0.0
    keys_non_neg = 0.0
    keys_sparsity = 0.0
    keys_l2 = 0.0
    norm, s_vis = 1.0, 1.0

    xs = model_inputs.x
    scalings = model_inputs.scalings
    s_visited = tf.unstack(model_inputs.s_visited, axis=1)

    for i in range(par.seq_len):
        # losses for each batch
        lx_g_ = m_u.sparse_softmax_cross_entropy_with_logits(xs[i], data.logits.x_g[i])
        lx_gt_ = m_u.sparse_softmax_cross_entropy_with_logits(xs[i], data.logits.x_gt[i])

        lg_ = m_u.squared_error(data.g.g[i], data.g.g_gen[i])

        lg_reg_ = tf.reduce_sum(data.g.g[i] ** 2, axis=1)
        lg_non_neg_ = tf.reduce_sum(tf.nn.relu(-data.g.g[i]), axis=1)
        keys_non_neg_ = tf.reduce_sum(tf.nn.relu(-data.g.g_mem_input[i]), axis=1)
        keys_sparsity_ = tf.reduce_sum(tf.math.abs(data.g.g_mem_input[i]), axis=1)
        keys_l2_ = tf.reduce_sum(data.g.g_mem_input[i] ** 2, axis=1)

        # don't train on any time-steps without when haven't visited that state before.
        if par.train_on_visited_states_only:
            s_vis = s_visited[i]
            batch_vis = tf.reduce_sum(s_vis) + eps
            # normalise for bptt sequence length
            norm = 1.0 / (batch_vis * par.seq_len)

        lx_g += tf.reduce_sum(lx_g_ * s_vis) * norm
        lx_gt += tf.reduce_sum(lx_gt_ * s_vis) * par.lx_gt_val * norm
        lg += tf.reduce_sum(lg_ * s_vis) * par.lg_val * norm

        lg_reg += tf.reduce_sum(lg_reg_ * s_vis) * par.g_reg_pen * norm
        lg_non_neg += tf.reduce_sum(lg_non_neg_ * s_vis) * par.g_non_neg_pen * norm
        keys_non_neg += tf.reduce_sum(keys_non_neg_ * s_vis) * par.g_non_neg_pen * norm
        keys_sparsity += tf.reduce_sum(keys_sparsity_ * s_vis) * par.keys_sparsity_pen * norm
        keys_l2 += tf.reduce_sum(keys_l2_ * s_vis) * par.keys_l2_pen * norm

    losses = m_u.DotDict()
    cost_all = 0.0
    losses.lx_gt = lx_gt
    losses.lx_g = lx_g
    if 'lx_gt' in par.which_costs:
        cost_all += lx_gt * (1 + scalings.g_gt)
    if 'lx_g' in par.which_costs:
        cost_all += lx_g * (1 - scalings.g_gt)
    if 'lg' in par.which_costs:
        cost_all += lg * scalings.temp * par.lg_temp
        losses.lg = lg * scalings.temp * par.lg_temp
        losses.lg_unscaled = lg
    if 'lg_reg' in par.which_costs:
        cost_all += lg_reg * scalings.g_cell_reg
        losses.lg_reg = lg_reg * scalings.g_cell_reg
        losses.lg_reg_unscaled = lg_reg
    if 'lg_non_neg' in par.which_costs:
        cost_all += lg_non_neg
        losses.lg_non_neg = lg_non_neg
    if 'keys_non_neg' in par.which_costs:
        cost_all += keys_non_neg
        losses.keys_non_neg = keys_non_neg
    if 'keys_sparsity' in par.which_costs:
        cost_all += keys_sparsity
        losses.keys_sparsity = keys_sparsity
    if 'keys_l2' in par.which_costs:
        cost_all += keys_l2
        losses.keys_l2 = keys_l2
    if 'weight_reg' in par.which_costs:
        losses.weight_reg = tf.add_n([tf.nn.l2_loss(v) for v in trainable_variables if
                                      any([x not in v.name for x in ['bias', 'layer_norm']])]) * par.weight_reg_val
        cost_all += losses.weight_reg
    if 'orthog_transformer' in par.which_costs:
        projection_g = [x for x in trainable_variables if 'projection' in x.name and 'kernel' in x.name][0]
        w_w_t = tf.matmul(tf.transpose(projection_g), projection_g)
        losses.orthog_transformer = tf.reduce_sum(tf.reduce_sum((w_w_t - tf.eye(par.phase_size)) ** 2, axis=1),
                                                  axis=0) * par.orthog_transformer_reg_val
        cost_all += losses.weight_reg

    losses.train_loss = cost_all

    return losses
