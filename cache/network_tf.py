import tensorflow as tf
import numpy as np


tf_types = {'float32': tf.float32, 'int8': tf.int8}


def get_trainable_params(input_shape, training):
    trainable_params = dict()
    input_data = tf.zeros(shape=input_shape, name=None)
    conv_5fo_filters_init = tf.keras.initializers.glorot_uniform(seed=1)((3, 3, input_data.get_shape().as_list()[-1], 16))
    conv_5fo_filters = tf.Variable(initial_value=conv_5fo_filters_init, trainable=True, caching_device=None, name='conv_5fo/filters', variable_def=None, dtype=tf_types['float32'], import_scope=None, constraint=None, synchronization=tf.VariableSynchronization.AUTO, shape=None)
    trainable_params['conv_5fo/filters'] = conv_5fo_filters
    conv_5fo = tf.nn.conv2d(input=input_data, filters=conv_5fo_filters, strides=[1, 1], padding='SAME', data_format='NHWC', dilations=1, use_bias=False, name='conv_5fo')
    batch_normalize_7he_mean_init = tf.zeros_initializer()((conv_5fo.get_shape().as_list()[-1], ))
    batch_normalize_7he_mean = tf.Variable(initial_value=batch_normalize_7he_mean_init, trainable=True, caching_device=None, name='batch_normalize_7he/mean', variable_def=None, dtype=tf_types['float32'], import_scope=None, constraint=None, synchronization=tf.VariableSynchronization.AUTO, shape=None)
    trainable_params['batch_normalize_7he/mean'] = batch_normalize_7he_mean
    batch_normalize_7he_variance_init = tf.ones_initializer()((conv_5fo.get_shape().as_list()[-1], ))
    batch_normalize_7he_variance = tf.Variable(initial_value=batch_normalize_7he_variance_init, trainable=True, caching_device=None, name='batch_normalize_7he/variance', variable_def=None, dtype=tf_types['float32'], import_scope=None, constraint=None, synchronization=tf.VariableSynchronization.AUTO, shape=None)
    trainable_params['batch_normalize_7he/variance'] = batch_normalize_7he_variance
    batch_normalize_7he_offset_init = tf.zeros_initializer()((conv_5fo.get_shape().as_list()[-1], ))
    batch_normalize_7he_offset = tf.Variable(initial_value=batch_normalize_7he_offset_init, trainable=True, caching_device=None, name='batch_normalize_7he/offset', variable_def=None, dtype=tf_types['float32'], import_scope=None, constraint=None, synchronization=tf.VariableSynchronization.AUTO, shape=None)
    trainable_params['batch_normalize_7he/offset'] = batch_normalize_7he_offset
    batch_normalize_7he_scale_init = tf.ones_initializer()((conv_5fo.get_shape().as_list()[-1], ))
    batch_normalize_7he_scale = tf.Variable(initial_value=batch_normalize_7he_scale_init, trainable=True, caching_device=None, name='batch_normalize_7he/scale', variable_def=None, dtype=tf_types['float32'], import_scope=None, constraint=None, synchronization=tf.VariableSynchronization.AUTO, shape=None)
    trainable_params['batch_normalize_7he/scale'] = batch_normalize_7he_scale
    batch_normalize_7he = tf.nn.batch_normalization(x=conv_5fo, mean=batch_normalize_7he_mean, variance=batch_normalize_7he_variance, offset=batch_normalize_7he_offset, scale=batch_normalize_7he_scale, variance_epsilon=0.001, name='batch_normalize_7he')
    relu_9ju = tf.nn.relu(features=batch_normalize_7he, name='relu_9ju')
    resnet_16_24yk_conv_11lk_filters_init = tf.keras.initializers.glorot_uniform(seed=1)((3, 3, relu_9ju.get_shape().as_list()[-1], 16))
    resnet_16_24yk_conv_11lk_filters = tf.Variable(initial_value=resnet_16_24yk_conv_11lk_filters_init, trainable=True, caching_device=None, name='resnet_16_24yk/conv_11lk/filters', variable_def=None, dtype=tf_types['float32'], import_scope=None, constraint=None, synchronization=tf.VariableSynchronization.AUTO, shape=None)
    trainable_params['resnet_16_24yk/conv_11lk/filters'] = resnet_16_24yk_conv_11lk_filters
    resnet_16_24yk_conv_11lk = tf.nn.conv2d(input=relu_9ju, filters=resnet_16_24yk_conv_11lk_filters, strides=[1, 1], padding='SAME', data_format='NHWC', dilations=1, use_bias=False, name='resnet_16_24yk/conv_11lk')
    resnet_16_24yk_batch_normalize_13na_mean_init = tf.zeros_initializer()((resnet_16_24yk_conv_11lk.get_shape().as_list()[-1], ))
    resnet_16_24yk_batch_normalize_13na_mean = tf.Variable(initial_value=resnet_16_24yk_batch_normalize_13na_mean_init, trainable=True, caching_device=None, name='resnet_16_24yk/batch_normalize_13na/mean', variable_def=None, dtype=tf_types['float32'], import_scope=None, constraint=None, synchronization=tf.VariableSynchronization.AUTO, shape=None)
    trainable_params['resnet_16_24yk/batch_normalize_13na/mean'] = resnet_16_24yk_batch_normalize_13na_mean
    resnet_16_24yk_batch_normalize_13na_variance_init = tf.ones_initializer()((resnet_16_24yk_conv_11lk.get_shape().as_list()[-1], ))
    resnet_16_24yk_batch_normalize_13na_variance = tf.Variable(initial_value=resnet_16_24yk_batch_normalize_13na_variance_init, trainable=True, caching_device=None, name='resnet_16_24yk/batch_normalize_13na/variance', variable_def=None, dtype=tf_types['float32'], import_scope=None, constraint=None, synchronization=tf.VariableSynchronization.AUTO, shape=None)
    trainable_params['resnet_16_24yk/batch_normalize_13na/variance'] = resnet_16_24yk_batch_normalize_13na_variance
    resnet_16_24yk_batch_normalize_13na_offset_init = tf.zeros_initializer()((resnet_16_24yk_conv_11lk.get_shape().as_list()[-1], ))
    resnet_16_24yk_batch_normalize_13na_offset = tf.Variable(initial_value=resnet_16_24yk_batch_normalize_13na_offset_init, trainable=True, caching_device=None, name='resnet_16_24yk/batch_normalize_13na/offset', variable_def=None, dtype=tf_types['float32'], import_scope=None, constraint=None, synchronization=tf.VariableSynchronization.AUTO, shape=None)
    trainable_params['resnet_16_24yk/batch_normalize_13na/offset'] = resnet_16_24yk_batch_normalize_13na_offset
    resnet_16_24yk_batch_normalize_13na_scale_init = tf.ones_initializer()((resnet_16_24yk_conv_11lk.get_shape().as_list()[-1], ))
    resnet_16_24yk_batch_normalize_13na_scale = tf.Variable(initial_value=resnet_16_24yk_batch_normalize_13na_scale_init, trainable=True, caching_device=None, name='resnet_16_24yk/batch_normalize_13na/scale', variable_def=None, dtype=tf_types['float32'], import_scope=None, constraint=None, synchronization=tf.VariableSynchronization.AUTO, shape=None)
    trainable_params['resnet_16_24yk/batch_normalize_13na/scale'] = resnet_16_24yk_batch_normalize_13na_scale
    resnet_16_24yk_batch_normalize_13na = tf.nn.batch_normalization(x=resnet_16_24yk_conv_11lk, mean=resnet_16_24yk_batch_normalize_13na_mean, variance=resnet_16_24yk_batch_normalize_13na_variance, offset=resnet_16_24yk_batch_normalize_13na_offset, scale=resnet_16_24yk_batch_normalize_13na_scale, variance_epsilon=0.001, name='resnet_16_24yk/batch_normalize_13na')
    resnet_16_24yk_relu_15pq = tf.nn.relu(features=resnet_16_24yk_batch_normalize_13na, name='resnet_16_24yk/relu_15pq')
    resnet_16_24yk_conv_17rg_filters_init = tf.keras.initializers.glorot_uniform(seed=1)((3, 3, resnet_16_24yk_relu_15pq.get_shape().as_list()[-1], 16))
    resnet_16_24yk_conv_17rg_filters = tf.Variable(initial_value=resnet_16_24yk_conv_17rg_filters_init, trainable=True, caching_device=None, name='resnet_16_24yk/conv_17rg/filters', variable_def=None, dtype=tf_types['float32'], import_scope=None, constraint=None, synchronization=tf.VariableSynchronization.AUTO, shape=None)
    trainable_params['resnet_16_24yk/conv_17rg/filters'] = resnet_16_24yk_conv_17rg_filters
    resnet_16_24yk_conv_17rg = tf.nn.conv2d(input=resnet_16_24yk_relu_15pq, filters=resnet_16_24yk_conv_17rg_filters, strides=[1, 1], padding='SAME', data_format='NHWC', dilations=1, use_bias=False, name='resnet_16_24yk/conv_17rg')
    resnet_16_24yk_conv_mean_init = tf.zeros_initializer()((resnet_16_24yk_conv_17rg.get_shape().as_list()[-1], ))
    resnet_16_24yk_conv_mean = tf.Variable(initial_value=resnet_16_24yk_conv_mean_init, trainable=True, caching_device=None, name='resnet_16_24yk/conv/mean', variable_def=None, dtype=tf_types['float32'], import_scope=None, constraint=None, synchronization=tf.VariableSynchronization.AUTO, shape=None)
    trainable_params['resnet_16_24yk/conv/mean'] = resnet_16_24yk_conv_mean
    resnet_16_24yk_conv_variance_init = tf.ones_initializer()((resnet_16_24yk_conv_17rg.get_shape().as_list()[-1], ))
    resnet_16_24yk_conv_variance = tf.Variable(initial_value=resnet_16_24yk_conv_variance_init, trainable=True, caching_device=None, name='resnet_16_24yk/conv/variance', variable_def=None, dtype=tf_types['float32'], import_scope=None, constraint=None, synchronization=tf.VariableSynchronization.AUTO, shape=None)
    trainable_params['resnet_16_24yk/conv/variance'] = resnet_16_24yk_conv_variance
    resnet_16_24yk_conv_offset_init = tf.zeros_initializer()((resnet_16_24yk_conv_17rg.get_shape().as_list()[-1], ))
    resnet_16_24yk_conv_offset = tf.Variable(initial_value=resnet_16_24yk_conv_offset_init, trainable=True, caching_device=None, name='resnet_16_24yk/conv/offset', variable_def=None, dtype=tf_types['float32'], import_scope=None, constraint=None, synchronization=tf.VariableSynchronization.AUTO, shape=None)
    trainable_params['resnet_16_24yk/conv/offset'] = resnet_16_24yk_conv_offset
    resnet_16_24yk_conv_scale_init = tf.ones_initializer()((resnet_16_24yk_conv_17rg.get_shape().as_list()[-1], ))
    resnet_16_24yk_conv_scale = tf.Variable(initial_value=resnet_16_24yk_conv_scale_init, trainable=True, caching_device=None, name='resnet_16_24yk/conv/scale', variable_def=None, dtype=tf_types['float32'], import_scope=None, constraint=None, synchronization=tf.VariableSynchronization.AUTO, shape=None)
    trainable_params['resnet_16_24yk/conv/scale'] = resnet_16_24yk_conv_scale
    resnet_16_24yk_conv = tf.nn.batch_normalization(x=resnet_16_24yk_conv_17rg, mean=resnet_16_24yk_conv_mean, variance=resnet_16_24yk_conv_variance, offset=resnet_16_24yk_conv_offset, scale=resnet_16_24yk_conv_scale, variance_epsilon=0.001, name='resnet_16_24yk/conv')
    resnet_16_24yk_add_21vm = tf.math.add(x=[relu_9ju, resnet_16_24yk_conv][0], y=[relu_9ju, resnet_16_24yk_conv][1], name='resnet_16_24yk/add_21vm')
    resnet_16_24yk_relu_23xc = tf.nn.relu(features=resnet_16_24yk_add_21vm, name='resnet_16_24yk/relu_23xc')
    resnet_32_short_cut_43rg_conv0_filters_init = tf.keras.initializers.glorot_uniform(seed=1)((3, 3, resnet_16_24yk_relu_23xc.get_shape().as_list()[-1], 32))
    resnet_32_short_cut_43rg_conv0_filters = tf.Variable(initial_value=resnet_32_short_cut_43rg_conv0_filters_init, trainable=True, caching_device=None, name='resnet_32_short_cut_43rg/conv0/filters', variable_def=None, dtype=tf_types['float32'], import_scope=None, constraint=None, synchronization=tf.VariableSynchronization.AUTO, shape=None)
    trainable_params['resnet_32_short_cut_43rg/conv0/filters'] = resnet_32_short_cut_43rg_conv0_filters
    resnet_32_short_cut_43rg_conv0 = tf.nn.conv2d(input=resnet_16_24yk_relu_23xc, filters=resnet_32_short_cut_43rg_conv0_filters, strides=[2, 2], padding='SAME', data_format='NHWC', dilations=1, use_bias=False, name='resnet_32_short_cut_43rg/conv0')
    resnet_32_short_cut_43rg_batch_normalize_28cq_mean_init = tf.zeros_initializer()((resnet_32_short_cut_43rg_conv0.get_shape().as_list()[-1], ))
    resnet_32_short_cut_43rg_batch_normalize_28cq_mean = tf.Variable(initial_value=resnet_32_short_cut_43rg_batch_normalize_28cq_mean_init, trainable=True, caching_device=None, name='resnet_32_short_cut_43rg/batch_normalize_28cq/mean', variable_def=None, dtype=tf_types['float32'], import_scope=None, constraint=None, synchronization=tf.VariableSynchronization.AUTO, shape=None)
    trainable_params['resnet_32_short_cut_43rg/batch_normalize_28cq/mean'] = resnet_32_short_cut_43rg_batch_normalize_28cq_mean
    resnet_32_short_cut_43rg_batch_normalize_28cq_variance_init = tf.ones_initializer()((resnet_32_short_cut_43rg_conv0.get_shape().as_list()[-1], ))
    resnet_32_short_cut_43rg_batch_normalize_28cq_variance = tf.Variable(initial_value=resnet_32_short_cut_43rg_batch_normalize_28cq_variance_init, trainable=True, caching_device=None, name='resnet_32_short_cut_43rg/batch_normalize_28cq/variance', variable_def=None, dtype=tf_types['float32'], import_scope=None, constraint=None, synchronization=tf.VariableSynchronization.AUTO, shape=None)
    trainable_params['resnet_32_short_cut_43rg/batch_normalize_28cq/variance'] = resnet_32_short_cut_43rg_batch_normalize_28cq_variance
    resnet_32_short_cut_43rg_batch_normalize_28cq_offset_init = tf.zeros_initializer()((resnet_32_short_cut_43rg_conv0.get_shape().as_list()[-1], ))
    resnet_32_short_cut_43rg_batch_normalize_28cq_offset = tf.Variable(initial_value=resnet_32_short_cut_43rg_batch_normalize_28cq_offset_init, trainable=True, caching_device=None, name='resnet_32_short_cut_43rg/batch_normalize_28cq/offset', variable_def=None, dtype=tf_types['float32'], import_scope=None, constraint=None, synchronization=tf.VariableSynchronization.AUTO, shape=None)
    trainable_params['resnet_32_short_cut_43rg/batch_normalize_28cq/offset'] = resnet_32_short_cut_43rg_batch_normalize_28cq_offset
    resnet_32_short_cut_43rg_batch_normalize_28cq_scale_init = tf.ones_initializer()((resnet_32_short_cut_43rg_conv0.get_shape().as_list()[-1], ))
    resnet_32_short_cut_43rg_batch_normalize_28cq_scale = tf.Variable(initial_value=resnet_32_short_cut_43rg_batch_normalize_28cq_scale_init, trainable=True, caching_device=None, name='resnet_32_short_cut_43rg/batch_normalize_28cq/scale', variable_def=None, dtype=tf_types['float32'], import_scope=None, constraint=None, synchronization=tf.VariableSynchronization.AUTO, shape=None)
    trainable_params['resnet_32_short_cut_43rg/batch_normalize_28cq/scale'] = resnet_32_short_cut_43rg_batch_normalize_28cq_scale
    resnet_32_short_cut_43rg_batch_normalize_28cq = tf.nn.batch_normalization(x=resnet_32_short_cut_43rg_conv0, mean=resnet_32_short_cut_43rg_batch_normalize_28cq_mean, variance=resnet_32_short_cut_43rg_batch_normalize_28cq_variance, offset=resnet_32_short_cut_43rg_batch_normalize_28cq_offset, scale=resnet_32_short_cut_43rg_batch_normalize_28cq_scale, variance_epsilon=0.001, name='resnet_32_short_cut_43rg/batch_normalize_28cq')
    resnet_32_short_cut_43rg_relu_30eg = tf.nn.relu(features=resnet_32_short_cut_43rg_batch_normalize_28cq, name='resnet_32_short_cut_43rg/relu_30eg')
    resnet_32_short_cut_43rg_conv_32gw_filters_init = tf.keras.initializers.glorot_uniform(seed=1)((3, 3, resnet_32_short_cut_43rg_relu_30eg.get_shape().as_list()[-1], 32))
    resnet_32_short_cut_43rg_conv_32gw_filters = tf.Variable(initial_value=resnet_32_short_cut_43rg_conv_32gw_filters_init, trainable=True, caching_device=None, name='resnet_32_short_cut_43rg/conv_32gw/filters', variable_def=None, dtype=tf_types['float32'], import_scope=None, constraint=None, synchronization=tf.VariableSynchronization.AUTO, shape=None)
    trainable_params['resnet_32_short_cut_43rg/conv_32gw/filters'] = resnet_32_short_cut_43rg_conv_32gw_filters
    resnet_32_short_cut_43rg_conv_32gw = tf.nn.conv2d(input=resnet_32_short_cut_43rg_relu_30eg, filters=resnet_32_short_cut_43rg_conv_32gw_filters, strides=[1, 1], padding='SAME', data_format='NHWC', dilations=1, use_bias=False, name='resnet_32_short_cut_43rg/conv_32gw')
    resnet_32_short_cut_43rg_batch_normalize_34im_mean_init = tf.zeros_initializer()((resnet_32_short_cut_43rg_conv_32gw.get_shape().as_list()[-1], ))
    resnet_32_short_cut_43rg_batch_normalize_34im_mean = tf.Variable(initial_value=resnet_32_short_cut_43rg_batch_normalize_34im_mean_init, trainable=True, caching_device=None, name='resnet_32_short_cut_43rg/batch_normalize_34im/mean', variable_def=None, dtype=tf_types['float32'], import_scope=None, constraint=None, synchronization=tf.VariableSynchronization.AUTO, shape=None)
    trainable_params['resnet_32_short_cut_43rg/batch_normalize_34im/mean'] = resnet_32_short_cut_43rg_batch_normalize_34im_mean
    resnet_32_short_cut_43rg_batch_normalize_34im_variance_init = tf.ones_initializer()((resnet_32_short_cut_43rg_conv_32gw.get_shape().as_list()[-1], ))
    resnet_32_short_cut_43rg_batch_normalize_34im_variance = tf.Variable(initial_value=resnet_32_short_cut_43rg_batch_normalize_34im_variance_init, trainable=True, caching_device=None, name='resnet_32_short_cut_43rg/batch_normalize_34im/variance', variable_def=None, dtype=tf_types['float32'], import_scope=None, constraint=None, synchronization=tf.VariableSynchronization.AUTO, shape=None)
    trainable_params['resnet_32_short_cut_43rg/batch_normalize_34im/variance'] = resnet_32_short_cut_43rg_batch_normalize_34im_variance
    resnet_32_short_cut_43rg_batch_normalize_34im_offset_init = tf.zeros_initializer()((resnet_32_short_cut_43rg_conv_32gw.get_shape().as_list()[-1], ))
    resnet_32_short_cut_43rg_batch_normalize_34im_offset = tf.Variable(initial_value=resnet_32_short_cut_43rg_batch_normalize_34im_offset_init, trainable=True, caching_device=None, name='resnet_32_short_cut_43rg/batch_normalize_34im/offset', variable_def=None, dtype=tf_types['float32'], import_scope=None, constraint=None, synchronization=tf.VariableSynchronization.AUTO, shape=None)
    trainable_params['resnet_32_short_cut_43rg/batch_normalize_34im/offset'] = resnet_32_short_cut_43rg_batch_normalize_34im_offset
    resnet_32_short_cut_43rg_batch_normalize_34im_scale_init = tf.ones_initializer()((resnet_32_short_cut_43rg_conv_32gw.get_shape().as_list()[-1], ))
    resnet_32_short_cut_43rg_batch_normalize_34im_scale = tf.Variable(initial_value=resnet_32_short_cut_43rg_batch_normalize_34im_scale_init, trainable=True, caching_device=None, name='resnet_32_short_cut_43rg/batch_normalize_34im/scale', variable_def=None, dtype=tf_types['float32'], import_scope=None, constraint=None, synchronization=tf.VariableSynchronization.AUTO, shape=None)
    trainable_params['resnet_32_short_cut_43rg/batch_normalize_34im/scale'] = resnet_32_short_cut_43rg_batch_normalize_34im_scale
    resnet_32_short_cut_43rg_batch_normalize_34im = tf.nn.batch_normalization(x=resnet_32_short_cut_43rg_conv_32gw, mean=resnet_32_short_cut_43rg_batch_normalize_34im_mean, variance=resnet_32_short_cut_43rg_batch_normalize_34im_variance, offset=resnet_32_short_cut_43rg_batch_normalize_34im_offset, scale=resnet_32_short_cut_43rg_batch_normalize_34im_scale, variance_epsilon=0.001, name='resnet_32_short_cut_43rg/batch_normalize_34im')
    resnet_32_short_cut_43rg_conv1 = tf.nn.relu(features=resnet_32_short_cut_43rg_batch_normalize_34im, name='resnet_32_short_cut_43rg/conv1')
    resnet_32_short_cut_43rg_conv_38ms_filters_init = tf.keras.initializers.glorot_uniform(seed=1)((1, 1, resnet_16_24yk_relu_23xc.get_shape().as_list()[-1], 32))
    resnet_32_short_cut_43rg_conv_38ms_filters = tf.Variable(initial_value=resnet_32_short_cut_43rg_conv_38ms_filters_init, trainable=True, caching_device=None, name='resnet_32_short_cut_43rg/conv_38ms/filters', variable_def=None, dtype=tf_types['float32'], import_scope=None, constraint=None, synchronization=tf.VariableSynchronization.AUTO, shape=None)
    trainable_params['resnet_32_short_cut_43rg/conv_38ms/filters'] = resnet_32_short_cut_43rg_conv_38ms_filters
    resnet_32_short_cut_43rg_conv_38ms = tf.nn.conv2d(input=resnet_16_24yk_relu_23xc, filters=resnet_32_short_cut_43rg_conv_38ms_filters, strides=[2, 2], padding='VALID', data_format='NHWC', dilations=1, use_bias=False, name='resnet_32_short_cut_43rg/conv_38ms')
    resnet_32_short_cut_43rg_short_cut_16_32_mean_init = tf.zeros_initializer()((resnet_32_short_cut_43rg_conv_38ms.get_shape().as_list()[-1], ))
    resnet_32_short_cut_43rg_short_cut_16_32_mean = tf.Variable(initial_value=resnet_32_short_cut_43rg_short_cut_16_32_mean_init, trainable=True, caching_device=None, name='resnet_32_short_cut_43rg/short_cut_16_32/mean', variable_def=None, dtype=tf_types['float32'], import_scope=None, constraint=None, synchronization=tf.VariableSynchronization.AUTO, shape=None)
    trainable_params['resnet_32_short_cut_43rg/short_cut_16_32/mean'] = resnet_32_short_cut_43rg_short_cut_16_32_mean
    resnet_32_short_cut_43rg_short_cut_16_32_variance_init = tf.ones_initializer()((resnet_32_short_cut_43rg_conv_38ms.get_shape().as_list()[-1], ))
    resnet_32_short_cut_43rg_short_cut_16_32_variance = tf.Variable(initial_value=resnet_32_short_cut_43rg_short_cut_16_32_variance_init, trainable=True, caching_device=None, name='resnet_32_short_cut_43rg/short_cut_16_32/variance', variable_def=None, dtype=tf_types['float32'], import_scope=None, constraint=None, synchronization=tf.VariableSynchronization.AUTO, shape=None)
    trainable_params['resnet_32_short_cut_43rg/short_cut_16_32/variance'] = resnet_32_short_cut_43rg_short_cut_16_32_variance
    resnet_32_short_cut_43rg_short_cut_16_32_offset_init = tf.zeros_initializer()((resnet_32_short_cut_43rg_conv_38ms.get_shape().as_list()[-1], ))
    resnet_32_short_cut_43rg_short_cut_16_32_offset = tf.Variable(initial_value=resnet_32_short_cut_43rg_short_cut_16_32_offset_init, trainable=True, caching_device=None, name='resnet_32_short_cut_43rg/short_cut_16_32/offset', variable_def=None, dtype=tf_types['float32'], import_scope=None, constraint=None, synchronization=tf.VariableSynchronization.AUTO, shape=None)
    trainable_params['resnet_32_short_cut_43rg/short_cut_16_32/offset'] = resnet_32_short_cut_43rg_short_cut_16_32_offset
    resnet_32_short_cut_43rg_short_cut_16_32_scale_init = tf.ones_initializer()((resnet_32_short_cut_43rg_conv_38ms.get_shape().as_list()[-1], ))
    resnet_32_short_cut_43rg_short_cut_16_32_scale = tf.Variable(initial_value=resnet_32_short_cut_43rg_short_cut_16_32_scale_init, trainable=True, caching_device=None, name='resnet_32_short_cut_43rg/short_cut_16_32/scale', variable_def=None, dtype=tf_types['float32'], import_scope=None, constraint=None, synchronization=tf.VariableSynchronization.AUTO, shape=None)
    trainable_params['resnet_32_short_cut_43rg/short_cut_16_32/scale'] = resnet_32_short_cut_43rg_short_cut_16_32_scale
    resnet_32_short_cut_43rg_short_cut_16_32 = tf.nn.batch_normalization(x=resnet_32_short_cut_43rg_conv_38ms, mean=resnet_32_short_cut_43rg_short_cut_16_32_mean, variance=resnet_32_short_cut_43rg_short_cut_16_32_variance, offset=resnet_32_short_cut_43rg_short_cut_16_32_offset, scale=resnet_32_short_cut_43rg_short_cut_16_32_scale, variance_epsilon=0.001, name='resnet_32_short_cut_43rg/short_cut_16_32')
    resnet_32_short_cut_43rg_add_42qy = tf.math.add(x=[resnet_32_short_cut_43rg_short_cut_16_32, resnet_32_short_cut_43rg_conv1][0], y=[resnet_32_short_cut_43rg_short_cut_16_32, resnet_32_short_cut_43rg_conv1][1], name='resnet_32_short_cut_43rg/add_42qy')
    resnet_32_58gw_conv_45tw_filters_init = tf.keras.initializers.glorot_uniform(seed=1)((3, 3, resnet_32_short_cut_43rg_add_42qy.get_shape().as_list()[-1], 32))
    resnet_32_58gw_conv_45tw_filters = tf.Variable(initial_value=resnet_32_58gw_conv_45tw_filters_init, trainable=True, caching_device=None, name='resnet_32_58gw/conv_45tw/filters', variable_def=None, dtype=tf_types['float32'], import_scope=None, constraint=None, synchronization=tf.VariableSynchronization.AUTO, shape=None)
    trainable_params['resnet_32_58gw/conv_45tw/filters'] = resnet_32_58gw_conv_45tw_filters
    resnet_32_58gw_conv_45tw = tf.nn.conv2d(input=resnet_32_short_cut_43rg_add_42qy, filters=resnet_32_58gw_conv_45tw_filters, strides=[1, 1], padding='SAME', data_format='NHWC', dilations=1, use_bias=False, name='resnet_32_58gw/conv_45tw')
    resnet_32_58gw_batch_normalize_47vm_mean_init = tf.zeros_initializer()((resnet_32_58gw_conv_45tw.get_shape().as_list()[-1], ))
    resnet_32_58gw_batch_normalize_47vm_mean = tf.Variable(initial_value=resnet_32_58gw_batch_normalize_47vm_mean_init, trainable=True, caching_device=None, name='resnet_32_58gw/batch_normalize_47vm/mean', variable_def=None, dtype=tf_types['float32'], import_scope=None, constraint=None, synchronization=tf.VariableSynchronization.AUTO, shape=None)
    trainable_params['resnet_32_58gw/batch_normalize_47vm/mean'] = resnet_32_58gw_batch_normalize_47vm_mean
    resnet_32_58gw_batch_normalize_47vm_variance_init = tf.ones_initializer()((resnet_32_58gw_conv_45tw.get_shape().as_list()[-1], ))
    resnet_32_58gw_batch_normalize_47vm_variance = tf.Variable(initial_value=resnet_32_58gw_batch_normalize_47vm_variance_init, trainable=True, caching_device=None, name='resnet_32_58gw/batch_normalize_47vm/variance', variable_def=None, dtype=tf_types['float32'], import_scope=None, constraint=None, synchronization=tf.VariableSynchronization.AUTO, shape=None)
    trainable_params['resnet_32_58gw/batch_normalize_47vm/variance'] = resnet_32_58gw_batch_normalize_47vm_variance
    resnet_32_58gw_batch_normalize_47vm_offset_init = tf.zeros_initializer()((resnet_32_58gw_conv_45tw.get_shape().as_list()[-1], ))
    resnet_32_58gw_batch_normalize_47vm_offset = tf.Variable(initial_value=resnet_32_58gw_batch_normalize_47vm_offset_init, trainable=True, caching_device=None, name='resnet_32_58gw/batch_normalize_47vm/offset', variable_def=None, dtype=tf_types['float32'], import_scope=None, constraint=None, synchronization=tf.VariableSynchronization.AUTO, shape=None)
    trainable_params['resnet_32_58gw/batch_normalize_47vm/offset'] = resnet_32_58gw_batch_normalize_47vm_offset
    resnet_32_58gw_batch_normalize_47vm_scale_init = tf.ones_initializer()((resnet_32_58gw_conv_45tw.get_shape().as_list()[-1], ))
    resnet_32_58gw_batch_normalize_47vm_scale = tf.Variable(initial_value=resnet_32_58gw_batch_normalize_47vm_scale_init, trainable=True, caching_device=None, name='resnet_32_58gw/batch_normalize_47vm/scale', variable_def=None, dtype=tf_types['float32'], import_scope=None, constraint=None, synchronization=tf.VariableSynchronization.AUTO, shape=None)
    trainable_params['resnet_32_58gw/batch_normalize_47vm/scale'] = resnet_32_58gw_batch_normalize_47vm_scale
    resnet_32_58gw_batch_normalize_47vm = tf.nn.batch_normalization(x=resnet_32_58gw_conv_45tw, mean=resnet_32_58gw_batch_normalize_47vm_mean, variance=resnet_32_58gw_batch_normalize_47vm_variance, offset=resnet_32_58gw_batch_normalize_47vm_offset, scale=resnet_32_58gw_batch_normalize_47vm_scale, variance_epsilon=0.001, name='resnet_32_58gw/batch_normalize_47vm')
    resnet_32_58gw_relu_49xc = tf.nn.relu(features=resnet_32_58gw_batch_normalize_47vm, name='resnet_32_58gw/relu_49xc')
    resnet_32_58gw_conv_51zs_filters_init = tf.keras.initializers.glorot_uniform(seed=1)((3, 3, resnet_32_58gw_relu_49xc.get_shape().as_list()[-1], 32))
    resnet_32_58gw_conv_51zs_filters = tf.Variable(initial_value=resnet_32_58gw_conv_51zs_filters_init, trainable=True, caching_device=None, name='resnet_32_58gw/conv_51zs/filters', variable_def=None, dtype=tf_types['float32'], import_scope=None, constraint=None, synchronization=tf.VariableSynchronization.AUTO, shape=None)
    trainable_params['resnet_32_58gw/conv_51zs/filters'] = resnet_32_58gw_conv_51zs_filters
    resnet_32_58gw_conv_51zs = tf.nn.conv2d(input=resnet_32_58gw_relu_49xc, filters=resnet_32_58gw_conv_51zs_filters, strides=[1, 1], padding='SAME', data_format='NHWC', dilations=1, use_bias=False, name='resnet_32_58gw/conv_51zs')
    resnet_32_58gw_conv_mean_init = tf.zeros_initializer()((resnet_32_58gw_conv_51zs.get_shape().as_list()[-1], ))
    resnet_32_58gw_conv_mean = tf.Variable(initial_value=resnet_32_58gw_conv_mean_init, trainable=True, caching_device=None, name='resnet_32_58gw/conv/mean', variable_def=None, dtype=tf_types['float32'], import_scope=None, constraint=None, synchronization=tf.VariableSynchronization.AUTO, shape=None)
    trainable_params['resnet_32_58gw/conv/mean'] = resnet_32_58gw_conv_mean
    resnet_32_58gw_conv_variance_init = tf.ones_initializer()((resnet_32_58gw_conv_51zs.get_shape().as_list()[-1], ))
    resnet_32_58gw_conv_variance = tf.Variable(initial_value=resnet_32_58gw_conv_variance_init, trainable=True, caching_device=None, name='resnet_32_58gw/conv/variance', variable_def=None, dtype=tf_types['float32'], import_scope=None, constraint=None, synchronization=tf.VariableSynchronization.AUTO, shape=None)
    trainable_params['resnet_32_58gw/conv/variance'] = resnet_32_58gw_conv_variance
    resnet_32_58gw_conv_offset_init = tf.zeros_initializer()((resnet_32_58gw_conv_51zs.get_shape().as_list()[-1], ))
    resnet_32_58gw_conv_offset = tf.Variable(initial_value=resnet_32_58gw_conv_offset_init, trainable=True, caching_device=None, name='resnet_32_58gw/conv/offset', variable_def=None, dtype=tf_types['float32'], import_scope=None, constraint=None, synchronization=tf.VariableSynchronization.AUTO, shape=None)
    trainable_params['resnet_32_58gw/conv/offset'] = resnet_32_58gw_conv_offset
    resnet_32_58gw_conv_scale_init = tf.ones_initializer()((resnet_32_58gw_conv_51zs.get_shape().as_list()[-1], ))
    resnet_32_58gw_conv_scale = tf.Variable(initial_value=resnet_32_58gw_conv_scale_init, trainable=True, caching_device=None, name='resnet_32_58gw/conv/scale', variable_def=None, dtype=tf_types['float32'], import_scope=None, constraint=None, synchronization=tf.VariableSynchronization.AUTO, shape=None)
    trainable_params['resnet_32_58gw/conv/scale'] = resnet_32_58gw_conv_scale
    resnet_32_58gw_conv = tf.nn.batch_normalization(x=resnet_32_58gw_conv_51zs, mean=resnet_32_58gw_conv_mean, variance=resnet_32_58gw_conv_variance, offset=resnet_32_58gw_conv_offset, scale=resnet_32_58gw_conv_scale, variance_epsilon=0.001, name='resnet_32_58gw/conv')
    resnet_32_58gw_add_55dy = tf.math.add(x=[resnet_32_short_cut_43rg_add_42qy, resnet_32_58gw_conv][0], y=[resnet_32_short_cut_43rg_add_42qy, resnet_32_58gw_conv][1], name='resnet_32_58gw/add_55dy')
    resnet_32_58gw_relu_57fo = tf.nn.relu(features=resnet_32_58gw_add_55dy, name='resnet_32_58gw/relu_57fo')
    resnet_64_short_cut_77zs_conv0_filters_init = tf.keras.initializers.glorot_uniform(seed=1)((3, 3, resnet_32_58gw_relu_57fo.get_shape().as_list()[-1], 64))
    resnet_64_short_cut_77zs_conv0_filters = tf.Variable(initial_value=resnet_64_short_cut_77zs_conv0_filters_init, trainable=True, caching_device=None, name='resnet_64_short_cut_77zs/conv0/filters', variable_def=None, dtype=tf_types['float32'], import_scope=None, constraint=None, synchronization=tf.VariableSynchronization.AUTO, shape=None)
    trainable_params['resnet_64_short_cut_77zs/conv0/filters'] = resnet_64_short_cut_77zs_conv0_filters
    resnet_64_short_cut_77zs_conv0 = tf.nn.conv2d(input=resnet_32_58gw_relu_57fo, filters=resnet_64_short_cut_77zs_conv0_filters, strides=[2, 2], padding='SAME', data_format='NHWC', dilations=1, use_bias=False, name='resnet_64_short_cut_77zs/conv0')
    resnet_64_short_cut_77zs_batch_normalize_62kc_mean_init = tf.zeros_initializer()((resnet_64_short_cut_77zs_conv0.get_shape().as_list()[-1], ))
    resnet_64_short_cut_77zs_batch_normalize_62kc_mean = tf.Variable(initial_value=resnet_64_short_cut_77zs_batch_normalize_62kc_mean_init, trainable=True, caching_device=None, name='resnet_64_short_cut_77zs/batch_normalize_62kc/mean', variable_def=None, dtype=tf_types['float32'], import_scope=None, constraint=None, synchronization=tf.VariableSynchronization.AUTO, shape=None)
    trainable_params['resnet_64_short_cut_77zs/batch_normalize_62kc/mean'] = resnet_64_short_cut_77zs_batch_normalize_62kc_mean
    resnet_64_short_cut_77zs_batch_normalize_62kc_variance_init = tf.ones_initializer()((resnet_64_short_cut_77zs_conv0.get_shape().as_list()[-1], ))
    resnet_64_short_cut_77zs_batch_normalize_62kc_variance = tf.Variable(initial_value=resnet_64_short_cut_77zs_batch_normalize_62kc_variance_init, trainable=True, caching_device=None, name='resnet_64_short_cut_77zs/batch_normalize_62kc/variance', variable_def=None, dtype=tf_types['float32'], import_scope=None, constraint=None, synchronization=tf.VariableSynchronization.AUTO, shape=None)
    trainable_params['resnet_64_short_cut_77zs/batch_normalize_62kc/variance'] = resnet_64_short_cut_77zs_batch_normalize_62kc_variance
    resnet_64_short_cut_77zs_batch_normalize_62kc_offset_init = tf.zeros_initializer()((resnet_64_short_cut_77zs_conv0.get_shape().as_list()[-1], ))
    resnet_64_short_cut_77zs_batch_normalize_62kc_offset = tf.Variable(initial_value=resnet_64_short_cut_77zs_batch_normalize_62kc_offset_init, trainable=True, caching_device=None, name='resnet_64_short_cut_77zs/batch_normalize_62kc/offset', variable_def=None, dtype=tf_types['float32'], import_scope=None, constraint=None, synchronization=tf.VariableSynchronization.AUTO, shape=None)
    trainable_params['resnet_64_short_cut_77zs/batch_normalize_62kc/offset'] = resnet_64_short_cut_77zs_batch_normalize_62kc_offset
    resnet_64_short_cut_77zs_batch_normalize_62kc_scale_init = tf.ones_initializer()((resnet_64_short_cut_77zs_conv0.get_shape().as_list()[-1], ))
    resnet_64_short_cut_77zs_batch_normalize_62kc_scale = tf.Variable(initial_value=resnet_64_short_cut_77zs_batch_normalize_62kc_scale_init, trainable=True, caching_device=None, name='resnet_64_short_cut_77zs/batch_normalize_62kc/scale', variable_def=None, dtype=tf_types['float32'], import_scope=None, constraint=None, synchronization=tf.VariableSynchronization.AUTO, shape=None)
    trainable_params['resnet_64_short_cut_77zs/batch_normalize_62kc/scale'] = resnet_64_short_cut_77zs_batch_normalize_62kc_scale
    resnet_64_short_cut_77zs_batch_normalize_62kc = tf.nn.batch_normalization(x=resnet_64_short_cut_77zs_conv0, mean=resnet_64_short_cut_77zs_batch_normalize_62kc_mean, variance=resnet_64_short_cut_77zs_batch_normalize_62kc_variance, offset=resnet_64_short_cut_77zs_batch_normalize_62kc_offset, scale=resnet_64_short_cut_77zs_batch_normalize_62kc_scale, variance_epsilon=0.001, name='resnet_64_short_cut_77zs/batch_normalize_62kc')
    resnet_64_short_cut_77zs_relu_64ms = tf.nn.relu(features=resnet_64_short_cut_77zs_batch_normalize_62kc, name='resnet_64_short_cut_77zs/relu_64ms')
    resnet_64_short_cut_77zs_conv_66oi_filters_init = tf.keras.initializers.glorot_uniform(seed=1)((3, 3, resnet_64_short_cut_77zs_relu_64ms.get_shape().as_list()[-1], 64))
    resnet_64_short_cut_77zs_conv_66oi_filters = tf.Variable(initial_value=resnet_64_short_cut_77zs_conv_66oi_filters_init, trainable=True, caching_device=None, name='resnet_64_short_cut_77zs/conv_66oi/filters', variable_def=None, dtype=tf_types['float32'], import_scope=None, constraint=None, synchronization=tf.VariableSynchronization.AUTO, shape=None)
    trainable_params['resnet_64_short_cut_77zs/conv_66oi/filters'] = resnet_64_short_cut_77zs_conv_66oi_filters
    resnet_64_short_cut_77zs_conv_66oi = tf.nn.conv2d(input=resnet_64_short_cut_77zs_relu_64ms, filters=resnet_64_short_cut_77zs_conv_66oi_filters, strides=[1, 1], padding='SAME', data_format='NHWC', dilations=1, use_bias=False, name='resnet_64_short_cut_77zs/conv_66oi')
    resnet_64_short_cut_77zs_batch_normalize_68qy_mean_init = tf.zeros_initializer()((resnet_64_short_cut_77zs_conv_66oi.get_shape().as_list()[-1], ))
    resnet_64_short_cut_77zs_batch_normalize_68qy_mean = tf.Variable(initial_value=resnet_64_short_cut_77zs_batch_normalize_68qy_mean_init, trainable=True, caching_device=None, name='resnet_64_short_cut_77zs/batch_normalize_68qy/mean', variable_def=None, dtype=tf_types['float32'], import_scope=None, constraint=None, synchronization=tf.VariableSynchronization.AUTO, shape=None)
    trainable_params['resnet_64_short_cut_77zs/batch_normalize_68qy/mean'] = resnet_64_short_cut_77zs_batch_normalize_68qy_mean
    resnet_64_short_cut_77zs_batch_normalize_68qy_variance_init = tf.ones_initializer()((resnet_64_short_cut_77zs_conv_66oi.get_shape().as_list()[-1], ))
    resnet_64_short_cut_77zs_batch_normalize_68qy_variance = tf.Variable(initial_value=resnet_64_short_cut_77zs_batch_normalize_68qy_variance_init, trainable=True, caching_device=None, name='resnet_64_short_cut_77zs/batch_normalize_68qy/variance', variable_def=None, dtype=tf_types['float32'], import_scope=None, constraint=None, synchronization=tf.VariableSynchronization.AUTO, shape=None)
    trainable_params['resnet_64_short_cut_77zs/batch_normalize_68qy/variance'] = resnet_64_short_cut_77zs_batch_normalize_68qy_variance
    resnet_64_short_cut_77zs_batch_normalize_68qy_offset_init = tf.zeros_initializer()((resnet_64_short_cut_77zs_conv_66oi.get_shape().as_list()[-1], ))
    resnet_64_short_cut_77zs_batch_normalize_68qy_offset = tf.Variable(initial_value=resnet_64_short_cut_77zs_batch_normalize_68qy_offset_init, trainable=True, caching_device=None, name='resnet_64_short_cut_77zs/batch_normalize_68qy/offset', variable_def=None, dtype=tf_types['float32'], import_scope=None, constraint=None, synchronization=tf.VariableSynchronization.AUTO, shape=None)
    trainable_params['resnet_64_short_cut_77zs/batch_normalize_68qy/offset'] = resnet_64_short_cut_77zs_batch_normalize_68qy_offset
    resnet_64_short_cut_77zs_batch_normalize_68qy_scale_init = tf.ones_initializer()((resnet_64_short_cut_77zs_conv_66oi.get_shape().as_list()[-1], ))
    resnet_64_short_cut_77zs_batch_normalize_68qy_scale = tf.Variable(initial_value=resnet_64_short_cut_77zs_batch_normalize_68qy_scale_init, trainable=True, caching_device=None, name='resnet_64_short_cut_77zs/batch_normalize_68qy/scale', variable_def=None, dtype=tf_types['float32'], import_scope=None, constraint=None, synchronization=tf.VariableSynchronization.AUTO, shape=None)
    trainable_params['resnet_64_short_cut_77zs/batch_normalize_68qy/scale'] = resnet_64_short_cut_77zs_batch_normalize_68qy_scale
    resnet_64_short_cut_77zs_batch_normalize_68qy = tf.nn.batch_normalization(x=resnet_64_short_cut_77zs_conv_66oi, mean=resnet_64_short_cut_77zs_batch_normalize_68qy_mean, variance=resnet_64_short_cut_77zs_batch_normalize_68qy_variance, offset=resnet_64_short_cut_77zs_batch_normalize_68qy_offset, scale=resnet_64_short_cut_77zs_batch_normalize_68qy_scale, variance_epsilon=0.001, name='resnet_64_short_cut_77zs/batch_normalize_68qy')
    resnet_64_short_cut_77zs_conv1 = tf.nn.relu(features=resnet_64_short_cut_77zs_batch_normalize_68qy, name='resnet_64_short_cut_77zs/conv1')
    resnet_64_short_cut_77zs_conv_72ue_filters_init = tf.keras.initializers.glorot_uniform(seed=1)((1, 1, resnet_32_58gw_relu_57fo.get_shape().as_list()[-1], 64))
    resnet_64_short_cut_77zs_conv_72ue_filters = tf.Variable(initial_value=resnet_64_short_cut_77zs_conv_72ue_filters_init, trainable=True, caching_device=None, name='resnet_64_short_cut_77zs/conv_72ue/filters', variable_def=None, dtype=tf_types['float32'], import_scope=None, constraint=None, synchronization=tf.VariableSynchronization.AUTO, shape=None)
    trainable_params['resnet_64_short_cut_77zs/conv_72ue/filters'] = resnet_64_short_cut_77zs_conv_72ue_filters
    resnet_64_short_cut_77zs_conv_72ue = tf.nn.conv2d(input=resnet_32_58gw_relu_57fo, filters=resnet_64_short_cut_77zs_conv_72ue_filters, strides=[2, 2], padding='VALID', data_format='NHWC', dilations=1, use_bias=False, name='resnet_64_short_cut_77zs/conv_72ue')
    resnet_64_short_cut_77zs_short_cut_32_64_mean_init = tf.zeros_initializer()((resnet_64_short_cut_77zs_conv_72ue.get_shape().as_list()[-1], ))
    resnet_64_short_cut_77zs_short_cut_32_64_mean = tf.Variable(initial_value=resnet_64_short_cut_77zs_short_cut_32_64_mean_init, trainable=True, caching_device=None, name='resnet_64_short_cut_77zs/short_cut_32_64/mean', variable_def=None, dtype=tf_types['float32'], import_scope=None, constraint=None, synchronization=tf.VariableSynchronization.AUTO, shape=None)
    trainable_params['resnet_64_short_cut_77zs/short_cut_32_64/mean'] = resnet_64_short_cut_77zs_short_cut_32_64_mean
    resnet_64_short_cut_77zs_short_cut_32_64_variance_init = tf.ones_initializer()((resnet_64_short_cut_77zs_conv_72ue.get_shape().as_list()[-1], ))
    resnet_64_short_cut_77zs_short_cut_32_64_variance = tf.Variable(initial_value=resnet_64_short_cut_77zs_short_cut_32_64_variance_init, trainable=True, caching_device=None, name='resnet_64_short_cut_77zs/short_cut_32_64/variance', variable_def=None, dtype=tf_types['float32'], import_scope=None, constraint=None, synchronization=tf.VariableSynchronization.AUTO, shape=None)
    trainable_params['resnet_64_short_cut_77zs/short_cut_32_64/variance'] = resnet_64_short_cut_77zs_short_cut_32_64_variance
    resnet_64_short_cut_77zs_short_cut_32_64_offset_init = tf.zeros_initializer()((resnet_64_short_cut_77zs_conv_72ue.get_shape().as_list()[-1], ))
    resnet_64_short_cut_77zs_short_cut_32_64_offset = tf.Variable(initial_value=resnet_64_short_cut_77zs_short_cut_32_64_offset_init, trainable=True, caching_device=None, name='resnet_64_short_cut_77zs/short_cut_32_64/offset', variable_def=None, dtype=tf_types['float32'], import_scope=None, constraint=None, synchronization=tf.VariableSynchronization.AUTO, shape=None)
    trainable_params['resnet_64_short_cut_77zs/short_cut_32_64/offset'] = resnet_64_short_cut_77zs_short_cut_32_64_offset
    resnet_64_short_cut_77zs_short_cut_32_64_scale_init = tf.ones_initializer()((resnet_64_short_cut_77zs_conv_72ue.get_shape().as_list()[-1], ))
    resnet_64_short_cut_77zs_short_cut_32_64_scale = tf.Variable(initial_value=resnet_64_short_cut_77zs_short_cut_32_64_scale_init, trainable=True, caching_device=None, name='resnet_64_short_cut_77zs/short_cut_32_64/scale', variable_def=None, dtype=tf_types['float32'], import_scope=None, constraint=None, synchronization=tf.VariableSynchronization.AUTO, shape=None)
    trainable_params['resnet_64_short_cut_77zs/short_cut_32_64/scale'] = resnet_64_short_cut_77zs_short_cut_32_64_scale
    resnet_64_short_cut_77zs_short_cut_32_64 = tf.nn.batch_normalization(x=resnet_64_short_cut_77zs_conv_72ue, mean=resnet_64_short_cut_77zs_short_cut_32_64_mean, variance=resnet_64_short_cut_77zs_short_cut_32_64_variance, offset=resnet_64_short_cut_77zs_short_cut_32_64_offset, scale=resnet_64_short_cut_77zs_short_cut_32_64_scale, variance_epsilon=0.001, name='resnet_64_short_cut_77zs/short_cut_32_64')
    resnet_64_short_cut_77zs_add_76yk = tf.math.add(x=[resnet_64_short_cut_77zs_short_cut_32_64, resnet_64_short_cut_77zs_conv1][0], y=[resnet_64_short_cut_77zs_short_cut_32_64, resnet_64_short_cut_77zs_conv1][1], name='resnet_64_short_cut_77zs/add_76yk')
    features_conv_79bi_filters_init = tf.keras.initializers.glorot_uniform(seed=1)((3, 3, resnet_64_short_cut_77zs_add_76yk.get_shape().as_list()[-1], 64))
    features_conv_79bi_filters = tf.Variable(initial_value=features_conv_79bi_filters_init, trainable=True, caching_device=None, name='features/conv_79bi/filters', variable_def=None, dtype=tf_types['float32'], import_scope=None, constraint=None, synchronization=tf.VariableSynchronization.AUTO, shape=None)
    trainable_params['features/conv_79bi/filters'] = features_conv_79bi_filters
    features_conv_79bi = tf.nn.conv2d(input=resnet_64_short_cut_77zs_add_76yk, filters=features_conv_79bi_filters, strides=[1, 1], padding='SAME', data_format='NHWC', dilations=1, use_bias=False, name='features/conv_79bi')
    features_batch_normalize_81dy_mean_init = tf.zeros_initializer()((features_conv_79bi.get_shape().as_list()[-1], ))
    features_batch_normalize_81dy_mean = tf.Variable(initial_value=features_batch_normalize_81dy_mean_init, trainable=True, caching_device=None, name='features/batch_normalize_81dy/mean', variable_def=None, dtype=tf_types['float32'], import_scope=None, constraint=None, synchronization=tf.VariableSynchronization.AUTO, shape=None)
    trainable_params['features/batch_normalize_81dy/mean'] = features_batch_normalize_81dy_mean
    features_batch_normalize_81dy_variance_init = tf.ones_initializer()((features_conv_79bi.get_shape().as_list()[-1], ))
    features_batch_normalize_81dy_variance = tf.Variable(initial_value=features_batch_normalize_81dy_variance_init, trainable=True, caching_device=None, name='features/batch_normalize_81dy/variance', variable_def=None, dtype=tf_types['float32'], import_scope=None, constraint=None, synchronization=tf.VariableSynchronization.AUTO, shape=None)
    trainable_params['features/batch_normalize_81dy/variance'] = features_batch_normalize_81dy_variance
    features_batch_normalize_81dy_offset_init = tf.zeros_initializer()((features_conv_79bi.get_shape().as_list()[-1], ))
    features_batch_normalize_81dy_offset = tf.Variable(initial_value=features_batch_normalize_81dy_offset_init, trainable=True, caching_device=None, name='features/batch_normalize_81dy/offset', variable_def=None, dtype=tf_types['float32'], import_scope=None, constraint=None, synchronization=tf.VariableSynchronization.AUTO, shape=None)
    trainable_params['features/batch_normalize_81dy/offset'] = features_batch_normalize_81dy_offset
    features_batch_normalize_81dy_scale_init = tf.ones_initializer()((features_conv_79bi.get_shape().as_list()[-1], ))
    features_batch_normalize_81dy_scale = tf.Variable(initial_value=features_batch_normalize_81dy_scale_init, trainable=True, caching_device=None, name='features/batch_normalize_81dy/scale', variable_def=None, dtype=tf_types['float32'], import_scope=None, constraint=None, synchronization=tf.VariableSynchronization.AUTO, shape=None)
    trainable_params['features/batch_normalize_81dy/scale'] = features_batch_normalize_81dy_scale
    features_batch_normalize_81dy = tf.nn.batch_normalization(x=features_conv_79bi, mean=features_batch_normalize_81dy_mean, variance=features_batch_normalize_81dy_variance, offset=features_batch_normalize_81dy_offset, scale=features_batch_normalize_81dy_scale, variance_epsilon=0.001, name='features/batch_normalize_81dy')
    features_relu_83fo = tf.nn.relu(features=features_batch_normalize_81dy, name='features/relu_83fo')
    features_conv_85he_filters_init = tf.keras.initializers.glorot_uniform(seed=1)((3, 3, features_relu_83fo.get_shape().as_list()[-1], 64))
    features_conv_85he_filters = tf.Variable(initial_value=features_conv_85he_filters_init, trainable=True, caching_device=None, name='features/conv_85he/filters', variable_def=None, dtype=tf_types['float32'], import_scope=None, constraint=None, synchronization=tf.VariableSynchronization.AUTO, shape=None)
    trainable_params['features/conv_85he/filters'] = features_conv_85he_filters
    features_conv_85he = tf.nn.conv2d(input=features_relu_83fo, filters=features_conv_85he_filters, strides=[1, 1], padding='SAME', data_format='NHWC', dilations=1, use_bias=False, name='features/conv_85he')
    features_conv_mean_init = tf.zeros_initializer()((features_conv_85he.get_shape().as_list()[-1], ))
    features_conv_mean = tf.Variable(initial_value=features_conv_mean_init, trainable=True, caching_device=None, name='features/conv/mean', variable_def=None, dtype=tf_types['float32'], import_scope=None, constraint=None, synchronization=tf.VariableSynchronization.AUTO, shape=None)
    trainable_params['features/conv/mean'] = features_conv_mean
    features_conv_variance_init = tf.ones_initializer()((features_conv_85he.get_shape().as_list()[-1], ))
    features_conv_variance = tf.Variable(initial_value=features_conv_variance_init, trainable=True, caching_device=None, name='features/conv/variance', variable_def=None, dtype=tf_types['float32'], import_scope=None, constraint=None, synchronization=tf.VariableSynchronization.AUTO, shape=None)
    trainable_params['features/conv/variance'] = features_conv_variance
    features_conv_offset_init = tf.zeros_initializer()((features_conv_85he.get_shape().as_list()[-1], ))
    features_conv_offset = tf.Variable(initial_value=features_conv_offset_init, trainable=True, caching_device=None, name='features/conv/offset', variable_def=None, dtype=tf_types['float32'], import_scope=None, constraint=None, synchronization=tf.VariableSynchronization.AUTO, shape=None)
    trainable_params['features/conv/offset'] = features_conv_offset
    features_conv_scale_init = tf.ones_initializer()((features_conv_85he.get_shape().as_list()[-1], ))
    features_conv_scale = tf.Variable(initial_value=features_conv_scale_init, trainable=True, caching_device=None, name='features/conv/scale', variable_def=None, dtype=tf_types['float32'], import_scope=None, constraint=None, synchronization=tf.VariableSynchronization.AUTO, shape=None)
    trainable_params['features/conv/scale'] = features_conv_scale
    features_conv = tf.nn.batch_normalization(x=features_conv_85he, mean=features_conv_mean, variance=features_conv_variance, offset=features_conv_offset, scale=features_conv_scale, variance_epsilon=0.001, name='features/conv')
    features_add_89lk = tf.math.add(x=[resnet_64_short_cut_77zs_add_76yk, features_conv][0], y=[resnet_64_short_cut_77zs_add_76yk, features_conv][1], name='features/add_89lk')
    features_relu_91na = tf.nn.relu(features=features_add_89lk, name='features/relu_91na')
    max_pool2d_94qy = tf.nn.max_pool(input=features_relu_91na, ksize=[3, 3], strides=[2, 2], padding='VALID', data_format='NHWC', name='max_pool2d_94qy')
    flatten_96so = tf.reshape(tensor=max_pool2d_94qy, shape=(-1, np.asarray(max_pool2d_94qy.get_shape().as_list()[1:]).prod()), name='flatten_96so')
    dense_98ue_weights_init = tf.keras.initializers.glorot_uniform(seed=2)((flatten_96so.get_shape().as_list()[-1], 10))
    dense_98ue_weights = tf.Variable(initial_value=dense_98ue_weights_init, trainable=True, caching_device=None, name='dense_98ue/weights', variable_def=None, dtype=tf_types['float32'], import_scope=None, constraint=None, synchronization=tf.VariableSynchronization.AUTO, shape=None)
    trainable_params['dense_98ue/weights'] = dense_98ue_weights
    dense_98ue_bias_init = tf.zeros_initializer()((1, ))
    dense_98ue_bias = tf.Variable(initial_value=dense_98ue_bias_init, trainable=True, caching_device=None, name='dense_98ue/bias', variable_def=None, dtype=tf_types['float32'], import_scope=None, constraint=None, synchronization=tf.VariableSynchronization.AUTO, shape=None)
    trainable_params['dense_98ue/bias'] = dense_98ue_bias
    return trainable_params


def model(input_data, trainable_params, training):
    conv_5fo = tf.nn.conv2d(input=input_data, filters=trainable_params['conv_5fo/filters'], strides=[1, 1], padding='SAME', data_format='NHWC', dilations=1, use_bias=False, name='conv_5fo')
    batch_normalize_7he = tf.nn.batch_normalization(x=conv_5fo, mean=trainable_params['batch_normalize_7he/mean'], variance=trainable_params['batch_normalize_7he/variance'], offset=trainable_params['batch_normalize_7he/offset'], scale=trainable_params['batch_normalize_7he/scale'], variance_epsilon=0.001, name='batch_normalize_7he')
    relu_9ju = tf.nn.relu(features=batch_normalize_7he, name='relu_9ju')
    resnet_16_24yk_conv_11lk = tf.nn.conv2d(input=relu_9ju, filters=trainable_params['resnet_16_24yk/conv_11lk/filters'], strides=[1, 1], padding='SAME', data_format='NHWC', dilations=1, use_bias=False, name='resnet_16_24yk/conv_11lk')
    resnet_16_24yk_batch_normalize_13na = tf.nn.batch_normalization(x=resnet_16_24yk_conv_11lk, mean=trainable_params['resnet_16_24yk/batch_normalize_13na/mean'], variance=trainable_params['resnet_16_24yk/batch_normalize_13na/variance'], offset=trainable_params['resnet_16_24yk/batch_normalize_13na/offset'], scale=trainable_params['resnet_16_24yk/batch_normalize_13na/scale'], variance_epsilon=0.001, name='resnet_16_24yk/batch_normalize_13na')
    resnet_16_24yk_relu_15pq = tf.nn.relu(features=resnet_16_24yk_batch_normalize_13na, name='resnet_16_24yk/relu_15pq')
    resnet_16_24yk_conv_17rg = tf.nn.conv2d(input=resnet_16_24yk_relu_15pq, filters=trainable_params['resnet_16_24yk/conv_17rg/filters'], strides=[1, 1], padding='SAME', data_format='NHWC', dilations=1, use_bias=False, name='resnet_16_24yk/conv_17rg')
    resnet_16_24yk_conv = tf.nn.batch_normalization(x=resnet_16_24yk_conv_17rg, mean=trainable_params['resnet_16_24yk/conv/mean'], variance=trainable_params['resnet_16_24yk/conv/variance'], offset=trainable_params['resnet_16_24yk/conv/offset'], scale=trainable_params['resnet_16_24yk/conv/scale'], variance_epsilon=0.001, name='resnet_16_24yk/conv')
    resnet_16_24yk_add_21vm = tf.math.add(x=[relu_9ju, resnet_16_24yk_conv][0], y=[relu_9ju, resnet_16_24yk_conv][1], name='resnet_16_24yk/add_21vm')
    resnet_16_24yk_relu_23xc = tf.nn.relu(features=resnet_16_24yk_add_21vm, name='resnet_16_24yk/relu_23xc')
    resnet_32_short_cut_43rg_conv0 = tf.nn.conv2d(input=resnet_16_24yk_relu_23xc, filters=trainable_params['resnet_32_short_cut_43rg/conv0/filters'], strides=[2, 2], padding='SAME', data_format='NHWC', dilations=1, use_bias=False, name='resnet_32_short_cut_43rg/conv0')
    resnet_32_short_cut_43rg_batch_normalize_28cq = tf.nn.batch_normalization(x=resnet_32_short_cut_43rg_conv0, mean=trainable_params['resnet_32_short_cut_43rg/batch_normalize_28cq/mean'], variance=trainable_params['resnet_32_short_cut_43rg/batch_normalize_28cq/variance'], offset=trainable_params['resnet_32_short_cut_43rg/batch_normalize_28cq/offset'], scale=trainable_params['resnet_32_short_cut_43rg/batch_normalize_28cq/scale'], variance_epsilon=0.001, name='resnet_32_short_cut_43rg/batch_normalize_28cq')
    resnet_32_short_cut_43rg_relu_30eg = tf.nn.relu(features=resnet_32_short_cut_43rg_batch_normalize_28cq, name='resnet_32_short_cut_43rg/relu_30eg')
    resnet_32_short_cut_43rg_conv_32gw = tf.nn.conv2d(input=resnet_32_short_cut_43rg_relu_30eg, filters=trainable_params['resnet_32_short_cut_43rg/conv_32gw/filters'], strides=[1, 1], padding='SAME', data_format='NHWC', dilations=1, use_bias=False, name='resnet_32_short_cut_43rg/conv_32gw')
    resnet_32_short_cut_43rg_batch_normalize_34im = tf.nn.batch_normalization(x=resnet_32_short_cut_43rg_conv_32gw, mean=trainable_params['resnet_32_short_cut_43rg/batch_normalize_34im/mean'], variance=trainable_params['resnet_32_short_cut_43rg/batch_normalize_34im/variance'], offset=trainable_params['resnet_32_short_cut_43rg/batch_normalize_34im/offset'], scale=trainable_params['resnet_32_short_cut_43rg/batch_normalize_34im/scale'], variance_epsilon=0.001, name='resnet_32_short_cut_43rg/batch_normalize_34im')
    resnet_32_short_cut_43rg_conv1 = tf.nn.relu(features=resnet_32_short_cut_43rg_batch_normalize_34im, name='resnet_32_short_cut_43rg/conv1')
    resnet_32_short_cut_43rg_conv_38ms = tf.nn.conv2d(input=resnet_16_24yk_relu_23xc, filters=trainable_params['resnet_32_short_cut_43rg/conv_38ms/filters'], strides=[2, 2], padding='VALID', data_format='NHWC', dilations=1, use_bias=False, name='resnet_32_short_cut_43rg/conv_38ms')
    resnet_32_short_cut_43rg_short_cut_16_32 = tf.nn.batch_normalization(x=resnet_32_short_cut_43rg_conv_38ms, mean=trainable_params['resnet_32_short_cut_43rg/short_cut_16_32/mean'], variance=trainable_params['resnet_32_short_cut_43rg/short_cut_16_32/variance'], offset=trainable_params['resnet_32_short_cut_43rg/short_cut_16_32/offset'], scale=trainable_params['resnet_32_short_cut_43rg/short_cut_16_32/scale'], variance_epsilon=0.001, name='resnet_32_short_cut_43rg/short_cut_16_32')
    resnet_32_short_cut_43rg_add_42qy = tf.math.add(x=[resnet_32_short_cut_43rg_short_cut_16_32, resnet_32_short_cut_43rg_conv1][0], y=[resnet_32_short_cut_43rg_short_cut_16_32, resnet_32_short_cut_43rg_conv1][1], name='resnet_32_short_cut_43rg/add_42qy')
    resnet_32_58gw_conv_45tw = tf.nn.conv2d(input=resnet_32_short_cut_43rg_add_42qy, filters=trainable_params['resnet_32_58gw/conv_45tw/filters'], strides=[1, 1], padding='SAME', data_format='NHWC', dilations=1, use_bias=False, name='resnet_32_58gw/conv_45tw')
    resnet_32_58gw_batch_normalize_47vm = tf.nn.batch_normalization(x=resnet_32_58gw_conv_45tw, mean=trainable_params['resnet_32_58gw/batch_normalize_47vm/mean'], variance=trainable_params['resnet_32_58gw/batch_normalize_47vm/variance'], offset=trainable_params['resnet_32_58gw/batch_normalize_47vm/offset'], scale=trainable_params['resnet_32_58gw/batch_normalize_47vm/scale'], variance_epsilon=0.001, name='resnet_32_58gw/batch_normalize_47vm')
    resnet_32_58gw_relu_49xc = tf.nn.relu(features=resnet_32_58gw_batch_normalize_47vm, name='resnet_32_58gw/relu_49xc')
    resnet_32_58gw_conv_51zs = tf.nn.conv2d(input=resnet_32_58gw_relu_49xc, filters=trainable_params['resnet_32_58gw/conv_51zs/filters'], strides=[1, 1], padding='SAME', data_format='NHWC', dilations=1, use_bias=False, name='resnet_32_58gw/conv_51zs')
    resnet_32_58gw_conv = tf.nn.batch_normalization(x=resnet_32_58gw_conv_51zs, mean=trainable_params['resnet_32_58gw/conv/mean'], variance=trainable_params['resnet_32_58gw/conv/variance'], offset=trainable_params['resnet_32_58gw/conv/offset'], scale=trainable_params['resnet_32_58gw/conv/scale'], variance_epsilon=0.001, name='resnet_32_58gw/conv')
    resnet_32_58gw_add_55dy = tf.math.add(x=[resnet_32_short_cut_43rg_add_42qy, resnet_32_58gw_conv][0], y=[resnet_32_short_cut_43rg_add_42qy, resnet_32_58gw_conv][1], name='resnet_32_58gw/add_55dy')
    resnet_32_58gw_relu_57fo = tf.nn.relu(features=resnet_32_58gw_add_55dy, name='resnet_32_58gw/relu_57fo')
    resnet_64_short_cut_77zs_conv0 = tf.nn.conv2d(input=resnet_32_58gw_relu_57fo, filters=trainable_params['resnet_64_short_cut_77zs/conv0/filters'], strides=[2, 2], padding='SAME', data_format='NHWC', dilations=1, use_bias=False, name='resnet_64_short_cut_77zs/conv0')
    resnet_64_short_cut_77zs_batch_normalize_62kc = tf.nn.batch_normalization(x=resnet_64_short_cut_77zs_conv0, mean=trainable_params['resnet_64_short_cut_77zs/batch_normalize_62kc/mean'], variance=trainable_params['resnet_64_short_cut_77zs/batch_normalize_62kc/variance'], offset=trainable_params['resnet_64_short_cut_77zs/batch_normalize_62kc/offset'], scale=trainable_params['resnet_64_short_cut_77zs/batch_normalize_62kc/scale'], variance_epsilon=0.001, name='resnet_64_short_cut_77zs/batch_normalize_62kc')
    resnet_64_short_cut_77zs_relu_64ms = tf.nn.relu(features=resnet_64_short_cut_77zs_batch_normalize_62kc, name='resnet_64_short_cut_77zs/relu_64ms')
    resnet_64_short_cut_77zs_conv_66oi = tf.nn.conv2d(input=resnet_64_short_cut_77zs_relu_64ms, filters=trainable_params['resnet_64_short_cut_77zs/conv_66oi/filters'], strides=[1, 1], padding='SAME', data_format='NHWC', dilations=1, use_bias=False, name='resnet_64_short_cut_77zs/conv_66oi')
    resnet_64_short_cut_77zs_batch_normalize_68qy = tf.nn.batch_normalization(x=resnet_64_short_cut_77zs_conv_66oi, mean=trainable_params['resnet_64_short_cut_77zs/batch_normalize_68qy/mean'], variance=trainable_params['resnet_64_short_cut_77zs/batch_normalize_68qy/variance'], offset=trainable_params['resnet_64_short_cut_77zs/batch_normalize_68qy/offset'], scale=trainable_params['resnet_64_short_cut_77zs/batch_normalize_68qy/scale'], variance_epsilon=0.001, name='resnet_64_short_cut_77zs/batch_normalize_68qy')
    resnet_64_short_cut_77zs_conv1 = tf.nn.relu(features=resnet_64_short_cut_77zs_batch_normalize_68qy, name='resnet_64_short_cut_77zs/conv1')
    resnet_64_short_cut_77zs_conv_72ue = tf.nn.conv2d(input=resnet_32_58gw_relu_57fo, filters=trainable_params['resnet_64_short_cut_77zs/conv_72ue/filters'], strides=[2, 2], padding='VALID', data_format='NHWC', dilations=1, use_bias=False, name='resnet_64_short_cut_77zs/conv_72ue')
    resnet_64_short_cut_77zs_short_cut_32_64 = tf.nn.batch_normalization(x=resnet_64_short_cut_77zs_conv_72ue, mean=trainable_params['resnet_64_short_cut_77zs/short_cut_32_64/mean'], variance=trainable_params['resnet_64_short_cut_77zs/short_cut_32_64/variance'], offset=trainable_params['resnet_64_short_cut_77zs/short_cut_32_64/offset'], scale=trainable_params['resnet_64_short_cut_77zs/short_cut_32_64/scale'], variance_epsilon=0.001, name='resnet_64_short_cut_77zs/short_cut_32_64')
    resnet_64_short_cut_77zs_add_76yk = tf.math.add(x=[resnet_64_short_cut_77zs_short_cut_32_64, resnet_64_short_cut_77zs_conv1][0], y=[resnet_64_short_cut_77zs_short_cut_32_64, resnet_64_short_cut_77zs_conv1][1], name='resnet_64_short_cut_77zs/add_76yk')
    features_conv_79bi = tf.nn.conv2d(input=resnet_64_short_cut_77zs_add_76yk, filters=trainable_params['features/conv_79bi/filters'], strides=[1, 1], padding='SAME', data_format='NHWC', dilations=1, use_bias=False, name='features/conv_79bi')
    features_batch_normalize_81dy = tf.nn.batch_normalization(x=features_conv_79bi, mean=trainable_params['features/batch_normalize_81dy/mean'], variance=trainable_params['features/batch_normalize_81dy/variance'], offset=trainable_params['features/batch_normalize_81dy/offset'], scale=trainable_params['features/batch_normalize_81dy/scale'], variance_epsilon=0.001, name='features/batch_normalize_81dy')
    features_relu_83fo = tf.nn.relu(features=features_batch_normalize_81dy, name='features/relu_83fo')
    features_conv_85he = tf.nn.conv2d(input=features_relu_83fo, filters=trainable_params['features/conv_85he/filters'], strides=[1, 1], padding='SAME', data_format='NHWC', dilations=1, use_bias=False, name='features/conv_85he')
    features_conv = tf.nn.batch_normalization(x=features_conv_85he, mean=trainable_params['features/conv/mean'], variance=trainable_params['features/conv/variance'], offset=trainable_params['features/conv/offset'], scale=trainable_params['features/conv/scale'], variance_epsilon=0.001, name='features/conv')
    features_add_89lk = tf.math.add(x=[resnet_64_short_cut_77zs_add_76yk, features_conv][0], y=[resnet_64_short_cut_77zs_add_76yk, features_conv][1], name='features/add_89lk')
    features_relu_91na = tf.nn.relu(features=features_add_89lk, name='features/relu_91na')
    max_pool2d_94qy = tf.nn.max_pool(input=features_relu_91na, ksize=[3, 3], strides=[2, 2], padding='VALID', data_format='NHWC', name='max_pool2d_94qy')
    flatten_96so = tf.reshape(tensor=max_pool2d_94qy, shape=(-1, np.asarray(max_pool2d_94qy.get_shape().as_list()[1:]).prod()), name='flatten_96so')
    dense_98ue = tf.add(x=tf.matmul(a=flatten_96so, b=trainable_params['dense_98ue/weights']), y=trainable_params['dense_98ue/bias'], name='dense_98ue')
    d_1 = tf.nn.dropout(x=dense_98ue, rate=0.2, noise_shape=None, seed=None, name='d_1')
    return d_1 


def get_loss(d_1, labels, trainable_params):
    cross_0 = tf.nn.softmax_cross_entropy_with_logits(labels=[labels, d_1][0], logits=[labels, d_1][1], axis=-1, name='cross_0')
    regularizer1 = 0.002*sum(list(map(lambda x: tf.nn.l2_loss(t=trainable_params[x], name='regularizer1'), ['conv_5fo/filters', 'resnet_16_24yk/conv_11lk/filters', 'resnet_16_24yk/conv_17rg/filters', 'resnet_32_short_cut_43rg/conv0/filters', 'resnet_32_short_cut_43rg/conv_32gw/filters', 'resnet_32_short_cut_43rg/conv_38ms/filters', 'resnet_32_58gw/conv_45tw/filters', 'resnet_32_58gw/conv_51zs/filters', 'resnet_64_short_cut_77zs/conv0/filters', 'resnet_64_short_cut_77zs/conv_66oi/filters', 'resnet_64_short_cut_77zs/conv_72ue/filters', 'features/conv_79bi/filters', 'features/conv_85he/filters'])))
    losses = tf.math.add(x=[cross_0, regularizer1][0], y=[cross_0, regularizer1][1], name='losses')
    return losses 


def get_optimizer(trainable_params):
    solver = tf.optimizers.Adam(learning_rate=tf.keras.optimizers.schedules.ExponentialDecay(initial_learning_rate=0.001, decay_steps=100000, decay_rate=0.96, staircase=True), beta_1=0.9, beta_2=0.999, epsilon=1e-08, name='solver')(trainable_params)
    return solver 
