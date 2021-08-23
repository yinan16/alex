import tensorflow as tf
import numpy as np


tf_dtypes = {'float32': tf.float32, 'int8': tf.int8}


def get_trainable_params(tf_dtypes, tf):
    trainable_params = dict()
    model_block_test_recipe_34im_conv_6gw_filters_initializer_xavier_uniform = tf.keras.initializers.glorot_uniform(seed=1)(shape=(3, 3, 3, 16))
    model_block_test_recipe_34im_conv_6gw_filters = tf.Variable(initial_value=model_block_test_recipe_34im_conv_6gw_filters_initializer_xavier_uniform, trainable=True, caching_device=None, name='model_block/test_recipe_34im/conv_6gw/filters', variable_def=None, dtype=tf_dtypes['float32'], import_scope=None, constraint=None, synchronization=tf.VariableSynchronization.AUTO, shape=None)
    trainable_params['model_block/test_recipe_34im/conv_6gw/filters'] = model_block_test_recipe_34im_conv_6gw_filters
    model_block_test_recipe_34im_batch_normalize_8im_mean_initializer_zeros_initializer = tf.zeros_initializer()(shape=[16, ])
    model_block_test_recipe_34im_batch_normalize_8im_mean = tf.Variable(initial_value=model_block_test_recipe_34im_batch_normalize_8im_mean_initializer_zeros_initializer, trainable=False, caching_device=None, name='model_block/test_recipe_34im/batch_normalize_8im/mean', variable_def=None, dtype=tf_dtypes['float32'], import_scope=None, constraint=None, synchronization=tf.VariableSynchronization.AUTO, shape=None)
    trainable_params['model_block/test_recipe_34im/batch_normalize_8im/mean'] = model_block_test_recipe_34im_batch_normalize_8im_mean
    model_block_test_recipe_34im_batch_normalize_8im_offset_initializer_zeros_initializer = tf.zeros_initializer()(shape=[16, ])
    model_block_test_recipe_34im_batch_normalize_8im_offset = tf.Variable(initial_value=model_block_test_recipe_34im_batch_normalize_8im_offset_initializer_zeros_initializer, trainable=True, caching_device=None, name='model_block/test_recipe_34im/batch_normalize_8im/offset', variable_def=None, dtype=tf_dtypes['float32'], import_scope=None, constraint=None, synchronization=tf.VariableSynchronization.AUTO, shape=None)
    trainable_params['model_block/test_recipe_34im/batch_normalize_8im/offset'] = model_block_test_recipe_34im_batch_normalize_8im_offset
    model_block_test_recipe_34im_batch_normalize_8im_scale_initializer_ones_initializer = tf.ones_initializer()(shape=[16, ])
    model_block_test_recipe_34im_batch_normalize_8im_scale = tf.Variable(initial_value=model_block_test_recipe_34im_batch_normalize_8im_scale_initializer_ones_initializer, trainable=True, caching_device=None, name='model_block/test_recipe_34im/batch_normalize_8im/scale', variable_def=None, dtype=tf_dtypes['float32'], import_scope=None, constraint=None, synchronization=tf.VariableSynchronization.AUTO, shape=None)
    trainable_params['model_block/test_recipe_34im/batch_normalize_8im/scale'] = model_block_test_recipe_34im_batch_normalize_8im_scale
    model_block_test_recipe_34im_batch_normalize_8im_variance_initializer_ones_initializer = tf.ones_initializer()(shape=[16, ])
    model_block_test_recipe_34im_batch_normalize_8im_variance = tf.Variable(initial_value=model_block_test_recipe_34im_batch_normalize_8im_variance_initializer_ones_initializer, trainable=False, caching_device=None, name='model_block/test_recipe_34im/batch_normalize_8im/variance', variable_def=None, dtype=tf_dtypes['float32'], import_scope=None, constraint=None, synchronization=tf.VariableSynchronization.AUTO, shape=None)
    trainable_params['model_block/test_recipe_34im/batch_normalize_8im/variance'] = model_block_test_recipe_34im_batch_normalize_8im_variance
    model_block_test_recipe_34im_conv_12ms_filters_initializer_xavier_uniform = tf.keras.initializers.glorot_uniform(seed=1)(shape=(3, 3, 16, 16))
    model_block_test_recipe_34im_conv_12ms_filters = tf.Variable(initial_value=model_block_test_recipe_34im_conv_12ms_filters_initializer_xavier_uniform, trainable=True, caching_device=None, name='model_block/test_recipe_34im/conv_12ms/filters', variable_def=None, dtype=tf_dtypes['float32'], import_scope=None, constraint=None, synchronization=tf.VariableSynchronization.AUTO, shape=None)
    trainable_params['model_block/test_recipe_34im/conv_12ms/filters'] = model_block_test_recipe_34im_conv_12ms_filters
    model_block_test_recipe_34im_conv_mean_initializer_zeros_initializer = tf.zeros_initializer()(shape=[16, ])
    model_block_test_recipe_34im_conv_mean = tf.Variable(initial_value=model_block_test_recipe_34im_conv_mean_initializer_zeros_initializer, trainable=False, caching_device=None, name='model_block/test_recipe_34im/conv/mean', variable_def=None, dtype=tf_dtypes['float32'], import_scope=None, constraint=None, synchronization=tf.VariableSynchronization.AUTO, shape=None)
    trainable_params['model_block/test_recipe_34im/conv/mean'] = model_block_test_recipe_34im_conv_mean
    model_block_test_recipe_34im_conv_offset_initializer_zeros_initializer = tf.zeros_initializer()(shape=[16, ])
    model_block_test_recipe_34im_conv_offset = tf.Variable(initial_value=model_block_test_recipe_34im_conv_offset_initializer_zeros_initializer, trainable=True, caching_device=None, name='model_block/test_recipe_34im/conv/offset', variable_def=None, dtype=tf_dtypes['float32'], import_scope=None, constraint=None, synchronization=tf.VariableSynchronization.AUTO, shape=None)
    trainable_params['model_block/test_recipe_34im/conv/offset'] = model_block_test_recipe_34im_conv_offset
    model_block_test_recipe_34im_conv_scale_initializer_ones_initializer = tf.ones_initializer()(shape=[16, ])
    model_block_test_recipe_34im_conv_scale = tf.Variable(initial_value=model_block_test_recipe_34im_conv_scale_initializer_ones_initializer, trainable=True, caching_device=None, name='model_block/test_recipe_34im/conv/scale', variable_def=None, dtype=tf_dtypes['float32'], import_scope=None, constraint=None, synchronization=tf.VariableSynchronization.AUTO, shape=None)
    trainable_params['model_block/test_recipe_34im/conv/scale'] = model_block_test_recipe_34im_conv_scale
    model_block_test_recipe_34im_conv_variance_initializer_ones_initializer = tf.ones_initializer()(shape=[16, ])
    model_block_test_recipe_34im_conv_variance = tf.Variable(initial_value=model_block_test_recipe_34im_conv_variance_initializer_ones_initializer, trainable=False, caching_device=None, name='model_block/test_recipe_34im/conv/variance', variable_def=None, dtype=tf_dtypes['float32'], import_scope=None, constraint=None, synchronization=tf.VariableSynchronization.AUTO, shape=None)
    trainable_params['model_block/test_recipe_34im/conv/variance'] = model_block_test_recipe_34im_conv_variance
    model_block_test_recipe_34im_resnet_16_33he_conv_20ue_filters_initializer_xavier_uniform = tf.keras.initializers.glorot_uniform(seed=1)(shape=(3, 3, 3, 16))
    model_block_test_recipe_34im_resnet_16_33he_conv_20ue_filters = tf.Variable(initial_value=model_block_test_recipe_34im_resnet_16_33he_conv_20ue_filters_initializer_xavier_uniform, trainable=True, caching_device=None, name='model_block/test_recipe_34im/resnet_16_33he/conv_20ue/filters', variable_def=None, dtype=tf_dtypes['float32'], import_scope=None, constraint=None, synchronization=tf.VariableSynchronization.AUTO, shape=None)
    trainable_params['model_block/test_recipe_34im/resnet_16_33he/conv_20ue/filters'] = model_block_test_recipe_34im_resnet_16_33he_conv_20ue_filters
    model_block_test_recipe_34im_resnet_16_33he_batch_normalize_22wu_mean_initializer_zeros_initializer = tf.zeros_initializer()(shape=[16, ])
    model_block_test_recipe_34im_resnet_16_33he_batch_normalize_22wu_mean = tf.Variable(initial_value=model_block_test_recipe_34im_resnet_16_33he_batch_normalize_22wu_mean_initializer_zeros_initializer, trainable=False, caching_device=None, name='model_block/test_recipe_34im/resnet_16_33he/batch_normalize_22wu/mean', variable_def=None, dtype=tf_dtypes['float32'], import_scope=None, constraint=None, synchronization=tf.VariableSynchronization.AUTO, shape=None)
    trainable_params['model_block/test_recipe_34im/resnet_16_33he/batch_normalize_22wu/mean'] = model_block_test_recipe_34im_resnet_16_33he_batch_normalize_22wu_mean
    model_block_test_recipe_34im_resnet_16_33he_batch_normalize_22wu_offset_initializer_zeros_initializer = tf.zeros_initializer()(shape=[16, ])
    model_block_test_recipe_34im_resnet_16_33he_batch_normalize_22wu_offset = tf.Variable(initial_value=model_block_test_recipe_34im_resnet_16_33he_batch_normalize_22wu_offset_initializer_zeros_initializer, trainable=True, caching_device=None, name='model_block/test_recipe_34im/resnet_16_33he/batch_normalize_22wu/offset', variable_def=None, dtype=tf_dtypes['float32'], import_scope=None, constraint=None, synchronization=tf.VariableSynchronization.AUTO, shape=None)
    trainable_params['model_block/test_recipe_34im/resnet_16_33he/batch_normalize_22wu/offset'] = model_block_test_recipe_34im_resnet_16_33he_batch_normalize_22wu_offset
    model_block_test_recipe_34im_resnet_16_33he_batch_normalize_22wu_scale_initializer_ones_initializer = tf.ones_initializer()(shape=[16, ])
    model_block_test_recipe_34im_resnet_16_33he_batch_normalize_22wu_scale = tf.Variable(initial_value=model_block_test_recipe_34im_resnet_16_33he_batch_normalize_22wu_scale_initializer_ones_initializer, trainable=True, caching_device=None, name='model_block/test_recipe_34im/resnet_16_33he/batch_normalize_22wu/scale', variable_def=None, dtype=tf_dtypes['float32'], import_scope=None, constraint=None, synchronization=tf.VariableSynchronization.AUTO, shape=None)
    trainable_params['model_block/test_recipe_34im/resnet_16_33he/batch_normalize_22wu/scale'] = model_block_test_recipe_34im_resnet_16_33he_batch_normalize_22wu_scale
    model_block_test_recipe_34im_resnet_16_33he_batch_normalize_22wu_variance_initializer_ones_initializer = tf.ones_initializer()(shape=[16, ])
    model_block_test_recipe_34im_resnet_16_33he_batch_normalize_22wu_variance = tf.Variable(initial_value=model_block_test_recipe_34im_resnet_16_33he_batch_normalize_22wu_variance_initializer_ones_initializer, trainable=False, caching_device=None, name='model_block/test_recipe_34im/resnet_16_33he/batch_normalize_22wu/variance', variable_def=None, dtype=tf_dtypes['float32'], import_scope=None, constraint=None, synchronization=tf.VariableSynchronization.AUTO, shape=None)
    trainable_params['model_block/test_recipe_34im/resnet_16_33he/batch_normalize_22wu/variance'] = model_block_test_recipe_34im_resnet_16_33he_batch_normalize_22wu_variance
    model_block_test_recipe_34im_resnet_16_33he_conv_26aa_filters_initializer_xavier_uniform = tf.keras.initializers.glorot_uniform(seed=1)(shape=(3, 3, 16, 16))
    model_block_test_recipe_34im_resnet_16_33he_conv_26aa_filters = tf.Variable(initial_value=model_block_test_recipe_34im_resnet_16_33he_conv_26aa_filters_initializer_xavier_uniform, trainable=True, caching_device=None, name='model_block/test_recipe_34im/resnet_16_33he/conv_26aa/filters', variable_def=None, dtype=tf_dtypes['float32'], import_scope=None, constraint=None, synchronization=tf.VariableSynchronization.AUTO, shape=None)
    trainable_params['model_block/test_recipe_34im/resnet_16_33he/conv_26aa/filters'] = model_block_test_recipe_34im_resnet_16_33he_conv_26aa_filters
    model_block_test_recipe_34im_resnet_16_33he_conv_mean_initializer_zeros_initializer = tf.zeros_initializer()(shape=[16, ])
    model_block_test_recipe_34im_resnet_16_33he_conv_mean = tf.Variable(initial_value=model_block_test_recipe_34im_resnet_16_33he_conv_mean_initializer_zeros_initializer, trainable=False, caching_device=None, name='model_block/test_recipe_34im/resnet_16_33he/conv/mean', variable_def=None, dtype=tf_dtypes['float32'], import_scope=None, constraint=None, synchronization=tf.VariableSynchronization.AUTO, shape=None)
    trainable_params['model_block/test_recipe_34im/resnet_16_33he/conv/mean'] = model_block_test_recipe_34im_resnet_16_33he_conv_mean
    model_block_test_recipe_34im_resnet_16_33he_conv_offset_initializer_zeros_initializer = tf.zeros_initializer()(shape=[16, ])
    model_block_test_recipe_34im_resnet_16_33he_conv_offset = tf.Variable(initial_value=model_block_test_recipe_34im_resnet_16_33he_conv_offset_initializer_zeros_initializer, trainable=True, caching_device=None, name='model_block/test_recipe_34im/resnet_16_33he/conv/offset', variable_def=None, dtype=tf_dtypes['float32'], import_scope=None, constraint=None, synchronization=tf.VariableSynchronization.AUTO, shape=None)
    trainable_params['model_block/test_recipe_34im/resnet_16_33he/conv/offset'] = model_block_test_recipe_34im_resnet_16_33he_conv_offset
    model_block_test_recipe_34im_resnet_16_33he_conv_scale_initializer_ones_initializer = tf.ones_initializer()(shape=[16, ])
    model_block_test_recipe_34im_resnet_16_33he_conv_scale = tf.Variable(initial_value=model_block_test_recipe_34im_resnet_16_33he_conv_scale_initializer_ones_initializer, trainable=True, caching_device=None, name='model_block/test_recipe_34im/resnet_16_33he/conv/scale', variable_def=None, dtype=tf_dtypes['float32'], import_scope=None, constraint=None, synchronization=tf.VariableSynchronization.AUTO, shape=None)
    trainable_params['model_block/test_recipe_34im/resnet_16_33he/conv/scale'] = model_block_test_recipe_34im_resnet_16_33he_conv_scale
    model_block_test_recipe_34im_resnet_16_33he_conv_variance_initializer_ones_initializer = tf.ones_initializer()(shape=[16, ])
    model_block_test_recipe_34im_resnet_16_33he_conv_variance = tf.Variable(initial_value=model_block_test_recipe_34im_resnet_16_33he_conv_variance_initializer_ones_initializer, trainable=False, caching_device=None, name='model_block/test_recipe_34im/resnet_16_33he/conv/variance', variable_def=None, dtype=tf_dtypes['float32'], import_scope=None, constraint=None, synchronization=tf.VariableSynchronization.AUTO, shape=None)
    trainable_params['model_block/test_recipe_34im/resnet_16_33he/conv/variance'] = model_block_test_recipe_34im_resnet_16_33he_conv_variance
    model_block_resnet_16_49xc_0_conv_36kc_filters_initializer_xavier_uniform = tf.keras.initializers.glorot_uniform(seed=1)(shape=(3, 3, 3, 16))
    model_block_resnet_16_49xc_0_conv_36kc_filters = tf.Variable(initial_value=model_block_resnet_16_49xc_0_conv_36kc_filters_initializer_xavier_uniform, trainable=True, caching_device=None, name='model_block/resnet_16_49xc_0/conv_36kc/filters', variable_def=None, dtype=tf_dtypes['float32'], import_scope=None, constraint=None, synchronization=tf.VariableSynchronization.AUTO, shape=None)
    trainable_params['model_block/resnet_16_49xc_0/conv_36kc/filters'] = model_block_resnet_16_49xc_0_conv_36kc_filters
    model_block_resnet_16_49xc_0_batch_normalize_38ms_mean_initializer_zeros_initializer = tf.zeros_initializer()(shape=[16, ])
    model_block_resnet_16_49xc_0_batch_normalize_38ms_mean = tf.Variable(initial_value=model_block_resnet_16_49xc_0_batch_normalize_38ms_mean_initializer_zeros_initializer, trainable=False, caching_device=None, name='model_block/resnet_16_49xc_0/batch_normalize_38ms/mean', variable_def=None, dtype=tf_dtypes['float32'], import_scope=None, constraint=None, synchronization=tf.VariableSynchronization.AUTO, shape=None)
    trainable_params['model_block/resnet_16_49xc_0/batch_normalize_38ms/mean'] = model_block_resnet_16_49xc_0_batch_normalize_38ms_mean
    model_block_resnet_16_49xc_0_batch_normalize_38ms_offset_initializer_zeros_initializer = tf.zeros_initializer()(shape=[16, ])
    model_block_resnet_16_49xc_0_batch_normalize_38ms_offset = tf.Variable(initial_value=model_block_resnet_16_49xc_0_batch_normalize_38ms_offset_initializer_zeros_initializer, trainable=True, caching_device=None, name='model_block/resnet_16_49xc_0/batch_normalize_38ms/offset', variable_def=None, dtype=tf_dtypes['float32'], import_scope=None, constraint=None, synchronization=tf.VariableSynchronization.AUTO, shape=None)
    trainable_params['model_block/resnet_16_49xc_0/batch_normalize_38ms/offset'] = model_block_resnet_16_49xc_0_batch_normalize_38ms_offset
    model_block_resnet_16_49xc_0_batch_normalize_38ms_scale_initializer_ones_initializer = tf.ones_initializer()(shape=[16, ])
    model_block_resnet_16_49xc_0_batch_normalize_38ms_scale = tf.Variable(initial_value=model_block_resnet_16_49xc_0_batch_normalize_38ms_scale_initializer_ones_initializer, trainable=True, caching_device=None, name='model_block/resnet_16_49xc_0/batch_normalize_38ms/scale', variable_def=None, dtype=tf_dtypes['float32'], import_scope=None, constraint=None, synchronization=tf.VariableSynchronization.AUTO, shape=None)
    trainable_params['model_block/resnet_16_49xc_0/batch_normalize_38ms/scale'] = model_block_resnet_16_49xc_0_batch_normalize_38ms_scale
    model_block_resnet_16_49xc_0_batch_normalize_38ms_variance_initializer_ones_initializer = tf.ones_initializer()(shape=[16, ])
    model_block_resnet_16_49xc_0_batch_normalize_38ms_variance = tf.Variable(initial_value=model_block_resnet_16_49xc_0_batch_normalize_38ms_variance_initializer_ones_initializer, trainable=False, caching_device=None, name='model_block/resnet_16_49xc_0/batch_normalize_38ms/variance', variable_def=None, dtype=tf_dtypes['float32'], import_scope=None, constraint=None, synchronization=tf.VariableSynchronization.AUTO, shape=None)
    trainable_params['model_block/resnet_16_49xc_0/batch_normalize_38ms/variance'] = model_block_resnet_16_49xc_0_batch_normalize_38ms_variance
    model_block_resnet_16_49xc_0_conv_42qy_filters_initializer_xavier_uniform = tf.keras.initializers.glorot_uniform(seed=1)(shape=(3, 3, 16, 16))
    model_block_resnet_16_49xc_0_conv_42qy_filters = tf.Variable(initial_value=model_block_resnet_16_49xc_0_conv_42qy_filters_initializer_xavier_uniform, trainable=True, caching_device=None, name='model_block/resnet_16_49xc_0/conv_42qy/filters', variable_def=None, dtype=tf_dtypes['float32'], import_scope=None, constraint=None, synchronization=tf.VariableSynchronization.AUTO, shape=None)
    trainable_params['model_block/resnet_16_49xc_0/conv_42qy/filters'] = model_block_resnet_16_49xc_0_conv_42qy_filters
    model_block_resnet_16_49xc_0_conv_mean_initializer_zeros_initializer = tf.zeros_initializer()(shape=[16, ])
    model_block_resnet_16_49xc_0_conv_mean = tf.Variable(initial_value=model_block_resnet_16_49xc_0_conv_mean_initializer_zeros_initializer, trainable=False, caching_device=None, name='model_block/resnet_16_49xc_0/conv/mean', variable_def=None, dtype=tf_dtypes['float32'], import_scope=None, constraint=None, synchronization=tf.VariableSynchronization.AUTO, shape=None)
    trainable_params['model_block/resnet_16_49xc_0/conv/mean'] = model_block_resnet_16_49xc_0_conv_mean
    model_block_resnet_16_49xc_0_conv_offset_initializer_zeros_initializer = tf.zeros_initializer()(shape=[16, ])
    model_block_resnet_16_49xc_0_conv_offset = tf.Variable(initial_value=model_block_resnet_16_49xc_0_conv_offset_initializer_zeros_initializer, trainable=True, caching_device=None, name='model_block/resnet_16_49xc_0/conv/offset', variable_def=None, dtype=tf_dtypes['float32'], import_scope=None, constraint=None, synchronization=tf.VariableSynchronization.AUTO, shape=None)
    trainable_params['model_block/resnet_16_49xc_0/conv/offset'] = model_block_resnet_16_49xc_0_conv_offset
    model_block_resnet_16_49xc_0_conv_scale_initializer_ones_initializer = tf.ones_initializer()(shape=[16, ])
    model_block_resnet_16_49xc_0_conv_scale = tf.Variable(initial_value=model_block_resnet_16_49xc_0_conv_scale_initializer_ones_initializer, trainable=True, caching_device=None, name='model_block/resnet_16_49xc_0/conv/scale', variable_def=None, dtype=tf_dtypes['float32'], import_scope=None, constraint=None, synchronization=tf.VariableSynchronization.AUTO, shape=None)
    trainable_params['model_block/resnet_16_49xc_0/conv/scale'] = model_block_resnet_16_49xc_0_conv_scale
    model_block_resnet_16_49xc_0_conv_variance_initializer_ones_initializer = tf.ones_initializer()(shape=[16, ])
    model_block_resnet_16_49xc_0_conv_variance = tf.Variable(initial_value=model_block_resnet_16_49xc_0_conv_variance_initializer_ones_initializer, trainable=False, caching_device=None, name='model_block/resnet_16_49xc_0/conv/variance', variable_def=None, dtype=tf_dtypes['float32'], import_scope=None, constraint=None, synchronization=tf.VariableSynchronization.AUTO, shape=None)
    trainable_params['model_block/resnet_16_49xc_0/conv/variance'] = model_block_resnet_16_49xc_0_conv_variance
    model_block_resnet_16_49xc_conv_36kc_filters_initializer_xavier_uniform = tf.keras.initializers.glorot_uniform(seed=1)(shape=(3, 3, 3, 16))
    model_block_resnet_16_49xc_conv_36kc_filters = tf.Variable(initial_value=model_block_resnet_16_49xc_conv_36kc_filters_initializer_xavier_uniform, trainable=True, caching_device=None, name='model_block/resnet_16_49xc/conv_36kc/filters', variable_def=None, dtype=tf_dtypes['float32'], import_scope=None, constraint=None, synchronization=tf.VariableSynchronization.AUTO, shape=None)
    trainable_params['model_block/resnet_16_49xc/conv_36kc/filters'] = model_block_resnet_16_49xc_conv_36kc_filters
    model_block_resnet_16_49xc_batch_normalize_38ms_mean_initializer_zeros_initializer = tf.zeros_initializer()(shape=[16, ])
    model_block_resnet_16_49xc_batch_normalize_38ms_mean = tf.Variable(initial_value=model_block_resnet_16_49xc_batch_normalize_38ms_mean_initializer_zeros_initializer, trainable=False, caching_device=None, name='model_block/resnet_16_49xc/batch_normalize_38ms/mean', variable_def=None, dtype=tf_dtypes['float32'], import_scope=None, constraint=None, synchronization=tf.VariableSynchronization.AUTO, shape=None)
    trainable_params['model_block/resnet_16_49xc/batch_normalize_38ms/mean'] = model_block_resnet_16_49xc_batch_normalize_38ms_mean
    model_block_resnet_16_49xc_batch_normalize_38ms_offset_initializer_zeros_initializer = tf.zeros_initializer()(shape=[16, ])
    model_block_resnet_16_49xc_batch_normalize_38ms_offset = tf.Variable(initial_value=model_block_resnet_16_49xc_batch_normalize_38ms_offset_initializer_zeros_initializer, trainable=True, caching_device=None, name='model_block/resnet_16_49xc/batch_normalize_38ms/offset', variable_def=None, dtype=tf_dtypes['float32'], import_scope=None, constraint=None, synchronization=tf.VariableSynchronization.AUTO, shape=None)
    trainable_params['model_block/resnet_16_49xc/batch_normalize_38ms/offset'] = model_block_resnet_16_49xc_batch_normalize_38ms_offset
    model_block_resnet_16_49xc_batch_normalize_38ms_scale_initializer_ones_initializer = tf.ones_initializer()(shape=[16, ])
    model_block_resnet_16_49xc_batch_normalize_38ms_scale = tf.Variable(initial_value=model_block_resnet_16_49xc_batch_normalize_38ms_scale_initializer_ones_initializer, trainable=True, caching_device=None, name='model_block/resnet_16_49xc/batch_normalize_38ms/scale', variable_def=None, dtype=tf_dtypes['float32'], import_scope=None, constraint=None, synchronization=tf.VariableSynchronization.AUTO, shape=None)
    trainable_params['model_block/resnet_16_49xc/batch_normalize_38ms/scale'] = model_block_resnet_16_49xc_batch_normalize_38ms_scale
    model_block_resnet_16_49xc_batch_normalize_38ms_variance_initializer_ones_initializer = tf.ones_initializer()(shape=[16, ])
    model_block_resnet_16_49xc_batch_normalize_38ms_variance = tf.Variable(initial_value=model_block_resnet_16_49xc_batch_normalize_38ms_variance_initializer_ones_initializer, trainable=False, caching_device=None, name='model_block/resnet_16_49xc/batch_normalize_38ms/variance', variable_def=None, dtype=tf_dtypes['float32'], import_scope=None, constraint=None, synchronization=tf.VariableSynchronization.AUTO, shape=None)
    trainable_params['model_block/resnet_16_49xc/batch_normalize_38ms/variance'] = model_block_resnet_16_49xc_batch_normalize_38ms_variance
    model_block_resnet_16_49xc_conv_42qy_filters_initializer_xavier_uniform = tf.keras.initializers.glorot_uniform(seed=1)(shape=(3, 3, 16, 16))
    model_block_resnet_16_49xc_conv_42qy_filters = tf.Variable(initial_value=model_block_resnet_16_49xc_conv_42qy_filters_initializer_xavier_uniform, trainable=True, caching_device=None, name='model_block/resnet_16_49xc/conv_42qy/filters', variable_def=None, dtype=tf_dtypes['float32'], import_scope=None, constraint=None, synchronization=tf.VariableSynchronization.AUTO, shape=None)
    trainable_params['model_block/resnet_16_49xc/conv_42qy/filters'] = model_block_resnet_16_49xc_conv_42qy_filters
    model_block_resnet_16_49xc_conv_mean_initializer_zeros_initializer = tf.zeros_initializer()(shape=[16, ])
    model_block_resnet_16_49xc_conv_mean = tf.Variable(initial_value=model_block_resnet_16_49xc_conv_mean_initializer_zeros_initializer, trainable=False, caching_device=None, name='model_block/resnet_16_49xc/conv/mean', variable_def=None, dtype=tf_dtypes['float32'], import_scope=None, constraint=None, synchronization=tf.VariableSynchronization.AUTO, shape=None)
    trainable_params['model_block/resnet_16_49xc/conv/mean'] = model_block_resnet_16_49xc_conv_mean
    model_block_resnet_16_49xc_conv_offset_initializer_zeros_initializer = tf.zeros_initializer()(shape=[16, ])
    model_block_resnet_16_49xc_conv_offset = tf.Variable(initial_value=model_block_resnet_16_49xc_conv_offset_initializer_zeros_initializer, trainable=True, caching_device=None, name='model_block/resnet_16_49xc/conv/offset', variable_def=None, dtype=tf_dtypes['float32'], import_scope=None, constraint=None, synchronization=tf.VariableSynchronization.AUTO, shape=None)
    trainable_params['model_block/resnet_16_49xc/conv/offset'] = model_block_resnet_16_49xc_conv_offset
    model_block_resnet_16_49xc_conv_scale_initializer_ones_initializer = tf.ones_initializer()(shape=[16, ])
    model_block_resnet_16_49xc_conv_scale = tf.Variable(initial_value=model_block_resnet_16_49xc_conv_scale_initializer_ones_initializer, trainable=True, caching_device=None, name='model_block/resnet_16_49xc/conv/scale', variable_def=None, dtype=tf_dtypes['float32'], import_scope=None, constraint=None, synchronization=tf.VariableSynchronization.AUTO, shape=None)
    trainable_params['model_block/resnet_16_49xc/conv/scale'] = model_block_resnet_16_49xc_conv_scale
    model_block_resnet_16_49xc_conv_variance_initializer_ones_initializer = tf.ones_initializer()(shape=[16, ])
    model_block_resnet_16_49xc_conv_variance = tf.Variable(initial_value=model_block_resnet_16_49xc_conv_variance_initializer_ones_initializer, trainable=False, caching_device=None, name='model_block/resnet_16_49xc/conv/variance', variable_def=None, dtype=tf_dtypes['float32'], import_scope=None, constraint=None, synchronization=tf.VariableSynchronization.AUTO, shape=None)
    trainable_params['model_block/resnet_16_49xc/conv/variance'] = model_block_resnet_16_49xc_conv_variance
    model_block_batch_normalize_55dy_mean_initializer_zeros_initializer = tf.zeros_initializer()(shape=[3, ])
    model_block_batch_normalize_55dy_mean = tf.Variable(initial_value=model_block_batch_normalize_55dy_mean_initializer_zeros_initializer, trainable=False, caching_device=None, name='model_block/batch_normalize_55dy/mean', variable_def=None, dtype=tf_dtypes['float32'], import_scope=None, constraint=None, synchronization=tf.VariableSynchronization.AUTO, shape=None)
    trainable_params['model_block/batch_normalize_55dy/mean'] = model_block_batch_normalize_55dy_mean
    model_block_batch_normalize_55dy_offset_initializer_zeros_initializer = tf.zeros_initializer()(shape=[3, ])
    model_block_batch_normalize_55dy_offset = tf.Variable(initial_value=model_block_batch_normalize_55dy_offset_initializer_zeros_initializer, trainable=True, caching_device=None, name='model_block/batch_normalize_55dy/offset', variable_def=None, dtype=tf_dtypes['float32'], import_scope=None, constraint=None, synchronization=tf.VariableSynchronization.AUTO, shape=None)
    trainable_params['model_block/batch_normalize_55dy/offset'] = model_block_batch_normalize_55dy_offset
    model_block_batch_normalize_55dy_scale_initializer_ones_initializer = tf.ones_initializer()(shape=[3, ])
    model_block_batch_normalize_55dy_scale = tf.Variable(initial_value=model_block_batch_normalize_55dy_scale_initializer_ones_initializer, trainable=True, caching_device=None, name='model_block/batch_normalize_55dy/scale', variable_def=None, dtype=tf_dtypes['float32'], import_scope=None, constraint=None, synchronization=tf.VariableSynchronization.AUTO, shape=None)
    trainable_params['model_block/batch_normalize_55dy/scale'] = model_block_batch_normalize_55dy_scale
    model_block_batch_normalize_55dy_variance_initializer_ones_initializer = tf.ones_initializer()(shape=[3, ])
    model_block_batch_normalize_55dy_variance = tf.Variable(initial_value=model_block_batch_normalize_55dy_variance_initializer_ones_initializer, trainable=False, caching_device=None, name='model_block/batch_normalize_55dy/variance', variable_def=None, dtype=tf_dtypes['float32'], import_scope=None, constraint=None, synchronization=tf.VariableSynchronization.AUTO, shape=None)
    trainable_params['model_block/batch_normalize_55dy/variance'] = model_block_batch_normalize_55dy_variance
    model_block_conv_57fo_filters_initializer_xavier_uniform = tf.keras.initializers.glorot_uniform(seed=1)(shape=(3, 3, 3, 16))
    model_block_conv_57fo_filters = tf.Variable(initial_value=model_block_conv_57fo_filters_initializer_xavier_uniform, trainable=True, caching_device=None, name='model_block/conv_57fo/filters', variable_def=None, dtype=tf_dtypes['float32'], import_scope=None, constraint=None, synchronization=tf.VariableSynchronization.AUTO, shape=None)
    trainable_params['model_block/conv_57fo/filters'] = model_block_conv_57fo_filters
    model_block_dense_63lk_bias_initializer_zeros_initializer = tf.zeros_initializer()(shape=[1, ])
    model_block_dense_63lk_bias = tf.Variable(initial_value=model_block_dense_63lk_bias_initializer_zeros_initializer, trainable=True, caching_device=None, name='model_block/dense_63lk/bias', variable_def=None, dtype=tf_dtypes['float32'], import_scope=None, constraint=None, synchronization=tf.VariableSynchronization.AUTO, shape=None)
    trainable_params['model_block/dense_63lk/bias'] = model_block_dense_63lk_bias
    model_block_dense_63lk_weights_initializer_xavier_uniform = tf.keras.initializers.glorot_uniform(seed=2)(shape=[3600, 10])
    model_block_dense_63lk_weights = tf.Variable(initial_value=model_block_dense_63lk_weights_initializer_xavier_uniform, trainable=True, caching_device=None, name='model_block/dense_63lk/weights', variable_def=None, dtype=tf_dtypes['float32'], import_scope=None, constraint=None, synchronization=tf.VariableSynchronization.AUTO, shape=None)
    trainable_params['model_block/dense_63lk/weights'] = model_block_dense_63lk_weights
    return trainable_params


def model(trainable_params, data_block_input_data):
    model_block_test_recipe_34im_conv_6gw = tf.nn.conv2d(input=data_block_input_data, filters=trainable_params['model_block/test_recipe_34im/conv_6gw/filters'], strides=1, padding='SAME', data_format='NHWC', dilations=1, name='model_block/test_recipe_34im/conv_6gw/filters')
    model_block_test_recipe_34im_batch_normalize_8im = tf.nn.batch_normalization(x=model_block_test_recipe_34im_conv_6gw, mean=trainable_params['model_block/test_recipe_34im/batch_normalize_8im/mean'], variance=trainable_params['model_block/test_recipe_34im/batch_normalize_8im/variance'], offset=trainable_params['model_block/test_recipe_34im/batch_normalize_8im/offset'], scale=trainable_params['model_block/test_recipe_34im/batch_normalize_8im/scale'], variance_epsilon=0.001, name='model_block/test_recipe_34im/batch_normalize_8im/variance')
    model_block_test_recipe_34im_relu_10kc = tf.nn.relu(name='model_block/test_recipe_34im/relu_10kc', features=model_block_test_recipe_34im_batch_normalize_8im)
    model_block_test_recipe_34im_conv_12ms = tf.nn.conv2d(input=model_block_test_recipe_34im_relu_10kc, filters=trainable_params['model_block/test_recipe_34im/conv_12ms/filters'], strides=1, padding='SAME', data_format='NHWC', dilations=1, name='model_block/test_recipe_34im/conv_12ms/filters')
    model_block_test_recipe_34im_conv = tf.nn.batch_normalization(x=model_block_test_recipe_34im_conv_12ms, mean=trainable_params['model_block/test_recipe_34im/conv/mean'], variance=trainable_params['model_block/test_recipe_34im/conv/variance'], offset=trainable_params['model_block/test_recipe_34im/conv/offset'], scale=trainable_params['model_block/test_recipe_34im/conv/scale'], variance_epsilon=0.001, name='model_block/test_recipe_34im/conv/variance')
    model_block_test_recipe_34im_add_16qy = tf.math.add(x=[data_block_input_data, model_block_test_recipe_34im_conv][0], y=[data_block_input_data, model_block_test_recipe_34im_conv][1], name='model_block/test_recipe_34im/add_16qy')
    model_block_test_recipe_34im_relu_18so = tf.nn.relu(name='model_block/test_recipe_34im/relu_18so', features=model_block_test_recipe_34im_add_16qy)
    model_block_test_recipe_34im_resnet_16_33he_conv_20ue = tf.nn.conv2d(input=model_block_test_recipe_34im_relu_18so, filters=trainable_params['model_block/test_recipe_34im/resnet_16_33he/conv_20ue/filters'], strides=1, padding='SAME', data_format='NHWC', dilations=1, name='model_block/test_recipe_34im/resnet_16_33he/conv_20ue/filters')
    model_block_test_recipe_34im_resnet_16_33he_batch_normalize_22wu = tf.nn.batch_normalization(x=model_block_test_recipe_34im_resnet_16_33he_conv_20ue, mean=trainable_params['model_block/test_recipe_34im/resnet_16_33he/batch_normalize_22wu/mean'], variance=trainable_params['model_block/test_recipe_34im/resnet_16_33he/batch_normalize_22wu/variance'], offset=trainable_params['model_block/test_recipe_34im/resnet_16_33he/batch_normalize_22wu/offset'], scale=trainable_params['model_block/test_recipe_34im/resnet_16_33he/batch_normalize_22wu/scale'], variance_epsilon=0.001, name='model_block/test_recipe_34im/resnet_16_33he/batch_normalize_22wu/variance')
    model_block_test_recipe_34im_resnet_16_33he_relu_24yk = tf.nn.relu(name='model_block/test_recipe_34im/resnet_16_33he/relu_24yk', features=model_block_test_recipe_34im_resnet_16_33he_batch_normalize_22wu)
    model_block_test_recipe_34im_resnet_16_33he_conv_26aa = tf.nn.conv2d(input=model_block_test_recipe_34im_resnet_16_33he_relu_24yk, filters=trainable_params['model_block/test_recipe_34im/resnet_16_33he/conv_26aa/filters'], strides=1, padding='SAME', data_format='NHWC', dilations=1, name='model_block/test_recipe_34im/resnet_16_33he/conv_26aa/filters')
    model_block_test_recipe_34im_resnet_16_33he_conv = tf.nn.batch_normalization(x=model_block_test_recipe_34im_resnet_16_33he_conv_26aa, mean=trainable_params['model_block/test_recipe_34im/resnet_16_33he/conv/mean'], variance=trainable_params['model_block/test_recipe_34im/resnet_16_33he/conv/variance'], offset=trainable_params['model_block/test_recipe_34im/resnet_16_33he/conv/offset'], scale=trainable_params['model_block/test_recipe_34im/resnet_16_33he/conv/scale'], variance_epsilon=0.001, name='model_block/test_recipe_34im/resnet_16_33he/conv/variance')
    model_block_test_recipe_34im_resnet_16_33he_add_30eg = tf.math.add(x=[model_block_test_recipe_34im_relu_18so, model_block_test_recipe_34im_resnet_16_33he_conv][0], y=[model_block_test_recipe_34im_relu_18so, model_block_test_recipe_34im_resnet_16_33he_conv][1], name='model_block/test_recipe_34im/resnet_16_33he/add_30eg')
    model_block_test_recipe_34im_resnet_16_33he_relu_32gw = tf.nn.relu(name='model_block/test_recipe_34im/resnet_16_33he/relu_32gw', features=model_block_test_recipe_34im_resnet_16_33he_add_30eg)
    model_block_resnet_16_49xc_0_conv_36kc = tf.nn.conv2d(input=model_block_test_recipe_34im_resnet_16_33he_relu_32gw, filters=trainable_params['model_block/resnet_16_49xc_0/conv_36kc/filters'], strides=1, padding='SAME', data_format='NHWC', dilations=1, name='model_block/resnet_16_49xc_0/conv_36kc/filters')
    model_block_resnet_16_49xc_0_batch_normalize_38ms = tf.nn.batch_normalization(x=model_block_resnet_16_49xc_0_conv_36kc, mean=trainable_params['model_block/resnet_16_49xc_0/batch_normalize_38ms/mean'], variance=trainable_params['model_block/resnet_16_49xc_0/batch_normalize_38ms/variance'], offset=trainable_params['model_block/resnet_16_49xc_0/batch_normalize_38ms/offset'], scale=trainable_params['model_block/resnet_16_49xc_0/batch_normalize_38ms/scale'], variance_epsilon=0.001, name='model_block/resnet_16_49xc_0/batch_normalize_38ms/variance')
    model_block_resnet_16_49xc_0_relu_40oi = tf.nn.relu(name='model_block/resnet_16_49xc_0/relu_40oi', features=model_block_resnet_16_49xc_0_batch_normalize_38ms)
    model_block_resnet_16_49xc_0_conv_42qy = tf.nn.conv2d(input=model_block_resnet_16_49xc_0_relu_40oi, filters=trainable_params['model_block/resnet_16_49xc_0/conv_42qy/filters'], strides=1, padding='SAME', data_format='NHWC', dilations=1, name='model_block/resnet_16_49xc_0/conv_42qy/filters')
    model_block_resnet_16_49xc_0_conv = tf.nn.batch_normalization(x=model_block_resnet_16_49xc_0_conv_42qy, mean=trainable_params['model_block/resnet_16_49xc_0/conv/mean'], variance=trainable_params['model_block/resnet_16_49xc_0/conv/variance'], offset=trainable_params['model_block/resnet_16_49xc_0/conv/offset'], scale=trainable_params['model_block/resnet_16_49xc_0/conv/scale'], variance_epsilon=0.001, name='model_block/resnet_16_49xc_0/conv/variance')
    model_block_resnet_16_49xc_0_add_46ue = tf.math.add(x=[model_block_test_recipe_34im_relu_18so, model_block_resnet_16_49xc_0_conv][0], y=[model_block_test_recipe_34im_relu_18so, model_block_resnet_16_49xc_0_conv][1], name='model_block/resnet_16_49xc_0/add_46ue')
    model_block_resnet_16_49xc_0_relu_48wu = tf.nn.relu(name='model_block/resnet_16_49xc_0/relu_48wu', features=model_block_resnet_16_49xc_0_add_46ue)
    model_block_resnet_16_49xc_conv_36kc = tf.nn.conv2d(input=model_block_resnet_16_49xc_0_relu_48wu, filters=trainable_params['model_block/resnet_16_49xc/conv_36kc/filters'], strides=1, padding='SAME', data_format='NHWC', dilations=1, name='model_block/resnet_16_49xc/conv_36kc/filters')
    model_block_resnet_16_49xc_batch_normalize_38ms = tf.nn.batch_normalization(x=model_block_resnet_16_49xc_conv_36kc, mean=trainable_params['model_block/resnet_16_49xc/batch_normalize_38ms/mean'], variance=trainable_params['model_block/resnet_16_49xc/batch_normalize_38ms/variance'], offset=trainable_params['model_block/resnet_16_49xc/batch_normalize_38ms/offset'], scale=trainable_params['model_block/resnet_16_49xc/batch_normalize_38ms/scale'], variance_epsilon=0.001, name='model_block/resnet_16_49xc/batch_normalize_38ms/variance')
    model_block_resnet_16_49xc_relu_40oi = tf.nn.relu(name='model_block/resnet_16_49xc/relu_40oi', features=model_block_resnet_16_49xc_batch_normalize_38ms)
    model_block_resnet_16_49xc_conv_42qy = tf.nn.conv2d(input=model_block_resnet_16_49xc_relu_40oi, filters=trainable_params['model_block/resnet_16_49xc/conv_42qy/filters'], strides=1, padding='SAME', data_format='NHWC', dilations=1, name='model_block/resnet_16_49xc/conv_42qy/filters')
    model_block_resnet_16_49xc_conv = tf.nn.batch_normalization(x=model_block_resnet_16_49xc_conv_42qy, mean=trainable_params['model_block/resnet_16_49xc/conv/mean'], variance=trainable_params['model_block/resnet_16_49xc/conv/variance'], offset=trainable_params['model_block/resnet_16_49xc/conv/offset'], scale=trainable_params['model_block/resnet_16_49xc/conv/scale'], variance_epsilon=0.001, name='model_block/resnet_16_49xc/conv/variance')
    model_block_resnet_16_49xc_add_46ue = tf.math.add(x=[model_block_resnet_16_49xc_0_relu_48wu, model_block_resnet_16_49xc_conv][0], y=[model_block_resnet_16_49xc_0_relu_48wu, model_block_resnet_16_49xc_conv][1], name='model_block/resnet_16_49xc/add_46ue')
    model_block_resnet_16_49xc_relu_48wu = tf.nn.relu(name='model_block/resnet_16_49xc/relu_48wu', features=model_block_resnet_16_49xc_add_46ue)
    model_block_relu_51zs = tf.nn.relu(name='model_block/relu_51zs', features=model_block_resnet_16_49xc_relu_48wu)
    model_block_dropout_53bi = tf.nn.dropout(x=model_block_relu_51zs, rate=0.2, noise_shape=None, seed=None, name='model_block/dropout_53bi')
    model_block_batch_normalize_55dy = tf.nn.batch_normalization(x=model_block_dropout_53bi, mean=trainable_params['model_block/batch_normalize_55dy/mean'], variance=trainable_params['model_block/batch_normalize_55dy/variance'], offset=trainable_params['model_block/batch_normalize_55dy/offset'], scale=trainable_params['model_block/batch_normalize_55dy/scale'], variance_epsilon=0.001, name='model_block/batch_normalize_55dy/variance')
    model_block_conv_57fo = tf.nn.conv2d(input=model_block_batch_normalize_55dy, filters=trainable_params['model_block/conv_57fo/filters'], strides=1, padding='SAME', data_format='NHWC', dilations=1, name='model_block/conv_57fo/filters')
    model_block_max_pool2d_59he = tf.nn.max_pool(input=model_block_conv_57fo, ksize=3, strides=2, padding='VALID', data_format='NHWC', name='model_block/max_pool2d_59he')
    model_block_flatten_61ju = tf.reshape(tensor=model_block_max_pool2d_59he, shape=(-1, tf.math.reduce_prod(tf.convert_to_tensor([15, 15, 16]))), name='model_block/flatten_61ju')
    model_block_dense_63lk = tf.add(x=tf.matmul(a=model_block_flatten_61ju, b=trainable_params['model_block/dense_63lk/weights']), y=trainable_params['model_block/dense_63lk/bias'], name='model_block/dense_63lk/weights')
    model_block_d_1 = tf.nn.softmax(logits=model_block_dense_63lk, name='model_block/d_1')
    return model_block_d_1 


def get_loss(trainable_params, inputs):
    loss_block_cross_0 = tf.nn.softmax_cross_entropy_with_logits(labels=inputs[0], logits=inputs[1], axis=-1, name='loss_block/cross_0')
    loss_block_regularizer = 0.002*sum(list(map(lambda x: tf.nn.l2_loss(t=trainable_params[x], name='loss_block/regularizer'), ['model_block/test_recipe_34im/conv_6gw/filters', 'model_block/test_recipe_34im/conv_12ms/filters', 'model_block/test_recipe_34im/resnet_16_33he/conv_20ue/filters', 'model_block/test_recipe_34im/resnet_16_33he/conv_26aa/filters', 'model_block/resnet_16_49xc_0/conv_36kc/filters', 'model_block/resnet_16_49xc_0/conv_42qy/filters', 'model_block/resnet_16_49xc/conv_36kc/filters', 'model_block/resnet_16_49xc/conv_42qy/filters', 'model_block/conv_57fo/filters', 'model_block/dense_63lk/weights'])))
    loss_block_losses = tf.math.add(x=[loss_block_cross_0, loss_block_regularizer][0], y=[loss_block_cross_0, loss_block_regularizer][1], name='loss_block/losses')
    return loss_block_losses 


def get_optimizer():
    optimizer_block_solver_decay_exponential_decay = tf.keras.optimizers.schedules.ExponentialDecay(initial_learning_rate=0.0001, decay_steps=100000, decay_rate=0.96, staircase=True)
    optimizer_block_solver = tf.optimizers.Adam(learning_rate=optimizer_block_solver_decay_exponential_decay, beta_1=0.9, beta_2=0.999, epsilon=1e-08, name='optimizer_block/solver')
    return optimizer_block_solver 

from alex.alex.checkpoint import Checkpoint

C = Checkpoint("examples/configs/small3.yml",
               None,
               None)

ckpt = C.load()

trainable_params = get_trainable_params(tf_dtypes, tf)

from alex.alex import registry
var_list = registry.get_trainable_params_list(trainable_params)

optimizer = get_optimizer()


def inference(trainable_params, data_block_input_data):
    
    preds = tf.math.argmax(model(trainable_params, data_block_input_data), 1)
    return preds
    
def evaluation(trainable_params, labels, data_block_input_data):
    
    preds = inference(trainable_params, data_block_input_data)
    
    matches = tf.equal(preds, tf.math.argmax(labels, 1))
    perf = tf.reduce_mean(tf.cast(matches, tf.float32))
    
    loss = tf.reduce_mean(get_loss(trainable_params, [labels, preds]))
    return perf, loss
    
    
def train(trainable_params, labels, var_list, data_block_input_data):
    
    with tf.GradientTape() as tape:
        preds = model(trainable_params, data_block_input_data)
        gradients = tape.gradient(get_loss(trainable_params, [labels, preds]), var_list)
        optimizer.apply_gradients(zip(gradients, var_list))
    
    
def loop(trainloader, test_inputs, test_labels, var_list):
    
    for epoch in range(90):
        for i, (inputs, labels) in enumerate(trainloader):
            train(trainable_params, labels, var_list, inputs)
            if i % 500 == 499:
                results = evaluation(trainable_params, val_labels, val_inputs)
                
                tf.print(results)
    print('Finished Training')
    
    
    

from tensorflow import keras
from tensorflow.keras import datasets
import matplotlib.pyplot as plt
import numpy as np

num_classes = 10
(x_train, label_train), (x_val, label_val) = datasets.cifar10.load_data()

x_train = x_train.astype("float32") / 255
x_val = x_val.astype("float32") / 255

y_train = keras.utils.to_categorical(label_train, num_classes)
y_val = keras.utils.to_categorical(label_val, num_classes)


class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']

batch_size = 100

train_loss_results = []
train_accuracy_results = []

trainloader = tf.data.Dataset.from_tensor_slices(
    (x_train, y_train)).shuffle(50000).batch(batch_size)

valloader = tf.data.Dataset.from_tensor_slices((x_val, y_val)).batch(10000)
for val_inputs, val_labels in valloader:
    break

loop(trainloader, val_inputs, val_labels, var_list)


