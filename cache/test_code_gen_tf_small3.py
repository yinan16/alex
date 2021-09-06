import tensorflow as tf
import numpy as np


tf_dtypes = {'float32': tf.float32, 'int8': tf.int8}


def get_trainable_params():
    trainable_params = dict()
    model_block_conv_6gw_filters_initializer_xavier_uniform = tf.keras.initializers.glorot_uniform(seed=1)(shape=(3, 3, 3, 16))
    model_block_conv_6gw_filters = tf.Variable(initial_value=model_block_conv_6gw_filters_initializer_xavier_uniform, trainable=True, caching_device=None, name='model_block/conv_6gw/filters', variable_def=None, dtype=tf_dtypes['float32'], import_scope=None, constraint=None, synchronization=tf.VariableSynchronization.AUTO, shape=None)
    trainable_params['model_block/conv_6gw/filters'] = model_block_conv_6gw_filters
    model_block_resnet_16_21vm_0_conv_8im_filters_initializer_xavier_uniform = tf.keras.initializers.glorot_uniform(seed=1)(shape=(3, 3, 16, 16))
    model_block_resnet_16_21vm_0_conv_8im_filters = tf.Variable(initial_value=model_block_resnet_16_21vm_0_conv_8im_filters_initializer_xavier_uniform, trainable=True, caching_device=None, name='model_block/resnet_16_21vm_0/conv_8im/filters', variable_def=None, dtype=tf_dtypes['float32'], import_scope=None, constraint=None, synchronization=tf.VariableSynchronization.AUTO, shape=None)
    trainable_params['model_block/resnet_16_21vm_0/conv_8im/filters'] = model_block_resnet_16_21vm_0_conv_8im_filters
    model_block_resnet_16_21vm_0_batch_normalize_10kc_mean_initializer_zeros_initializer = tf.zeros_initializer()(shape=[16, ])
    model_block_resnet_16_21vm_0_batch_normalize_10kc_mean = tf.Variable(initial_value=model_block_resnet_16_21vm_0_batch_normalize_10kc_mean_initializer_zeros_initializer, trainable=False, caching_device=None, name='model_block/resnet_16_21vm_0/batch_normalize_10kc/mean', variable_def=None, dtype=tf_dtypes['float32'], import_scope=None, constraint=None, synchronization=tf.VariableSynchronization.AUTO, shape=None)
    trainable_params['model_block/resnet_16_21vm_0/batch_normalize_10kc/mean'] = model_block_resnet_16_21vm_0_batch_normalize_10kc_mean
    model_block_resnet_16_21vm_0_batch_normalize_10kc_offset_initializer_zeros_initializer = tf.zeros_initializer()(shape=[16, ])
    model_block_resnet_16_21vm_0_batch_normalize_10kc_offset = tf.Variable(initial_value=model_block_resnet_16_21vm_0_batch_normalize_10kc_offset_initializer_zeros_initializer, trainable=True, caching_device=None, name='model_block/resnet_16_21vm_0/batch_normalize_10kc/offset', variable_def=None, dtype=tf_dtypes['float32'], import_scope=None, constraint=None, synchronization=tf.VariableSynchronization.AUTO, shape=None)
    trainable_params['model_block/resnet_16_21vm_0/batch_normalize_10kc/offset'] = model_block_resnet_16_21vm_0_batch_normalize_10kc_offset
    model_block_resnet_16_21vm_0_batch_normalize_10kc_scale_initializer_ones_initializer = tf.ones_initializer()(shape=[16, ])
    model_block_resnet_16_21vm_0_batch_normalize_10kc_scale = tf.Variable(initial_value=model_block_resnet_16_21vm_0_batch_normalize_10kc_scale_initializer_ones_initializer, trainable=True, caching_device=None, name='model_block/resnet_16_21vm_0/batch_normalize_10kc/scale', variable_def=None, dtype=tf_dtypes['float32'], import_scope=None, constraint=None, synchronization=tf.VariableSynchronization.AUTO, shape=None)
    trainable_params['model_block/resnet_16_21vm_0/batch_normalize_10kc/scale'] = model_block_resnet_16_21vm_0_batch_normalize_10kc_scale
    model_block_resnet_16_21vm_0_batch_normalize_10kc_variance_initializer_ones_initializer = tf.ones_initializer()(shape=[16, ])
    model_block_resnet_16_21vm_0_batch_normalize_10kc_variance = tf.Variable(initial_value=model_block_resnet_16_21vm_0_batch_normalize_10kc_variance_initializer_ones_initializer, trainable=False, caching_device=None, name='model_block/resnet_16_21vm_0/batch_normalize_10kc/variance', variable_def=None, dtype=tf_dtypes['float32'], import_scope=None, constraint=None, synchronization=tf.VariableSynchronization.AUTO, shape=None)
    trainable_params['model_block/resnet_16_21vm_0/batch_normalize_10kc/variance'] = model_block_resnet_16_21vm_0_batch_normalize_10kc_variance
    model_block_resnet_16_21vm_0_conv_14oi_filters_initializer_xavier_uniform = tf.keras.initializers.glorot_uniform(seed=1)(shape=(3, 3, 16, 16))
    model_block_resnet_16_21vm_0_conv_14oi_filters = tf.Variable(initial_value=model_block_resnet_16_21vm_0_conv_14oi_filters_initializer_xavier_uniform, trainable=True, caching_device=None, name='model_block/resnet_16_21vm_0/conv_14oi/filters', variable_def=None, dtype=tf_dtypes['float32'], import_scope=None, constraint=None, synchronization=tf.VariableSynchronization.AUTO, shape=None)
    trainable_params['model_block/resnet_16_21vm_0/conv_14oi/filters'] = model_block_resnet_16_21vm_0_conv_14oi_filters
    model_block_resnet_16_21vm_0_conv_mean_initializer_zeros_initializer = tf.zeros_initializer()(shape=[16, ])
    model_block_resnet_16_21vm_0_conv_mean = tf.Variable(initial_value=model_block_resnet_16_21vm_0_conv_mean_initializer_zeros_initializer, trainable=False, caching_device=None, name='model_block/resnet_16_21vm_0/conv/mean', variable_def=None, dtype=tf_dtypes['float32'], import_scope=None, constraint=None, synchronization=tf.VariableSynchronization.AUTO, shape=None)
    trainable_params['model_block/resnet_16_21vm_0/conv/mean'] = model_block_resnet_16_21vm_0_conv_mean
    model_block_resnet_16_21vm_0_conv_offset_initializer_zeros_initializer = tf.zeros_initializer()(shape=[16, ])
    model_block_resnet_16_21vm_0_conv_offset = tf.Variable(initial_value=model_block_resnet_16_21vm_0_conv_offset_initializer_zeros_initializer, trainable=True, caching_device=None, name='model_block/resnet_16_21vm_0/conv/offset', variable_def=None, dtype=tf_dtypes['float32'], import_scope=None, constraint=None, synchronization=tf.VariableSynchronization.AUTO, shape=None)
    trainable_params['model_block/resnet_16_21vm_0/conv/offset'] = model_block_resnet_16_21vm_0_conv_offset
    model_block_resnet_16_21vm_0_conv_scale_initializer_ones_initializer = tf.ones_initializer()(shape=[16, ])
    model_block_resnet_16_21vm_0_conv_scale = tf.Variable(initial_value=model_block_resnet_16_21vm_0_conv_scale_initializer_ones_initializer, trainable=True, caching_device=None, name='model_block/resnet_16_21vm_0/conv/scale', variable_def=None, dtype=tf_dtypes['float32'], import_scope=None, constraint=None, synchronization=tf.VariableSynchronization.AUTO, shape=None)
    trainable_params['model_block/resnet_16_21vm_0/conv/scale'] = model_block_resnet_16_21vm_0_conv_scale
    model_block_resnet_16_21vm_0_conv_variance_initializer_ones_initializer = tf.ones_initializer()(shape=[16, ])
    model_block_resnet_16_21vm_0_conv_variance = tf.Variable(initial_value=model_block_resnet_16_21vm_0_conv_variance_initializer_ones_initializer, trainable=False, caching_device=None, name='model_block/resnet_16_21vm_0/conv/variance', variable_def=None, dtype=tf_dtypes['float32'], import_scope=None, constraint=None, synchronization=tf.VariableSynchronization.AUTO, shape=None)
    trainable_params['model_block/resnet_16_21vm_0/conv/variance'] = model_block_resnet_16_21vm_0_conv_variance
    model_block_resnet_16_21vm_conv_8im_filters_initializer_xavier_uniform = tf.keras.initializers.glorot_uniform(seed=1)(shape=(3, 3, 16, 16))
    model_block_resnet_16_21vm_conv_8im_filters = tf.Variable(initial_value=model_block_resnet_16_21vm_conv_8im_filters_initializer_xavier_uniform, trainable=True, caching_device=None, name='model_block/resnet_16_21vm/conv_8im/filters', variable_def=None, dtype=tf_dtypes['float32'], import_scope=None, constraint=None, synchronization=tf.VariableSynchronization.AUTO, shape=None)
    trainable_params['model_block/resnet_16_21vm/conv_8im/filters'] = model_block_resnet_16_21vm_conv_8im_filters
    model_block_resnet_16_21vm_batch_normalize_10kc_mean_initializer_zeros_initializer = tf.zeros_initializer()(shape=[16, ])
    model_block_resnet_16_21vm_batch_normalize_10kc_mean = tf.Variable(initial_value=model_block_resnet_16_21vm_batch_normalize_10kc_mean_initializer_zeros_initializer, trainable=False, caching_device=None, name='model_block/resnet_16_21vm/batch_normalize_10kc/mean', variable_def=None, dtype=tf_dtypes['float32'], import_scope=None, constraint=None, synchronization=tf.VariableSynchronization.AUTO, shape=None)
    trainable_params['model_block/resnet_16_21vm/batch_normalize_10kc/mean'] = model_block_resnet_16_21vm_batch_normalize_10kc_mean
    model_block_resnet_16_21vm_batch_normalize_10kc_offset_initializer_zeros_initializer = tf.zeros_initializer()(shape=[16, ])
    model_block_resnet_16_21vm_batch_normalize_10kc_offset = tf.Variable(initial_value=model_block_resnet_16_21vm_batch_normalize_10kc_offset_initializer_zeros_initializer, trainable=True, caching_device=None, name='model_block/resnet_16_21vm/batch_normalize_10kc/offset', variable_def=None, dtype=tf_dtypes['float32'], import_scope=None, constraint=None, synchronization=tf.VariableSynchronization.AUTO, shape=None)
    trainable_params['model_block/resnet_16_21vm/batch_normalize_10kc/offset'] = model_block_resnet_16_21vm_batch_normalize_10kc_offset
    model_block_resnet_16_21vm_batch_normalize_10kc_scale_initializer_ones_initializer = tf.ones_initializer()(shape=[16, ])
    model_block_resnet_16_21vm_batch_normalize_10kc_scale = tf.Variable(initial_value=model_block_resnet_16_21vm_batch_normalize_10kc_scale_initializer_ones_initializer, trainable=True, caching_device=None, name='model_block/resnet_16_21vm/batch_normalize_10kc/scale', variable_def=None, dtype=tf_dtypes['float32'], import_scope=None, constraint=None, synchronization=tf.VariableSynchronization.AUTO, shape=None)
    trainable_params['model_block/resnet_16_21vm/batch_normalize_10kc/scale'] = model_block_resnet_16_21vm_batch_normalize_10kc_scale
    model_block_resnet_16_21vm_batch_normalize_10kc_variance_initializer_ones_initializer = tf.ones_initializer()(shape=[16, ])
    model_block_resnet_16_21vm_batch_normalize_10kc_variance = tf.Variable(initial_value=model_block_resnet_16_21vm_batch_normalize_10kc_variance_initializer_ones_initializer, trainable=False, caching_device=None, name='model_block/resnet_16_21vm/batch_normalize_10kc/variance', variable_def=None, dtype=tf_dtypes['float32'], import_scope=None, constraint=None, synchronization=tf.VariableSynchronization.AUTO, shape=None)
    trainable_params['model_block/resnet_16_21vm/batch_normalize_10kc/variance'] = model_block_resnet_16_21vm_batch_normalize_10kc_variance
    model_block_resnet_16_21vm_conv_14oi_filters_initializer_xavier_uniform = tf.keras.initializers.glorot_uniform(seed=1)(shape=(3, 3, 16, 16))
    model_block_resnet_16_21vm_conv_14oi_filters = tf.Variable(initial_value=model_block_resnet_16_21vm_conv_14oi_filters_initializer_xavier_uniform, trainable=True, caching_device=None, name='model_block/resnet_16_21vm/conv_14oi/filters', variable_def=None, dtype=tf_dtypes['float32'], import_scope=None, constraint=None, synchronization=tf.VariableSynchronization.AUTO, shape=None)
    trainable_params['model_block/resnet_16_21vm/conv_14oi/filters'] = model_block_resnet_16_21vm_conv_14oi_filters
    model_block_resnet_16_21vm_conv_mean_initializer_zeros_initializer = tf.zeros_initializer()(shape=[16, ])
    model_block_resnet_16_21vm_conv_mean = tf.Variable(initial_value=model_block_resnet_16_21vm_conv_mean_initializer_zeros_initializer, trainable=False, caching_device=None, name='model_block/resnet_16_21vm/conv/mean', variable_def=None, dtype=tf_dtypes['float32'], import_scope=None, constraint=None, synchronization=tf.VariableSynchronization.AUTO, shape=None)
    trainable_params['model_block/resnet_16_21vm/conv/mean'] = model_block_resnet_16_21vm_conv_mean
    model_block_resnet_16_21vm_conv_offset_initializer_zeros_initializer = tf.zeros_initializer()(shape=[16, ])
    model_block_resnet_16_21vm_conv_offset = tf.Variable(initial_value=model_block_resnet_16_21vm_conv_offset_initializer_zeros_initializer, trainable=True, caching_device=None, name='model_block/resnet_16_21vm/conv/offset', variable_def=None, dtype=tf_dtypes['float32'], import_scope=None, constraint=None, synchronization=tf.VariableSynchronization.AUTO, shape=None)
    trainable_params['model_block/resnet_16_21vm/conv/offset'] = model_block_resnet_16_21vm_conv_offset
    model_block_resnet_16_21vm_conv_scale_initializer_ones_initializer = tf.ones_initializer()(shape=[16, ])
    model_block_resnet_16_21vm_conv_scale = tf.Variable(initial_value=model_block_resnet_16_21vm_conv_scale_initializer_ones_initializer, trainable=True, caching_device=None, name='model_block/resnet_16_21vm/conv/scale', variable_def=None, dtype=tf_dtypes['float32'], import_scope=None, constraint=None, synchronization=tf.VariableSynchronization.AUTO, shape=None)
    trainable_params['model_block/resnet_16_21vm/conv/scale'] = model_block_resnet_16_21vm_conv_scale
    model_block_resnet_16_21vm_conv_variance_initializer_ones_initializer = tf.ones_initializer()(shape=[16, ])
    model_block_resnet_16_21vm_conv_variance = tf.Variable(initial_value=model_block_resnet_16_21vm_conv_variance_initializer_ones_initializer, trainable=False, caching_device=None, name='model_block/resnet_16_21vm/conv/variance', variable_def=None, dtype=tf_dtypes['float32'], import_scope=None, constraint=None, synchronization=tf.VariableSynchronization.AUTO, shape=None)
    trainable_params['model_block/resnet_16_21vm/conv/variance'] = model_block_resnet_16_21vm_conv_variance
    model_block_test_recipe_51zs_conv_23xc_filters_initializer_xavier_uniform = tf.keras.initializers.glorot_uniform(seed=1)(shape=(3, 3, 16, 16))
    model_block_test_recipe_51zs_conv_23xc_filters = tf.Variable(initial_value=model_block_test_recipe_51zs_conv_23xc_filters_initializer_xavier_uniform, trainable=True, caching_device=None, name='model_block/test_recipe_51zs/conv_23xc/filters', variable_def=None, dtype=tf_dtypes['float32'], import_scope=None, constraint=None, synchronization=tf.VariableSynchronization.AUTO, shape=None)
    trainable_params['model_block/test_recipe_51zs/conv_23xc/filters'] = model_block_test_recipe_51zs_conv_23xc_filters
    model_block_test_recipe_51zs_batch_normalize_25zs_mean_initializer_zeros_initializer = tf.zeros_initializer()(shape=[16, ])
    model_block_test_recipe_51zs_batch_normalize_25zs_mean = tf.Variable(initial_value=model_block_test_recipe_51zs_batch_normalize_25zs_mean_initializer_zeros_initializer, trainable=False, caching_device=None, name='model_block/test_recipe_51zs/batch_normalize_25zs/mean', variable_def=None, dtype=tf_dtypes['float32'], import_scope=None, constraint=None, synchronization=tf.VariableSynchronization.AUTO, shape=None)
    trainable_params['model_block/test_recipe_51zs/batch_normalize_25zs/mean'] = model_block_test_recipe_51zs_batch_normalize_25zs_mean
    model_block_test_recipe_51zs_batch_normalize_25zs_offset_initializer_zeros_initializer = tf.zeros_initializer()(shape=[16, ])
    model_block_test_recipe_51zs_batch_normalize_25zs_offset = tf.Variable(initial_value=model_block_test_recipe_51zs_batch_normalize_25zs_offset_initializer_zeros_initializer, trainable=True, caching_device=None, name='model_block/test_recipe_51zs/batch_normalize_25zs/offset', variable_def=None, dtype=tf_dtypes['float32'], import_scope=None, constraint=None, synchronization=tf.VariableSynchronization.AUTO, shape=None)
    trainable_params['model_block/test_recipe_51zs/batch_normalize_25zs/offset'] = model_block_test_recipe_51zs_batch_normalize_25zs_offset
    model_block_test_recipe_51zs_batch_normalize_25zs_scale_initializer_ones_initializer = tf.ones_initializer()(shape=[16, ])
    model_block_test_recipe_51zs_batch_normalize_25zs_scale = tf.Variable(initial_value=model_block_test_recipe_51zs_batch_normalize_25zs_scale_initializer_ones_initializer, trainable=True, caching_device=None, name='model_block/test_recipe_51zs/batch_normalize_25zs/scale', variable_def=None, dtype=tf_dtypes['float32'], import_scope=None, constraint=None, synchronization=tf.VariableSynchronization.AUTO, shape=None)
    trainable_params['model_block/test_recipe_51zs/batch_normalize_25zs/scale'] = model_block_test_recipe_51zs_batch_normalize_25zs_scale
    model_block_test_recipe_51zs_batch_normalize_25zs_variance_initializer_ones_initializer = tf.ones_initializer()(shape=[16, ])
    model_block_test_recipe_51zs_batch_normalize_25zs_variance = tf.Variable(initial_value=model_block_test_recipe_51zs_batch_normalize_25zs_variance_initializer_ones_initializer, trainable=False, caching_device=None, name='model_block/test_recipe_51zs/batch_normalize_25zs/variance', variable_def=None, dtype=tf_dtypes['float32'], import_scope=None, constraint=None, synchronization=tf.VariableSynchronization.AUTO, shape=None)
    trainable_params['model_block/test_recipe_51zs/batch_normalize_25zs/variance'] = model_block_test_recipe_51zs_batch_normalize_25zs_variance
    model_block_test_recipe_51zs_conv_29dy_filters_initializer_xavier_uniform = tf.keras.initializers.glorot_uniform(seed=1)(shape=(3, 3, 16, 16))
    model_block_test_recipe_51zs_conv_29dy_filters = tf.Variable(initial_value=model_block_test_recipe_51zs_conv_29dy_filters_initializer_xavier_uniform, trainable=True, caching_device=None, name='model_block/test_recipe_51zs/conv_29dy/filters', variable_def=None, dtype=tf_dtypes['float32'], import_scope=None, constraint=None, synchronization=tf.VariableSynchronization.AUTO, shape=None)
    trainable_params['model_block/test_recipe_51zs/conv_29dy/filters'] = model_block_test_recipe_51zs_conv_29dy_filters
    model_block_test_recipe_51zs_conv_mean_initializer_zeros_initializer = tf.zeros_initializer()(shape=[16, ])
    model_block_test_recipe_51zs_conv_mean = tf.Variable(initial_value=model_block_test_recipe_51zs_conv_mean_initializer_zeros_initializer, trainable=False, caching_device=None, name='model_block/test_recipe_51zs/conv/mean', variable_def=None, dtype=tf_dtypes['float32'], import_scope=None, constraint=None, synchronization=tf.VariableSynchronization.AUTO, shape=None)
    trainable_params['model_block/test_recipe_51zs/conv/mean'] = model_block_test_recipe_51zs_conv_mean
    model_block_test_recipe_51zs_conv_offset_initializer_zeros_initializer = tf.zeros_initializer()(shape=[16, ])
    model_block_test_recipe_51zs_conv_offset = tf.Variable(initial_value=model_block_test_recipe_51zs_conv_offset_initializer_zeros_initializer, trainable=True, caching_device=None, name='model_block/test_recipe_51zs/conv/offset', variable_def=None, dtype=tf_dtypes['float32'], import_scope=None, constraint=None, synchronization=tf.VariableSynchronization.AUTO, shape=None)
    trainable_params['model_block/test_recipe_51zs/conv/offset'] = model_block_test_recipe_51zs_conv_offset
    model_block_test_recipe_51zs_conv_scale_initializer_ones_initializer = tf.ones_initializer()(shape=[16, ])
    model_block_test_recipe_51zs_conv_scale = tf.Variable(initial_value=model_block_test_recipe_51zs_conv_scale_initializer_ones_initializer, trainable=True, caching_device=None, name='model_block/test_recipe_51zs/conv/scale', variable_def=None, dtype=tf_dtypes['float32'], import_scope=None, constraint=None, synchronization=tf.VariableSynchronization.AUTO, shape=None)
    trainable_params['model_block/test_recipe_51zs/conv/scale'] = model_block_test_recipe_51zs_conv_scale
    model_block_test_recipe_51zs_conv_variance_initializer_ones_initializer = tf.ones_initializer()(shape=[16, ])
    model_block_test_recipe_51zs_conv_variance = tf.Variable(initial_value=model_block_test_recipe_51zs_conv_variance_initializer_ones_initializer, trainable=False, caching_device=None, name='model_block/test_recipe_51zs/conv/variance', variable_def=None, dtype=tf_dtypes['float32'], import_scope=None, constraint=None, synchronization=tf.VariableSynchronization.AUTO, shape=None)
    trainable_params['model_block/test_recipe_51zs/conv/variance'] = model_block_test_recipe_51zs_conv_variance
    model_block_test_recipe_51zs_resnet_16_50yk_conv_37lk_filters_initializer_xavier_uniform = tf.keras.initializers.glorot_uniform(seed=1)(shape=(3, 3, 16, 16))
    model_block_test_recipe_51zs_resnet_16_50yk_conv_37lk_filters = tf.Variable(initial_value=model_block_test_recipe_51zs_resnet_16_50yk_conv_37lk_filters_initializer_xavier_uniform, trainable=True, caching_device=None, name='model_block/test_recipe_51zs/resnet_16_50yk/conv_37lk/filters', variable_def=None, dtype=tf_dtypes['float32'], import_scope=None, constraint=None, synchronization=tf.VariableSynchronization.AUTO, shape=None)
    trainable_params['model_block/test_recipe_51zs/resnet_16_50yk/conv_37lk/filters'] = model_block_test_recipe_51zs_resnet_16_50yk_conv_37lk_filters
    model_block_test_recipe_51zs_resnet_16_50yk_batch_normalize_39na_mean_initializer_zeros_initializer = tf.zeros_initializer()(shape=[16, ])
    model_block_test_recipe_51zs_resnet_16_50yk_batch_normalize_39na_mean = tf.Variable(initial_value=model_block_test_recipe_51zs_resnet_16_50yk_batch_normalize_39na_mean_initializer_zeros_initializer, trainable=False, caching_device=None, name='model_block/test_recipe_51zs/resnet_16_50yk/batch_normalize_39na/mean', variable_def=None, dtype=tf_dtypes['float32'], import_scope=None, constraint=None, synchronization=tf.VariableSynchronization.AUTO, shape=None)
    trainable_params['model_block/test_recipe_51zs/resnet_16_50yk/batch_normalize_39na/mean'] = model_block_test_recipe_51zs_resnet_16_50yk_batch_normalize_39na_mean
    model_block_test_recipe_51zs_resnet_16_50yk_batch_normalize_39na_offset_initializer_zeros_initializer = tf.zeros_initializer()(shape=[16, ])
    model_block_test_recipe_51zs_resnet_16_50yk_batch_normalize_39na_offset = tf.Variable(initial_value=model_block_test_recipe_51zs_resnet_16_50yk_batch_normalize_39na_offset_initializer_zeros_initializer, trainable=True, caching_device=None, name='model_block/test_recipe_51zs/resnet_16_50yk/batch_normalize_39na/offset', variable_def=None, dtype=tf_dtypes['float32'], import_scope=None, constraint=None, synchronization=tf.VariableSynchronization.AUTO, shape=None)
    trainable_params['model_block/test_recipe_51zs/resnet_16_50yk/batch_normalize_39na/offset'] = model_block_test_recipe_51zs_resnet_16_50yk_batch_normalize_39na_offset
    model_block_test_recipe_51zs_resnet_16_50yk_batch_normalize_39na_scale_initializer_ones_initializer = tf.ones_initializer()(shape=[16, ])
    model_block_test_recipe_51zs_resnet_16_50yk_batch_normalize_39na_scale = tf.Variable(initial_value=model_block_test_recipe_51zs_resnet_16_50yk_batch_normalize_39na_scale_initializer_ones_initializer, trainable=True, caching_device=None, name='model_block/test_recipe_51zs/resnet_16_50yk/batch_normalize_39na/scale', variable_def=None, dtype=tf_dtypes['float32'], import_scope=None, constraint=None, synchronization=tf.VariableSynchronization.AUTO, shape=None)
    trainable_params['model_block/test_recipe_51zs/resnet_16_50yk/batch_normalize_39na/scale'] = model_block_test_recipe_51zs_resnet_16_50yk_batch_normalize_39na_scale
    model_block_test_recipe_51zs_resnet_16_50yk_batch_normalize_39na_variance_initializer_ones_initializer = tf.ones_initializer()(shape=[16, ])
    model_block_test_recipe_51zs_resnet_16_50yk_batch_normalize_39na_variance = tf.Variable(initial_value=model_block_test_recipe_51zs_resnet_16_50yk_batch_normalize_39na_variance_initializer_ones_initializer, trainable=False, caching_device=None, name='model_block/test_recipe_51zs/resnet_16_50yk/batch_normalize_39na/variance', variable_def=None, dtype=tf_dtypes['float32'], import_scope=None, constraint=None, synchronization=tf.VariableSynchronization.AUTO, shape=None)
    trainable_params['model_block/test_recipe_51zs/resnet_16_50yk/batch_normalize_39na/variance'] = model_block_test_recipe_51zs_resnet_16_50yk_batch_normalize_39na_variance
    model_block_test_recipe_51zs_resnet_16_50yk_conv_43rg_filters_initializer_xavier_uniform = tf.keras.initializers.glorot_uniform(seed=1)(shape=(3, 3, 16, 16))
    model_block_test_recipe_51zs_resnet_16_50yk_conv_43rg_filters = tf.Variable(initial_value=model_block_test_recipe_51zs_resnet_16_50yk_conv_43rg_filters_initializer_xavier_uniform, trainable=True, caching_device=None, name='model_block/test_recipe_51zs/resnet_16_50yk/conv_43rg/filters', variable_def=None, dtype=tf_dtypes['float32'], import_scope=None, constraint=None, synchronization=tf.VariableSynchronization.AUTO, shape=None)
    trainable_params['model_block/test_recipe_51zs/resnet_16_50yk/conv_43rg/filters'] = model_block_test_recipe_51zs_resnet_16_50yk_conv_43rg_filters
    model_block_test_recipe_51zs_resnet_16_50yk_conv_mean_initializer_zeros_initializer = tf.zeros_initializer()(shape=[16, ])
    model_block_test_recipe_51zs_resnet_16_50yk_conv_mean = tf.Variable(initial_value=model_block_test_recipe_51zs_resnet_16_50yk_conv_mean_initializer_zeros_initializer, trainable=False, caching_device=None, name='model_block/test_recipe_51zs/resnet_16_50yk/conv/mean', variable_def=None, dtype=tf_dtypes['float32'], import_scope=None, constraint=None, synchronization=tf.VariableSynchronization.AUTO, shape=None)
    trainable_params['model_block/test_recipe_51zs/resnet_16_50yk/conv/mean'] = model_block_test_recipe_51zs_resnet_16_50yk_conv_mean
    model_block_test_recipe_51zs_resnet_16_50yk_conv_offset_initializer_zeros_initializer = tf.zeros_initializer()(shape=[16, ])
    model_block_test_recipe_51zs_resnet_16_50yk_conv_offset = tf.Variable(initial_value=model_block_test_recipe_51zs_resnet_16_50yk_conv_offset_initializer_zeros_initializer, trainable=True, caching_device=None, name='model_block/test_recipe_51zs/resnet_16_50yk/conv/offset', variable_def=None, dtype=tf_dtypes['float32'], import_scope=None, constraint=None, synchronization=tf.VariableSynchronization.AUTO, shape=None)
    trainable_params['model_block/test_recipe_51zs/resnet_16_50yk/conv/offset'] = model_block_test_recipe_51zs_resnet_16_50yk_conv_offset
    model_block_test_recipe_51zs_resnet_16_50yk_conv_scale_initializer_ones_initializer = tf.ones_initializer()(shape=[16, ])
    model_block_test_recipe_51zs_resnet_16_50yk_conv_scale = tf.Variable(initial_value=model_block_test_recipe_51zs_resnet_16_50yk_conv_scale_initializer_ones_initializer, trainable=True, caching_device=None, name='model_block/test_recipe_51zs/resnet_16_50yk/conv/scale', variable_def=None, dtype=tf_dtypes['float32'], import_scope=None, constraint=None, synchronization=tf.VariableSynchronization.AUTO, shape=None)
    trainable_params['model_block/test_recipe_51zs/resnet_16_50yk/conv/scale'] = model_block_test_recipe_51zs_resnet_16_50yk_conv_scale
    model_block_test_recipe_51zs_resnet_16_50yk_conv_variance_initializer_ones_initializer = tf.ones_initializer()(shape=[16, ])
    model_block_test_recipe_51zs_resnet_16_50yk_conv_variance = tf.Variable(initial_value=model_block_test_recipe_51zs_resnet_16_50yk_conv_variance_initializer_ones_initializer, trainable=False, caching_device=None, name='model_block/test_recipe_51zs/resnet_16_50yk/conv/variance', variable_def=None, dtype=tf_dtypes['float32'], import_scope=None, constraint=None, synchronization=tf.VariableSynchronization.AUTO, shape=None)
    trainable_params['model_block/test_recipe_51zs/resnet_16_50yk/conv/variance'] = model_block_test_recipe_51zs_resnet_16_50yk_conv_variance
    model_block_conv_53bi_filters_initializer_xavier_uniform = tf.keras.initializers.glorot_uniform(seed=1)(shape=(3, 3, 16, 16))
    model_block_conv_53bi_filters = tf.Variable(initial_value=model_block_conv_53bi_filters_initializer_xavier_uniform, trainable=True, caching_device=None, name='model_block/conv_53bi/filters', variable_def=None, dtype=tf_dtypes['float32'], import_scope=None, constraint=None, synchronization=tf.VariableSynchronization.AUTO, shape=None)
    trainable_params['model_block/conv_53bi/filters'] = model_block_conv_53bi_filters
    model_block_batch_normalize_59he_mean_initializer_zeros_initializer = tf.zeros_initializer()(shape=[16, ])
    model_block_batch_normalize_59he_mean = tf.Variable(initial_value=model_block_batch_normalize_59he_mean_initializer_zeros_initializer, trainable=False, caching_device=None, name='model_block/batch_normalize_59he/mean', variable_def=None, dtype=tf_dtypes['float32'], import_scope=None, constraint=None, synchronization=tf.VariableSynchronization.AUTO, shape=None)
    trainable_params['model_block/batch_normalize_59he/mean'] = model_block_batch_normalize_59he_mean
    model_block_batch_normalize_59he_offset_initializer_zeros_initializer = tf.zeros_initializer()(shape=[16, ])
    model_block_batch_normalize_59he_offset = tf.Variable(initial_value=model_block_batch_normalize_59he_offset_initializer_zeros_initializer, trainable=True, caching_device=None, name='model_block/batch_normalize_59he/offset', variable_def=None, dtype=tf_dtypes['float32'], import_scope=None, constraint=None, synchronization=tf.VariableSynchronization.AUTO, shape=None)
    trainable_params['model_block/batch_normalize_59he/offset'] = model_block_batch_normalize_59he_offset
    model_block_batch_normalize_59he_scale_initializer_ones_initializer = tf.ones_initializer()(shape=[16, ])
    model_block_batch_normalize_59he_scale = tf.Variable(initial_value=model_block_batch_normalize_59he_scale_initializer_ones_initializer, trainable=True, caching_device=None, name='model_block/batch_normalize_59he/scale', variable_def=None, dtype=tf_dtypes['float32'], import_scope=None, constraint=None, synchronization=tf.VariableSynchronization.AUTO, shape=None)
    trainable_params['model_block/batch_normalize_59he/scale'] = model_block_batch_normalize_59he_scale
    model_block_batch_normalize_59he_variance_initializer_ones_initializer = tf.ones_initializer()(shape=[16, ])
    model_block_batch_normalize_59he_variance = tf.Variable(initial_value=model_block_batch_normalize_59he_variance_initializer_ones_initializer, trainable=False, caching_device=None, name='model_block/batch_normalize_59he/variance', variable_def=None, dtype=tf_dtypes['float32'], import_scope=None, constraint=None, synchronization=tf.VariableSynchronization.AUTO, shape=None)
    trainable_params['model_block/batch_normalize_59he/variance'] = model_block_batch_normalize_59he_variance
    model_block_conv_61ju_filters_initializer_xavier_uniform = tf.keras.initializers.glorot_uniform(seed=1)(shape=(3, 3, 16, 16))
    model_block_conv_61ju_filters = tf.Variable(initial_value=model_block_conv_61ju_filters_initializer_xavier_uniform, trainable=True, caching_device=None, name='model_block/conv_61ju/filters', variable_def=None, dtype=tf_dtypes['float32'], import_scope=None, constraint=None, synchronization=tf.VariableSynchronization.AUTO, shape=None)
    trainable_params['model_block/conv_61ju/filters'] = model_block_conv_61ju_filters
    model_block_dense_67pq_bias_initializer_zeros_initializer = tf.zeros_initializer()(shape=[1, ])
    model_block_dense_67pq_bias = tf.Variable(initial_value=model_block_dense_67pq_bias_initializer_zeros_initializer, trainable=True, caching_device=None, name='model_block/dense_67pq/bias', variable_def=None, dtype=tf_dtypes['float32'], import_scope=None, constraint=None, synchronization=tf.VariableSynchronization.AUTO, shape=None)
    trainable_params['model_block/dense_67pq/bias'] = model_block_dense_67pq_bias
    model_block_dense_67pq_weights_initializer_xavier_uniform = tf.keras.initializers.glorot_uniform(seed=2)(shape=[14400, 10])
    model_block_dense_67pq_weights = tf.Variable(initial_value=model_block_dense_67pq_weights_initializer_xavier_uniform, trainable=True, caching_device=None, name='model_block/dense_67pq/weights', variable_def=None, dtype=tf_dtypes['float32'], import_scope=None, constraint=None, synchronization=tf.VariableSynchronization.AUTO, shape=None)
    trainable_params['model_block/dense_67pq/weights'] = model_block_dense_67pq_weights
    return trainable_params


def model(data_block_input_data, trainable_params):
    model_block_conv_6gw = tf.nn.conv2d(input=data_block_input_data, filters=trainable_params['model_block/conv_6gw/filters'], strides=1, padding='SAME', data_format='NHWC', dilations=1, name='model_block/conv_6gw')
    model_block_resnet_16_21vm_0_conv_8im = tf.nn.conv2d(input=model_block_conv_6gw, filters=trainable_params['model_block/resnet_16_21vm_0/conv_8im/filters'], strides=1, padding='SAME', data_format='NHWC', dilations=1, name='model_block/resnet_16_21vm_0/conv_8im')
    model_block_resnet_16_21vm_0_batch_normalize_10kc = tf.nn.batch_normalization(x=model_block_resnet_16_21vm_0_conv_8im, mean=trainable_params['model_block/resnet_16_21vm_0/batch_normalize_10kc/mean'], variance=trainable_params['model_block/resnet_16_21vm_0/batch_normalize_10kc/variance'], offset=trainable_params['model_block/resnet_16_21vm_0/batch_normalize_10kc/offset'], scale=trainable_params['model_block/resnet_16_21vm_0/batch_normalize_10kc/scale'], variance_epsilon=0.001, name='model_block/resnet_16_21vm_0/batch_normalize_10kc')
    model_block_resnet_16_21vm_0_relu_12ms = tf.nn.relu(name='model_block/resnet_16_21vm_0/relu_12ms', features=model_block_resnet_16_21vm_0_batch_normalize_10kc)
    model_block_resnet_16_21vm_0_conv_14oi = tf.nn.conv2d(input=model_block_resnet_16_21vm_0_relu_12ms, filters=trainable_params['model_block/resnet_16_21vm_0/conv_14oi/filters'], strides=1, padding='SAME', data_format='NHWC', dilations=1, name='model_block/resnet_16_21vm_0/conv_14oi')
    model_block_resnet_16_21vm_0_conv = tf.nn.batch_normalization(x=model_block_resnet_16_21vm_0_conv_14oi, mean=trainable_params['model_block/resnet_16_21vm_0/conv/mean'], variance=trainable_params['model_block/resnet_16_21vm_0/conv/variance'], offset=trainable_params['model_block/resnet_16_21vm_0/conv/offset'], scale=trainable_params['model_block/resnet_16_21vm_0/conv/scale'], variance_epsilon=0.001, name='model_block/resnet_16_21vm_0/conv')
    model_block_resnet_16_21vm_0_add_18so = tf.math.add(x=[model_block_conv_6gw, model_block_resnet_16_21vm_0_conv][0], y=[model_block_conv_6gw, model_block_resnet_16_21vm_0_conv][1], name='model_block/resnet_16_21vm_0/add_18so')
    model_block_resnet_16_21vm_0_relu_20ue = tf.nn.relu(name='model_block/resnet_16_21vm_0/relu_20ue', features=model_block_resnet_16_21vm_0_add_18so)
    model_block_resnet_16_21vm_conv_8im = tf.nn.conv2d(input=model_block_resnet_16_21vm_0_relu_20ue, filters=trainable_params['model_block/resnet_16_21vm/conv_8im/filters'], strides=1, padding='SAME', data_format='NHWC', dilations=1, name='model_block/resnet_16_21vm/conv_8im')
    model_block_resnet_16_21vm_batch_normalize_10kc = tf.nn.batch_normalization(x=model_block_resnet_16_21vm_conv_8im, mean=trainable_params['model_block/resnet_16_21vm/batch_normalize_10kc/mean'], variance=trainable_params['model_block/resnet_16_21vm/batch_normalize_10kc/variance'], offset=trainable_params['model_block/resnet_16_21vm/batch_normalize_10kc/offset'], scale=trainable_params['model_block/resnet_16_21vm/batch_normalize_10kc/scale'], variance_epsilon=0.001, name='model_block/resnet_16_21vm/batch_normalize_10kc')
    model_block_resnet_16_21vm_relu_12ms = tf.nn.relu(name='model_block/resnet_16_21vm/relu_12ms', features=model_block_resnet_16_21vm_batch_normalize_10kc)
    model_block_resnet_16_21vm_conv_14oi = tf.nn.conv2d(input=model_block_resnet_16_21vm_relu_12ms, filters=trainable_params['model_block/resnet_16_21vm/conv_14oi/filters'], strides=1, padding='SAME', data_format='NHWC', dilations=1, name='model_block/resnet_16_21vm/conv_14oi')
    model_block_resnet_16_21vm_conv = tf.nn.batch_normalization(x=model_block_resnet_16_21vm_conv_14oi, mean=trainable_params['model_block/resnet_16_21vm/conv/mean'], variance=trainable_params['model_block/resnet_16_21vm/conv/variance'], offset=trainable_params['model_block/resnet_16_21vm/conv/offset'], scale=trainable_params['model_block/resnet_16_21vm/conv/scale'], variance_epsilon=0.001, name='model_block/resnet_16_21vm/conv')
    model_block_resnet_16_21vm_add_18so = tf.math.add(x=[model_block_resnet_16_21vm_0_relu_20ue, model_block_resnet_16_21vm_conv][0], y=[model_block_resnet_16_21vm_0_relu_20ue, model_block_resnet_16_21vm_conv][1], name='model_block/resnet_16_21vm/add_18so')
    model_block_resnet_16_21vm_relu_20ue = tf.nn.relu(name='model_block/resnet_16_21vm/relu_20ue', features=model_block_resnet_16_21vm_add_18so)
    model_block_test_recipe_51zs_conv_23xc = tf.nn.conv2d(input=model_block_resnet_16_21vm_relu_20ue, filters=trainable_params['model_block/test_recipe_51zs/conv_23xc/filters'], strides=1, padding='SAME', data_format='NHWC', dilations=1, name='model_block/test_recipe_51zs/conv_23xc')
    model_block_test_recipe_51zs_batch_normalize_25zs = tf.nn.batch_normalization(x=model_block_test_recipe_51zs_conv_23xc, mean=trainable_params['model_block/test_recipe_51zs/batch_normalize_25zs/mean'], variance=trainable_params['model_block/test_recipe_51zs/batch_normalize_25zs/variance'], offset=trainable_params['model_block/test_recipe_51zs/batch_normalize_25zs/offset'], scale=trainable_params['model_block/test_recipe_51zs/batch_normalize_25zs/scale'], variance_epsilon=0.001, name='model_block/test_recipe_51zs/batch_normalize_25zs')
    model_block_test_recipe_51zs_relu_27bi = tf.nn.relu(name='model_block/test_recipe_51zs/relu_27bi', features=model_block_test_recipe_51zs_batch_normalize_25zs)
    model_block_test_recipe_51zs_conv_29dy = tf.nn.conv2d(input=model_block_test_recipe_51zs_relu_27bi, filters=trainable_params['model_block/test_recipe_51zs/conv_29dy/filters'], strides=1, padding='SAME', data_format='NHWC', dilations=1, name='model_block/test_recipe_51zs/conv_29dy')
    model_block_test_recipe_51zs_conv = tf.nn.batch_normalization(x=model_block_test_recipe_51zs_conv_29dy, mean=trainable_params['model_block/test_recipe_51zs/conv/mean'], variance=trainable_params['model_block/test_recipe_51zs/conv/variance'], offset=trainable_params['model_block/test_recipe_51zs/conv/offset'], scale=trainable_params['model_block/test_recipe_51zs/conv/scale'], variance_epsilon=0.001, name='model_block/test_recipe_51zs/conv')
    model_block_test_recipe_51zs_add_33he = tf.math.add(x=[model_block_resnet_16_21vm_relu_20ue, model_block_test_recipe_51zs_conv][0], y=[model_block_resnet_16_21vm_relu_20ue, model_block_test_recipe_51zs_conv][1], name='model_block/test_recipe_51zs/add_33he')
    model_block_test_recipe_51zs_relu_35ju = tf.nn.relu(name='model_block/test_recipe_51zs/relu_35ju', features=model_block_test_recipe_51zs_add_33he)
    model_block_test_recipe_51zs_resnet_16_50yk_conv_37lk = tf.nn.conv2d(input=model_block_test_recipe_51zs_relu_35ju, filters=trainable_params['model_block/test_recipe_51zs/resnet_16_50yk/conv_37lk/filters'], strides=1, padding='SAME', data_format='NHWC', dilations=1, name='model_block/test_recipe_51zs/resnet_16_50yk/conv_37lk')
    model_block_test_recipe_51zs_resnet_16_50yk_batch_normalize_39na = tf.nn.batch_normalization(x=model_block_test_recipe_51zs_resnet_16_50yk_conv_37lk, mean=trainable_params['model_block/test_recipe_51zs/resnet_16_50yk/batch_normalize_39na/mean'], variance=trainable_params['model_block/test_recipe_51zs/resnet_16_50yk/batch_normalize_39na/variance'], offset=trainable_params['model_block/test_recipe_51zs/resnet_16_50yk/batch_normalize_39na/offset'], scale=trainable_params['model_block/test_recipe_51zs/resnet_16_50yk/batch_normalize_39na/scale'], variance_epsilon=0.001, name='model_block/test_recipe_51zs/resnet_16_50yk/batch_normalize_39na')
    model_block_test_recipe_51zs_resnet_16_50yk_relu_41pq = tf.nn.relu(name='model_block/test_recipe_51zs/resnet_16_50yk/relu_41pq', features=model_block_test_recipe_51zs_resnet_16_50yk_batch_normalize_39na)
    model_block_test_recipe_51zs_resnet_16_50yk_conv_43rg = tf.nn.conv2d(input=model_block_test_recipe_51zs_resnet_16_50yk_relu_41pq, filters=trainable_params['model_block/test_recipe_51zs/resnet_16_50yk/conv_43rg/filters'], strides=1, padding='SAME', data_format='NHWC', dilations=1, name='model_block/test_recipe_51zs/resnet_16_50yk/conv_43rg')
    model_block_test_recipe_51zs_resnet_16_50yk_conv = tf.nn.batch_normalization(x=model_block_test_recipe_51zs_resnet_16_50yk_conv_43rg, mean=trainable_params['model_block/test_recipe_51zs/resnet_16_50yk/conv/mean'], variance=trainable_params['model_block/test_recipe_51zs/resnet_16_50yk/conv/variance'], offset=trainable_params['model_block/test_recipe_51zs/resnet_16_50yk/conv/offset'], scale=trainable_params['model_block/test_recipe_51zs/resnet_16_50yk/conv/scale'], variance_epsilon=0.001, name='model_block/test_recipe_51zs/resnet_16_50yk/conv')
    model_block_test_recipe_51zs_resnet_16_50yk_add_47vm = tf.math.add(x=[model_block_test_recipe_51zs_relu_35ju, model_block_test_recipe_51zs_resnet_16_50yk_conv][0], y=[model_block_test_recipe_51zs_relu_35ju, model_block_test_recipe_51zs_resnet_16_50yk_conv][1], name='model_block/test_recipe_51zs/resnet_16_50yk/add_47vm')
    model_block_test_recipe_51zs_resnet_16_50yk_relu_49xc = tf.nn.relu(name='model_block/test_recipe_51zs/resnet_16_50yk/relu_49xc', features=model_block_test_recipe_51zs_resnet_16_50yk_add_47vm)
    model_block_conv_53bi = tf.nn.conv2d(input=model_block_test_recipe_51zs_resnet_16_50yk_relu_49xc, filters=trainable_params['model_block/conv_53bi/filters'], strides=1, padding='SAME', data_format='NHWC', dilations=1, name='model_block/conv_53bi')
    model_block_relu_55dy = tf.nn.relu(name='model_block/relu_55dy', features=model_block_conv_53bi)
    model_block_dropout_57fo = tf.nn.dropout(x=model_block_relu_55dy, rate=0.2, noise_shape=None, seed=None, name='model_block/dropout_57fo')
    model_block_batch_normalize_59he = tf.nn.batch_normalization(x=model_block_dropout_57fo, mean=trainable_params['model_block/batch_normalize_59he/mean'], variance=trainable_params['model_block/batch_normalize_59he/variance'], offset=trainable_params['model_block/batch_normalize_59he/offset'], scale=trainable_params['model_block/batch_normalize_59he/scale'], variance_epsilon=0.001, name='model_block/batch_normalize_59he')
    model_block_conv_61ju = tf.nn.conv2d(input=model_block_batch_normalize_59he, filters=trainable_params['model_block/conv_61ju/filters'], strides=1, padding='SAME', data_format='NHWC', dilations=1, name='model_block/conv_61ju')
    model_block_max_pool2d_63lk = tf.nn.max_pool(input=model_block_conv_61ju, ksize=3, strides=1, padding='VALID', data_format='NHWC', name='model_block/max_pool2d_63lk')
    model_block_flatten_65na = tf.reshape(tensor=model_block_max_pool2d_63lk, shape=(-1, tf.math.reduce_prod(tf.convert_to_tensor([30, 30, 16]))), name='model_block/flatten_65na')
    model_block_dense_67pq = tf.add(x=tf.matmul(a=model_block_flatten_65na, b=trainable_params['model_block/dense_67pq/weights']), y=trainable_params['model_block/dense_67pq/bias'], name='model_block/dense_67pq')
    model_block_output = tf.nn.softmax(logits=model_block_dense_67pq, name='model_block/output')
    return model_block_output


def get_loss(data_block_labels, model_block_output, trainable_params):
    loss_block_cross_0 = tf.nn.softmax_cross_entropy_with_logits(labels=[data_block_labels, model_block_output][0], logits=[data_block_labels, model_block_output][1], axis=-1, name='loss_block/cross_0')
    loss_block_regularizer = 0.002*sum(list(map(lambda x: tf.nn.l2_loss(t=trainable_params[x], name='loss_block/regularizer'), ['model_block/conv_6gw/filters', 'model_block/resnet_16_21vm_0/conv_8im/filters', 'model_block/resnet_16_21vm_0/conv_14oi/filters', 'model_block/resnet_16_21vm/conv_8im/filters', 'model_block/resnet_16_21vm/conv_14oi/filters', 'model_block/test_recipe_51zs/conv_23xc/filters', 'model_block/test_recipe_51zs/conv_29dy/filters', 'model_block/test_recipe_51zs/resnet_16_50yk/conv_37lk/filters', 'model_block/test_recipe_51zs/resnet_16_50yk/conv_43rg/filters', 'model_block/conv_53bi/filters', 'model_block/conv_61ju/filters', 'model_block/dense_67pq/weights'])))
    loss_block_losses = tf.math.add(x=[loss_block_cross_0, loss_block_regularizer][0], y=[loss_block_cross_0, loss_block_regularizer][1], name='loss_block/losses')
    return loss_block_losses 


def get_optimizer():
    optimizer_block_solver_decay_exponential_decay = tf.keras.optimizers.schedules.ExponentialDecay(initial_learning_rate=0.0001, decay_steps=100000, decay_rate=0.96, staircase=True)
    optimizer_block_solver = tf.optimizers.Adam(learning_rate=optimizer_block_solver_decay_exponential_decay, beta_1=0.9, beta_2=0.999, epsilon=1e-08, name='optimizer_block/solver')
    return optimizer_block_solver 

from alex.alex.checkpoint import Checkpoint

C = Checkpoint("examples/configs/small3.yml",
               'tf',
               None,
               None)

ckpt = C.load()

trainable_params = get_trainable_params()

from alex.alex import registry
var_list = registry.get_trainable_params_list(trainable_params)

optimizer = get_optimizer()

probes = dict()

def inference(data_block_input_data, trainable_params):
    
    preds = model(data_block_input_data, trainable_params)
    classes = tf.math.argmax(preds, 1)
    return preds, classes
    
def evaluation(data_block_input_data, labels, trainable_params):
    
    
    results = inference(data_block_input_data, trainable_params)
    preds = results[0]
    classes = results[1]
    
    
    matches = tf.equal(classes, tf.math.argmax(labels, 1))
    perf = tf.reduce_mean(tf.cast(matches, tf.float32))
    
    loss = tf.reduce_mean(get_loss(labels, preds, trainable_params))
    return perf, loss
    
    
def train(data_block_input_data, data_block_labels, trainable_params, var_list):
    
    with tf.GradientTape() as tape:
        preds = model(data_block_input_data, trainable_params)
        gradients = tape.gradient(get_loss(data_block_labels, preds, trainable_params), var_list)
        optimizer.apply_gradients(zip(gradients, var_list))
    
    
def loop(trainable_params, val_inputs, val_labels, var_list):
    
    for epoch in range(90):
        i = 0
        for batch in trainloader:
            inputs = batch[0]
            labels = batch[1]
            train(inputs, labels, trainable_params, var_list)
            if i % 500 == 499:
                results = evaluation(val_inputs, val_labels, trainable_params)
                
                tf.print("Epoch", epoch, results)
            i += 1
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

valloader = tf.data.Dataset.from_tensor_slices((x_val, y_val)).batch(1000)
for val_inputs, val_labels in valloader:
    break

loop(trainable_params, val_inputs, val_labels, var_list)

