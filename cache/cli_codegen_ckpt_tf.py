import tensorflow as tf
import numpy as np


tf_dtypes = {'float32': tf.float32, 'int8': tf.int8}


def get_trainable_params(ckpt):
    trainable_params = dict()
    model_block_conv_6gw_filters_initializer_xavier_uniform = tf.convert_to_tensor(value=np.asarray(ckpt['model_block/conv_6gw/filters']), dtype=tf_dtypes['float32'], dtype_hint=None)
    model_block_conv_6gw_filters = tf.Variable(initial_value=model_block_conv_6gw_filters_initializer_xavier_uniform, trainable=True, caching_device=None, name='model_block/conv_6gw/filters', variable_def=None, dtype=tf_dtypes['float32'], import_scope=None, constraint=None, synchronization=tf.VariableSynchronization.AUTO, shape=None)
    trainable_params['model_block/conv_6gw/filters'] = model_block_conv_6gw_filters
    model_block_batch_normalize_12ms_mean_initializer_zeros_initializer = tf.convert_to_tensor(value=np.asarray(ckpt['model_block/batch_normalize_12ms/mean']), dtype=tf_dtypes['float32'], dtype_hint=None)
    model_block_batch_normalize_12ms_mean = tf.Variable(initial_value=model_block_batch_normalize_12ms_mean_initializer_zeros_initializer, trainable=False, caching_device=None, name='model_block/batch_normalize_12ms/mean', variable_def=None, dtype=tf_dtypes['float32'], import_scope=None, constraint=None, synchronization=tf.VariableSynchronization.AUTO, shape=None)
    trainable_params['model_block/batch_normalize_12ms/mean'] = model_block_batch_normalize_12ms_mean
    model_block_batch_normalize_12ms_offset_initializer_zeros_initializer = tf.convert_to_tensor(value=np.asarray(ckpt['model_block/batch_normalize_12ms/offset']), dtype=tf_dtypes['float32'], dtype_hint=None)
    model_block_batch_normalize_12ms_offset = tf.Variable(initial_value=model_block_batch_normalize_12ms_offset_initializer_zeros_initializer, trainable=True, caching_device=None, name='model_block/batch_normalize_12ms/offset', variable_def=None, dtype=tf_dtypes['float32'], import_scope=None, constraint=None, synchronization=tf.VariableSynchronization.AUTO, shape=None)
    trainable_params['model_block/batch_normalize_12ms/offset'] = model_block_batch_normalize_12ms_offset
    model_block_batch_normalize_12ms_scale_initializer_ones_initializer = tf.convert_to_tensor(value=np.asarray(ckpt['model_block/batch_normalize_12ms/scale']), dtype=tf_dtypes['float32'], dtype_hint=None)
    model_block_batch_normalize_12ms_scale = tf.Variable(initial_value=model_block_batch_normalize_12ms_scale_initializer_ones_initializer, trainable=True, caching_device=None, name='model_block/batch_normalize_12ms/scale', variable_def=None, dtype=tf_dtypes['float32'], import_scope=None, constraint=None, synchronization=tf.VariableSynchronization.AUTO, shape=None)
    trainable_params['model_block/batch_normalize_12ms/scale'] = model_block_batch_normalize_12ms_scale
    model_block_batch_normalize_12ms_variance_initializer_ones_initializer = tf.convert_to_tensor(value=np.asarray(ckpt['model_block/batch_normalize_12ms/variance']), dtype=tf_dtypes['float32'], dtype_hint=None)
    model_block_batch_normalize_12ms_variance = tf.Variable(initial_value=model_block_batch_normalize_12ms_variance_initializer_ones_initializer, trainable=False, caching_device=None, name='model_block/batch_normalize_12ms/variance', variable_def=None, dtype=tf_dtypes['float32'], import_scope=None, constraint=None, synchronization=tf.VariableSynchronization.AUTO, shape=None)
    trainable_params['model_block/batch_normalize_12ms/variance'] = model_block_batch_normalize_12ms_variance
    model_block_conv_14oi_filters_initializer_xavier_uniform = tf.convert_to_tensor(value=np.asarray(ckpt['model_block/conv_14oi/filters']), dtype=tf_dtypes['float32'], dtype_hint=None)
    model_block_conv_14oi_filters = tf.Variable(initial_value=model_block_conv_14oi_filters_initializer_xavier_uniform, trainable=True, caching_device=None, name='model_block/conv_14oi/filters', variable_def=None, dtype=tf_dtypes['float32'], import_scope=None, constraint=None, synchronization=tf.VariableSynchronization.AUTO, shape=None)
    trainable_params['model_block/conv_14oi/filters'] = model_block_conv_14oi_filters
    model_block_conv_16qy_filters_initializer_xavier_uniform = tf.convert_to_tensor(value=np.asarray(ckpt['model_block/conv_16qy/filters']), dtype=tf_dtypes['float32'], dtype_hint=None)
    model_block_conv_16qy_filters = tf.Variable(initial_value=model_block_conv_16qy_filters_initializer_xavier_uniform, trainable=True, caching_device=None, name='model_block/conv_16qy/filters', variable_def=None, dtype=tf_dtypes['float32'], import_scope=None, constraint=None, synchronization=tf.VariableSynchronization.AUTO, shape=None)
    trainable_params['model_block/conv_16qy/filters'] = model_block_conv_16qy_filters
    model_block_dense_20ue_bias_initializer_zeros_initializer = tf.convert_to_tensor(value=np.asarray(ckpt['model_block/dense_20ue/bias']), dtype=tf_dtypes['float32'], dtype_hint=None)
    model_block_dense_20ue_bias = tf.Variable(initial_value=model_block_dense_20ue_bias_initializer_zeros_initializer, trainable=True, caching_device=None, name='model_block/dense_20ue/bias', variable_def=None, dtype=tf_dtypes['float32'], import_scope=None, constraint=None, synchronization=tf.VariableSynchronization.AUTO, shape=None)
    trainable_params['model_block/dense_20ue/bias'] = model_block_dense_20ue_bias
    model_block_dense_20ue_weights_initializer_xavier_uniform = tf.convert_to_tensor(value=np.asarray(ckpt['model_block/dense_20ue/weights']), dtype=tf_dtypes['float32'], dtype_hint=None)
    model_block_dense_20ue_weights = tf.Variable(initial_value=model_block_dense_20ue_weights_initializer_xavier_uniform, trainable=True, caching_device=None, name='model_block/dense_20ue/weights', variable_def=None, dtype=tf_dtypes['float32'], import_scope=None, constraint=None, synchronization=tf.VariableSynchronization.AUTO, shape=None)
    trainable_params['model_block/dense_20ue/weights'] = model_block_dense_20ue_weights
    return trainable_params


def model(probes, trainable_params, data_block_input_data):
    model_block_conv_6gw = tf.nn.conv2d(input=data_block_input_data, filters=trainable_params['model_block/conv_6gw/filters'], strides=1, padding='SAME', data_format='NHWC', dilations=1, name='model_block/conv_6gw')
    model_block_reluu = tf.nn.relu(name='model_block/reluu', features=model_block_conv_6gw)
    model_block_dropout_10kc = tf.nn.dropout(x=model_block_reluu, rate=0.2, noise_shape=None, seed=None, name='model_block/dropout_10kc')
    model_block_batch_normalize_12ms = tf.nn.batch_normalization(x=model_block_dropout_10kc, mean=trainable_params['model_block/batch_normalize_12ms/mean'], variance=trainable_params['model_block/batch_normalize_12ms/variance'], offset=trainable_params['model_block/batch_normalize_12ms/offset'], scale=trainable_params['model_block/batch_normalize_12ms/scale'], variance_epsilon=0.001, name='model_block/batch_normalize_12ms')
    model_block_conv_14oi = tf.nn.conv2d(input=model_block_batch_normalize_12ms, filters=trainable_params['model_block/conv_14oi/filters'], strides=1, padding='SAME', data_format='NHWC', dilations=1, name='model_block/conv_14oi')
    model_block_conv_16qy = tf.nn.conv2d(input=model_block_conv_14oi, filters=trainable_params['model_block/conv_16qy/filters'], strides=1, padding='SAME', data_format='NHWC', dilations=1, name='model_block/conv_16qy')
    model_block_flatten_18so = tf.reshape(tensor=model_block_conv_16qy, shape=(-1, tf.math.reduce_prod(tf.convert_to_tensor([32, 32, 64]))), name='model_block/flatten_18so')
    probes['model_block/flatten_18so'] = model_block_flatten_18so
    model_block_dense_20ue = tf.add(x=tf.matmul(a=model_block_flatten_18so, b=trainable_params['model_block/dense_20ue/weights']), y=trainable_params['model_block/dense_20ue/bias'], name='model_block/dense_20ue')
    model_block_output = tf.nn.softmax(logits=model_block_dense_20ue, name='model_block/output')
    return model_block_output


def get_loss(data_block_labels, model_block_output, trainable_params):
    loss_block_cross_0 = tf.nn.softmax_cross_entropy_with_logits(labels=[data_block_labels, model_block_output][0], logits=[data_block_labels, model_block_output][1], axis=-1, name='loss_block/cross_0')
    loss_block_regularizer = 0.002*sum(list(map(lambda x: tf.nn.l2_loss(t=trainable_params[x], name='loss_block/regularizer'), ['model_block/conv_6gw/filters', 'model_block/conv_14oi/filters', 'model_block/conv_16qy/filters', 'model_block/dense_20ue/weights'])))
    loss_block_losses = tf.math.add(x=[loss_block_cross_0, loss_block_regularizer][0], y=[loss_block_cross_0, loss_block_regularizer][1], name='loss_block/losses')
    return loss_block_losses 


def get_optimizer():
    optimizer_block_solver_decay_exponential_decay = tf.keras.optimizers.schedules.ExponentialDecay(initial_learning_rate=0.0001, decay_steps=100000, decay_rate=0.96, staircase=True)
    optimizer_block_solver = tf.optimizers.Adam(learning_rate=optimizer_block_solver_decay_exponential_decay, beta_1=0.9, beta_2=0.999, epsilon=1e-08, name='optimizer_block/solver')
    return optimizer_block_solver 

from alex.alex.checkpoint import Checkpoint

C = Checkpoint("examples/configs/small1.yml",
               'tf',
               ['checkpoints', 'test.json'],
               None)

ckpt = C.load()

trainable_params = get_trainable_params(ckpt)

from alex.alex import registry
var_list = registry.get_trainable_params_list(trainable_params)

optimizer = get_optimizer()

probes = dict()

def inference(probes, trainable_params, data_block_input_data):
    
    preds = tf.math.argmax(model(probes, trainable_params, data_block_input_data), 1)
    return preds
    
def evaluation(probes, trainable_params, data_block_input_data, labels):
    
    preds = inference(probes, trainable_params, data_block_input_data)
    
    matches = tf.equal(preds, tf.math.argmax(labels, 1))
    perf = tf.reduce_mean(tf.cast(matches, tf.float32))
    
    loss = tf.reduce_mean(get_loss(labels, preds, trainable_params))
    return perf, loss
    
    
def train(probes, data_block_labels, var_list, trainable_params, data_block_input_data):
    
    with tf.GradientTape() as tape:
        preds = model(probes, trainable_params, data_block_input_data)
        gradients = tape.gradient(get_loss(data_block_labels, preds, trainable_params), var_list)
        optimizer.apply_gradients(zip(gradients, var_list))
    
    
def loop(probes, var_list, trainable_params, val_inputs):
    
    for epoch in range(90):
        i = 0
        for batch in trainloader:
            inputs = batch[0]
            labels = batch[1]
            train(probes, labels, var_list, trainable_params, inputs)
            if i % 500 == 499:
                results = evaluation(probes, trainable_params, val_inputs, labels)
                
                tf.print("Epoch", epoch, results)
            i += 1
    print('Finished Training')
    
    
    