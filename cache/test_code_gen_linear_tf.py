import tensorflow as tf
import numpy as np


tf_dtypes = {'float32': tf.float32, 'int8': tf.int8}


def get_trainable_params(ckpt):
    trainable_params = dict()
    model_block_conv_6gw_filters_initializer_xavier_uniform = tf.keras.initializers.glorot_uniform(seed=1)(shape=(3, 3, 3, 64))
    model_block_conv_6gw_filters = tf.Variable(initial_value=model_block_conv_6gw_filters_initializer_xavier_uniform, trainable=True, caching_device=None, name='model_block/conv_6gw/filters', variable_def=None, dtype=tf_dtypes['float32'], import_scope=None, constraint=None, synchronization=tf.VariableSynchronization.AUTO, shape=None)
    trainable_params['model_block/conv_6gw/filters'] = model_block_conv_6gw_filters
    model_block_batch_normalize_10kc_mean_initializer_zeros_initializer = tf.zeros_initializer()(shape=[64, ])
    model_block_batch_normalize_10kc_mean = tf.Variable(initial_value=model_block_batch_normalize_10kc_mean_initializer_zeros_initializer, trainable=False, caching_device=None, name='model_block/batch_normalize_10kc/mean', variable_def=None, dtype=tf_dtypes['float32'], import_scope=None, constraint=None, synchronization=tf.VariableSynchronization.AUTO, shape=None)
    trainable_params['model_block/batch_normalize_10kc/mean'] = model_block_batch_normalize_10kc_mean
    model_block_batch_normalize_10kc_offset_initializer_zeros_initializer = tf.zeros_initializer()(shape=[64, ])
    model_block_batch_normalize_10kc_offset = tf.Variable(initial_value=model_block_batch_normalize_10kc_offset_initializer_zeros_initializer, trainable=True, caching_device=None, name='model_block/batch_normalize_10kc/offset', variable_def=None, dtype=tf_dtypes['float32'], import_scope=None, constraint=None, synchronization=tf.VariableSynchronization.AUTO, shape=None)
    trainable_params['model_block/batch_normalize_10kc/offset'] = model_block_batch_normalize_10kc_offset
    model_block_batch_normalize_10kc_scale_initializer_ones_initializer = tf.ones_initializer()(shape=[64, ])
    model_block_batch_normalize_10kc_scale = tf.Variable(initial_value=model_block_batch_normalize_10kc_scale_initializer_ones_initializer, trainable=True, caching_device=None, name='model_block/batch_normalize_10kc/scale', variable_def=None, dtype=tf_dtypes['float32'], import_scope=None, constraint=None, synchronization=tf.VariableSynchronization.AUTO, shape=None)
    trainable_params['model_block/batch_normalize_10kc/scale'] = model_block_batch_normalize_10kc_scale
    model_block_batch_normalize_10kc_variance_initializer_ones_initializer = tf.ones_initializer()(shape=[64, ])
    model_block_batch_normalize_10kc_variance = tf.Variable(initial_value=model_block_batch_normalize_10kc_variance_initializer_ones_initializer, trainable=False, caching_device=None, name='model_block/batch_normalize_10kc/variance', variable_def=None, dtype=tf_dtypes['float32'], import_scope=None, constraint=None, synchronization=tf.VariableSynchronization.AUTO, shape=None)
    trainable_params['model_block/batch_normalize_10kc/variance'] = model_block_batch_normalize_10kc_variance
    model_block_dense_18so_bias_initializer_zeros_initializer = tf.convert_to_tensor(value=np.asarray(ckpt['model_block/dense_18so/bias']), dtype=tf_dtypes['float32'], dtype_hint=None)
    model_block_dense_18so_bias = tf.Variable(initial_value=model_block_dense_18so_bias_initializer_zeros_initializer, trainable=True, caching_device=None, name='model_block/dense_18so/bias', variable_def=None, dtype=tf_dtypes['float32'], import_scope=None, constraint=None, synchronization=tf.VariableSynchronization.AUTO, shape=None)
    trainable_params['model_block/dense_18so/bias'] = model_block_dense_18so_bias
    model_block_dense_18so_weights_initializer_xavier_uniform = tf.convert_to_tensor(value=np.asarray(ckpt['model_block/dense_18so/weights']), dtype=tf_dtypes['float32'], dtype_hint=None)
    model_block_dense_18so_weights = tf.Variable(initial_value=model_block_dense_18so_weights_initializer_xavier_uniform, trainable=True, caching_device=None, name='model_block/dense_18so/weights', variable_def=None, dtype=tf_dtypes['float32'], import_scope=None, constraint=None, synchronization=tf.VariableSynchronization.AUTO, shape=None)
    trainable_params['model_block/dense_18so/weights'] = model_block_dense_18so_weights
    loss_block_conv_23xc_filters_initializer_xavier_uniform = tf.keras.initializers.glorot_uniform(seed=1)(shape=(3, 3, 3, 16))
    loss_block_conv_23xc_filters = tf.Variable(initial_value=loss_block_conv_23xc_filters_initializer_xavier_uniform, trainable=False, caching_device=None, name='loss_block/conv_23xc/filters', variable_def=None, dtype=tf_dtypes['float32'], import_scope=None, constraint=None, synchronization=tf.VariableSynchronization.AUTO, shape=None)
    trainable_params['loss_block/conv_23xc/filters'] = loss_block_conv_23xc_filters
    loss_block_batch_normalize_27bi_mean_initializer_zeros_initializer = tf.zeros_initializer()(shape=[16, ])
    loss_block_batch_normalize_27bi_mean = tf.Variable(initial_value=loss_block_batch_normalize_27bi_mean_initializer_zeros_initializer, trainable=False, caching_device=None, name='loss_block/batch_normalize_27bi/mean', variable_def=None, dtype=tf_dtypes['float32'], import_scope=None, constraint=None, synchronization=tf.VariableSynchronization.AUTO, shape=None)
    trainable_params['loss_block/batch_normalize_27bi/mean'] = loss_block_batch_normalize_27bi_mean
    loss_block_batch_normalize_27bi_offset_initializer_zeros_initializer = tf.zeros_initializer()(shape=[16, ])
    loss_block_batch_normalize_27bi_offset = tf.Variable(initial_value=loss_block_batch_normalize_27bi_offset_initializer_zeros_initializer, trainable=False, caching_device=None, name='loss_block/batch_normalize_27bi/offset', variable_def=None, dtype=tf_dtypes['float32'], import_scope=None, constraint=None, synchronization=tf.VariableSynchronization.AUTO, shape=None)
    trainable_params['loss_block/batch_normalize_27bi/offset'] = loss_block_batch_normalize_27bi_offset
    loss_block_batch_normalize_27bi_scale_initializer_ones_initializer = tf.ones_initializer()(shape=[16, ])
    loss_block_batch_normalize_27bi_scale = tf.Variable(initial_value=loss_block_batch_normalize_27bi_scale_initializer_ones_initializer, trainable=False, caching_device=None, name='loss_block/batch_normalize_27bi/scale', variable_def=None, dtype=tf_dtypes['float32'], import_scope=None, constraint=None, synchronization=tf.VariableSynchronization.AUTO, shape=None)
    trainable_params['loss_block/batch_normalize_27bi/scale'] = loss_block_batch_normalize_27bi_scale
    loss_block_batch_normalize_27bi_variance_initializer_ones_initializer = tf.ones_initializer()(shape=[16, ])
    loss_block_batch_normalize_27bi_variance = tf.Variable(initial_value=loss_block_batch_normalize_27bi_variance_initializer_ones_initializer, trainable=False, caching_device=None, name='loss_block/batch_normalize_27bi/variance', variable_def=None, dtype=tf_dtypes['float32'], import_scope=None, constraint=None, synchronization=tf.VariableSynchronization.AUTO, shape=None)
    trainable_params['loss_block/batch_normalize_27bi/variance'] = loss_block_batch_normalize_27bi_variance
    loss_block_conv_29dy_filters_initializer_xavier_uniform = tf.keras.initializers.glorot_uniform(seed=1)(shape=(3, 3, 16, 64))
    loss_block_conv_29dy_filters = tf.Variable(initial_value=loss_block_conv_29dy_filters_initializer_xavier_uniform, trainable=False, caching_device=None, name='loss_block/conv_29dy/filters', variable_def=None, dtype=tf_dtypes['float32'], import_scope=None, constraint=None, synchronization=tf.VariableSynchronization.AUTO, shape=None)
    trainable_params['loss_block/conv_29dy/filters'] = loss_block_conv_29dy_filters
    loss_block_conv_31fo_filters_initializer_xavier_uniform = tf.keras.initializers.glorot_uniform(seed=1)(shape=(3, 3, 64, 64))
    loss_block_conv_31fo_filters = tf.Variable(initial_value=loss_block_conv_31fo_filters_initializer_xavier_uniform, trainable=False, caching_device=None, name='loss_block/conv_31fo/filters', variable_def=None, dtype=tf_dtypes['float32'], import_scope=None, constraint=None, synchronization=tf.VariableSynchronization.AUTO, shape=None)
    trainable_params['loss_block/conv_31fo/filters'] = loss_block_conv_31fo_filters
    return trainable_params


def model(data_block_input_data, probes, trainable_params):
    model_block_conv_6gw = tf.nn.conv2d(input=data_block_input_data, filters=trainable_params['model_block/conv_6gw/filters'], strides=1, padding='SAME', data_format='NHWC', dilations=1, name='model_block/conv_6gw')
    model_block_relu_8im = tf.nn.relu(name='model_block/relu_8im', features=model_block_conv_6gw)
    model_block_batch_normalize_10kc = tf.nn.batch_normalization(x=model_block_relu_8im, mean=trainable_params['model_block/batch_normalize_10kc/mean'], variance=trainable_params['model_block/batch_normalize_10kc/variance'], offset=trainable_params['model_block/batch_normalize_10kc/offset'], scale=trainable_params['model_block/batch_normalize_10kc/scale'], variance_epsilon=0.001, name='model_block/batch_normalize_10kc')
    model_block_max_pool2d_12ms = tf.nn.max_pool(input=model_block_batch_normalize_10kc, ksize=3, strides=1, padding='VALID', data_format='NHWC', name='model_block/max_pool2d_12ms')
    model_block_max_pool2d_14oi = tf.nn.max_pool(input=model_block_max_pool2d_12ms, ksize=3, strides=1, padding='VALID', data_format='NHWC', name='model_block/max_pool2d_14oi')
    model_block_output = tf.reshape(tensor=model_block_max_pool2d_14oi, shape=(-1, tf.math.reduce_prod(tf.convert_to_tensor([28, 28, 64]))), name='model_block/output')
    model_block_dense_18so = tf.add(x=tf.matmul(a=model_block_output, b=trainable_params['model_block/dense_18so/weights']), y=trainable_params['model_block/dense_18so/bias'], name='model_block/dense_18so')
    model_block_probes = tf.nn.softmax(logits=model_block_dense_18so, name='model_block/probes')
    probes['model_block/probes'] = model_block_probes
    return model_block_output


def get_loss(data_block_input_data, model_block_output, trainable_params):
    loss_block_conv_23xc = tf.nn.conv2d(input=data_block_input_data, filters=trainable_params['loss_block/conv_23xc/filters'], strides=1, padding='SAME', data_format='NHWC', dilations=1, name='loss_block/conv_23xc')
    loss_block_reluu = tf.nn.relu(name='loss_block/reluu', features=loss_block_conv_23xc)
    loss_block_batch_normalize_27bi = tf.nn.batch_normalization(x=loss_block_reluu, mean=trainable_params['loss_block/batch_normalize_27bi/mean'], variance=trainable_params['loss_block/batch_normalize_27bi/variance'], offset=trainable_params['loss_block/batch_normalize_27bi/offset'], scale=trainable_params['loss_block/batch_normalize_27bi/scale'], variance_epsilon=0.001, name='loss_block/batch_normalize_27bi')
    loss_block_conv_29dy = tf.nn.conv2d(input=loss_block_batch_normalize_27bi, filters=trainable_params['loss_block/conv_29dy/filters'], strides=1, padding='VALID', data_format='NHWC', dilations=1, name='loss_block/conv_29dy')
    loss_block_conv_31fo = tf.nn.conv2d(input=loss_block_conv_29dy, filters=trainable_params['loss_block/conv_31fo/filters'], strides=1, padding='VALID', data_format='NHWC', dilations=1, name='loss_block/conv_31fo')
    loss_block_feature = tf.reshape(tensor=loss_block_conv_31fo, shape=(-1, tf.math.reduce_prod(tf.convert_to_tensor([28, 28, 64]))), name='loss_block/feature')
    loss_block_cross_0 = tf.keras.metrics.mean_squared_error(y_true=[loss_block_feature, model_block_output][0], y_pred=[loss_block_feature, model_block_output][1])
    loss_block_regularizer = 0.002*sum(list(map(lambda x: tf.nn.l2_loss(t=trainable_params[x], name='loss_block/regularizer'), ['model_block/conv_6gw/filters', 'model_block/dense_18so/weights', 'loss_block/conv_23xc/filters', 'loss_block/conv_29dy/filters', 'loss_block/conv_31fo/filters'])))
    loss_block_losses = tf.math.add(x=[loss_block_cross_0, loss_block_regularizer][0], y=[loss_block_cross_0, loss_block_regularizer][1], name='loss_block/losses')
    return loss_block_losses 


def get_optimizer():
    optimizer_block_solver_decay_exponential_decay = tf.keras.optimizers.schedules.ExponentialDecay(initial_learning_rate=0.0001, decay_steps=100000, decay_rate=0.96, staircase=True)
    optimizer_block_solver = tf.optimizers.Adam(learning_rate=optimizer_block_solver_decay_exponential_decay, beta_1=0.9, beta_2=0.999, epsilon=1e-08, name='optimizer_block/solver')
    return optimizer_block_solver 

from alex.alex.checkpoint import Checkpoint

C = Checkpoint("examples/configs/small1_linear.yml",
               'tf',
               ['checkpoints', 'test_code_gen_ckpt_trained.json'],
               None)

ckpt = C.load()

trainable_params = get_trainable_params(ckpt)

from alex.alex import registry
var_list = registry.get_trainable_params_list(trainable_params)

optimizer = get_optimizer()

probes = dict()

def inference(data_block_input_data, probes, trainable_params):
    
    preds = model(data_block_input_data, probes, trainable_params)
    
    return preds
    
def evaluation(data_block_input_data, probes, trainable_params):
    
    
    preds = inference(data_block_input_data, probes, trainable_params)
    
    
    loss = tf.reduce_mean(get_loss(data_block_input_data, preds, trainable_params))
    return loss
    
    
def train(data_block_input_data, probes, trainable_params, var_list):
    
    with tf.GradientTape() as tape:
        preds = model(data_block_input_data, probes, trainable_params)
        gradients = tape.gradient(get_loss(data_block_input_data, preds, trainable_params), var_list)
        optimizer.apply_gradients(zip(gradients, var_list))
    
    
def loop(probes, trainable_params, val_inputs, var_list):
    
    for epoch in range(90):
        i = 0
        for batch in trainloader:
            inputs = batch[0]
            labels = batch[1]
            train(inputs, probes, trainable_params, var_list)
            if i % 500 == 499:
                results = evaluation(val_inputs, probes, trainable_params)
                
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

loop(probes, trainable_params, val_inputs, var_list)

