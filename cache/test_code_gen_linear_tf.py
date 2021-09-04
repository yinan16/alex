import tensorflow as tf
import numpy as np


tf_dtypes = {'float32': tf.float32, 'int8': tf.int8}


def get_trainable_params(ckpt):
    trainable_params = dict()
    model_block_conv_4eg_filters_initializer_xavier_uniform = tf.keras.initializers.glorot_uniform(seed=1)(shape=(3, 3, 3, 64))
    model_block_conv_4eg_filters = tf.Variable(initial_value=model_block_conv_4eg_filters_initializer_xavier_uniform, trainable=True, caching_device=None, name='model_block/conv_4eg/filters', variable_def=None, dtype=tf_dtypes['float32'], import_scope=None, constraint=None, synchronization=tf.VariableSynchronization.AUTO, shape=None)
    trainable_params['model_block/conv_4eg/filters'] = model_block_conv_4eg_filters
    model_block_dense_12ms_bias_initializer_zeros_initializer = tf.convert_to_tensor(value=np.asarray(ckpt['model_block/dense_12ms/bias']), dtype=tf_dtypes['float32'], dtype_hint=None)
    model_block_dense_12ms_bias = tf.Variable(initial_value=model_block_dense_12ms_bias_initializer_zeros_initializer, trainable=True, caching_device=None, name='model_block/dense_12ms/bias', variable_def=None, dtype=tf_dtypes['float32'], import_scope=None, constraint=None, synchronization=tf.VariableSynchronization.AUTO, shape=None)
    trainable_params['model_block/dense_12ms/bias'] = model_block_dense_12ms_bias
    model_block_dense_12ms_weights_initializer_xavier_uniform = tf.convert_to_tensor(value=np.asarray(ckpt['model_block/dense_12ms/weights']), dtype=tf_dtypes['float32'], dtype_hint=None)
    model_block_dense_12ms_weights = tf.Variable(initial_value=model_block_dense_12ms_weights_initializer_xavier_uniform, trainable=True, caching_device=None, name='model_block/dense_12ms/weights', variable_def=None, dtype=tf_dtypes['float32'], import_scope=None, constraint=None, synchronization=tf.VariableSynchronization.AUTO, shape=None)
    trainable_params['model_block/dense_12ms/weights'] = model_block_dense_12ms_weights
    loss_block_conv_17rg_filters_initializer_xavier_uniform = tf.keras.initializers.glorot_uniform(seed=1)(shape=(3, 3, 3, 16))
    loss_block_conv_17rg_filters = tf.Variable(initial_value=loss_block_conv_17rg_filters_initializer_xavier_uniform, trainable=False, caching_device=None, name='loss_block/conv_17rg/filters', variable_def=None, dtype=tf_dtypes['float32'], import_scope=None, constraint=None, synchronization=tf.VariableSynchronization.AUTO, shape=None)
    trainable_params['loss_block/conv_17rg/filters'] = loss_block_conv_17rg_filters
    loss_block_batch_normalize_21vm_mean_initializer_zeros_initializer = tf.zeros_initializer()(shape=[16, ])
    loss_block_batch_normalize_21vm_mean = tf.Variable(initial_value=loss_block_batch_normalize_21vm_mean_initializer_zeros_initializer, trainable=False, caching_device=None, name='loss_block/batch_normalize_21vm/mean', variable_def=None, dtype=tf_dtypes['float32'], import_scope=None, constraint=None, synchronization=tf.VariableSynchronization.AUTO, shape=None)
    trainable_params['loss_block/batch_normalize_21vm/mean'] = loss_block_batch_normalize_21vm_mean
    loss_block_batch_normalize_21vm_offset_initializer_zeros_initializer = tf.zeros_initializer()(shape=[16, ])
    loss_block_batch_normalize_21vm_offset = tf.Variable(initial_value=loss_block_batch_normalize_21vm_offset_initializer_zeros_initializer, trainable=False, caching_device=None, name='loss_block/batch_normalize_21vm/offset', variable_def=None, dtype=tf_dtypes['float32'], import_scope=None, constraint=None, synchronization=tf.VariableSynchronization.AUTO, shape=None)
    trainable_params['loss_block/batch_normalize_21vm/offset'] = loss_block_batch_normalize_21vm_offset
    loss_block_batch_normalize_21vm_scale_initializer_ones_initializer = tf.ones_initializer()(shape=[16, ])
    loss_block_batch_normalize_21vm_scale = tf.Variable(initial_value=loss_block_batch_normalize_21vm_scale_initializer_ones_initializer, trainable=False, caching_device=None, name='loss_block/batch_normalize_21vm/scale', variable_def=None, dtype=tf_dtypes['float32'], import_scope=None, constraint=None, synchronization=tf.VariableSynchronization.AUTO, shape=None)
    trainable_params['loss_block/batch_normalize_21vm/scale'] = loss_block_batch_normalize_21vm_scale
    loss_block_batch_normalize_21vm_variance_initializer_ones_initializer = tf.ones_initializer()(shape=[16, ])
    loss_block_batch_normalize_21vm_variance = tf.Variable(initial_value=loss_block_batch_normalize_21vm_variance_initializer_ones_initializer, trainable=False, caching_device=None, name='loss_block/batch_normalize_21vm/variance', variable_def=None, dtype=tf_dtypes['float32'], import_scope=None, constraint=None, synchronization=tf.VariableSynchronization.AUTO, shape=None)
    trainable_params['loss_block/batch_normalize_21vm/variance'] = loss_block_batch_normalize_21vm_variance
    loss_block_conv_23xc_filters_initializer_xavier_uniform = tf.keras.initializers.glorot_uniform(seed=1)(shape=(3, 3, 16, 64))
    loss_block_conv_23xc_filters = tf.Variable(initial_value=loss_block_conv_23xc_filters_initializer_xavier_uniform, trainable=False, caching_device=None, name='loss_block/conv_23xc/filters', variable_def=None, dtype=tf_dtypes['float32'], import_scope=None, constraint=None, synchronization=tf.VariableSynchronization.AUTO, shape=None)
    trainable_params['loss_block/conv_23xc/filters'] = loss_block_conv_23xc_filters
    loss_block_conv_25zs_filters_initializer_xavier_uniform = tf.keras.initializers.glorot_uniform(seed=1)(shape=(3, 3, 64, 64))
    loss_block_conv_25zs_filters = tf.Variable(initial_value=loss_block_conv_25zs_filters_initializer_xavier_uniform, trainable=False, caching_device=None, name='loss_block/conv_25zs/filters', variable_def=None, dtype=tf_dtypes['float32'], import_scope=None, constraint=None, synchronization=tf.VariableSynchronization.AUTO, shape=None)
    trainable_params['loss_block/conv_25zs/filters'] = loss_block_conv_25zs_filters
    return trainable_params


def model(data_block_input_data, trainable_params, probes):
    model_block_conv_4eg = tf.nn.conv2d(input=data_block_input_data, filters=trainable_params['model_block/conv_4eg/filters'], strides=1, padding='SAME', data_format='NHWC', dilations=1, name='model_block/conv_4eg')
    model_block_max_pool2d_6gw = tf.nn.max_pool(input=model_block_conv_4eg, ksize=3, strides=1, padding='VALID', data_format='NHWC', name='model_block/max_pool2d_6gw')
    model_block_max_pool2d_8im = tf.nn.max_pool(input=model_block_max_pool2d_6gw, ksize=3, strides=1, padding='VALID', data_format='NHWC', name='model_block/max_pool2d_8im')
    model_block_output = tf.reshape(tensor=model_block_max_pool2d_8im, shape=(-1, tf.math.reduce_prod(tf.convert_to_tensor([28, 28, 64]))), name='model_block/output')
    model_block_dense_12ms = tf.add(x=tf.matmul(a=model_block_output, b=trainable_params['model_block/dense_12ms/weights']), y=trainable_params['model_block/dense_12ms/bias'], name='model_block/dense_12ms')
    model_block_probes = tf.nn.softmax(logits=model_block_dense_12ms, name='model_block/probes')
    probes['model_block/probes'] = model_block_probes
    return model_block_output


def get_loss(data_block_input_data, trainable_params, model_block_output):
    loss_block_conv_17rg = tf.nn.conv2d(input=data_block_input_data, filters=trainable_params['loss_block/conv_17rg/filters'], strides=1, padding='SAME', data_format='NHWC', dilations=1, name='loss_block/conv_17rg')
    loss_block_reluu = tf.nn.relu(name='loss_block/reluu', features=loss_block_conv_17rg)
    loss_block_batch_normalize_21vm = tf.nn.batch_normalization(x=loss_block_reluu, mean=trainable_params['loss_block/batch_normalize_21vm/mean'], variance=trainable_params['loss_block/batch_normalize_21vm/variance'], offset=trainable_params['loss_block/batch_normalize_21vm/offset'], scale=trainable_params['loss_block/batch_normalize_21vm/scale'], variance_epsilon=0.001, name='loss_block/batch_normalize_21vm')
    loss_block_conv_23xc = tf.nn.conv2d(input=loss_block_batch_normalize_21vm, filters=trainable_params['loss_block/conv_23xc/filters'], strides=1, padding='VALID', data_format='NHWC', dilations=1, name='loss_block/conv_23xc')
    loss_block_conv_25zs = tf.nn.conv2d(input=loss_block_conv_23xc, filters=trainable_params['loss_block/conv_25zs/filters'], strides=1, padding='VALID', data_format='NHWC', dilations=1, name='loss_block/conv_25zs')
    loss_block_feature = tf.reshape(tensor=loss_block_conv_25zs, shape=(-1, tf.math.reduce_prod(tf.convert_to_tensor([28, 28, 64]))), name='loss_block/feature')
    loss_block_cross_0 = tf.keras.metrics.mean_squared_error(y_true=[loss_block_feature, model_block_output][0], y_pred=[loss_block_feature, model_block_output][1])
    loss_block_regularizer = 0.002*sum(list(map(lambda x: tf.nn.l2_loss(t=trainable_params[x], name='loss_block/regularizer'), ['model_block/conv_4eg/filters', 'model_block/dense_12ms/weights', 'loss_block/conv_17rg/filters', 'loss_block/conv_23xc/filters', 'loss_block/conv_25zs/filters'])))
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

def inference(data_block_input_data, trainable_params, probes):
    
    preds = model(data_block_input_data, trainable_params, probes)
    
    return preds
    
def evaluation(data_block_input_data, trainable_params, probes):
    
    preds = inference(data_block_input_data, trainable_params, probes)
    
    loss = tf.reduce_mean(get_loss(data_block_input_data, trainable_params, preds))
    return loss
    
    
def train(data_block_input_data, trainable_params, var_list, probes):
    
    with tf.GradientTape() as tape:
        preds = model(data_block_input_data, trainable_params, probes)
        gradients = tape.gradient(get_loss(data_block_input_data, trainable_params, preds), var_list)
        optimizer.apply_gradients(zip(gradients, var_list))
    
    
def loop(probes, val_inputs, trainable_params, var_list):
    
    for epoch in range(90):
        i = 0
        for batch in trainloader:
            inputs = batch[0]
            labels = batch[1]
            train(inputs, trainable_params, var_list, probes)
            if i % 500 == 499:
                results = evaluation(val_inputs, trainable_params, probes)
                
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

loop(probes, val_inputs, trainable_params, var_list)

