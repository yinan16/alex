import tensorflow as tf
import numpy as np


tf_dtypes = {'float32': tf.float32, 'int8': tf.int8}


def get_trainable_params(ckpt):
    trainable_params = dict()
    model_block_conv_4eg_filters_initializer_xavier_uniform = tf.keras.initializers.glorot_uniform(seed=1)(shape=(3, 3, 3, 4))
    model_block_conv_4eg_filters = tf.Variable(initial_value=model_block_conv_4eg_filters_initializer_xavier_uniform, trainable=True, caching_device=None, name='model_block/conv_4eg/filters', variable_def=None, dtype=tf_dtypes['float32'], import_scope=None, constraint=None, synchronization=tf.VariableSynchronization.AUTO, shape=None)
    trainable_params['model_block/conv_4eg/filters'] = model_block_conv_4eg_filters
    loss_block_conv_13na_filters_initializer_xavier_uniform = tf.convert_to_tensor(value=np.asarray(ckpt['loss_block/conv_13na/filters']), dtype=tf_dtypes['dtype'], dtype_hint=None)
    loss_block_conv_13na_filters = tf.Variable(initial_value=loss_block_conv_13na_filters_initializer_xavier_uniform, trainable=False, caching_device=None, name='loss_block/conv_13na/filters', variable_def=None, dtype=tf_dtypes['float32'], import_scope=None, constraint=None, synchronization=tf.VariableSynchronization.AUTO, shape=None)
    trainable_params['loss_block/conv_13na/filters'] = loss_block_conv_13na_filters
    loss_block_batch_normalize_19tw_mean_initializer_zeros_initializer = tf.convert_to_tensor(value=np.asarray(ckpt['loss_block/batch_normalize_19tw/mean']), dtype=tf_dtypes['dtype'], dtype_hint=None)
    loss_block_batch_normalize_19tw_mean = tf.Variable(initial_value=loss_block_batch_normalize_19tw_mean_initializer_zeros_initializer, trainable=False, caching_device=None, name='loss_block/batch_normalize_19tw/mean', variable_def=None, dtype=tf_dtypes['float32'], import_scope=None, constraint=None, synchronization=tf.VariableSynchronization.AUTO, shape=None)
    trainable_params['loss_block/batch_normalize_19tw/mean'] = loss_block_batch_normalize_19tw_mean
    loss_block_batch_normalize_19tw_offset_initializer_zeros_initializer = tf.convert_to_tensor(value=np.asarray(ckpt['loss_block/batch_normalize_19tw/offset']), dtype=tf_dtypes['dtype'], dtype_hint=None)
    loss_block_batch_normalize_19tw_offset = tf.Variable(initial_value=loss_block_batch_normalize_19tw_offset_initializer_zeros_initializer, trainable=False, caching_device=None, name='loss_block/batch_normalize_19tw/offset', variable_def=None, dtype=tf_dtypes['float32'], import_scope=None, constraint=None, synchronization=tf.VariableSynchronization.AUTO, shape=None)
    trainable_params['loss_block/batch_normalize_19tw/offset'] = loss_block_batch_normalize_19tw_offset
    loss_block_batch_normalize_19tw_scale_initializer_ones_initializer = tf.convert_to_tensor(value=np.asarray(ckpt['loss_block/batch_normalize_19tw/scale']), dtype=tf_dtypes['dtype'], dtype_hint=None)
    loss_block_batch_normalize_19tw_scale = tf.Variable(initial_value=loss_block_batch_normalize_19tw_scale_initializer_ones_initializer, trainable=False, caching_device=None, name='loss_block/batch_normalize_19tw/scale', variable_def=None, dtype=tf_dtypes['float32'], import_scope=None, constraint=None, synchronization=tf.VariableSynchronization.AUTO, shape=None)
    trainable_params['loss_block/batch_normalize_19tw/scale'] = loss_block_batch_normalize_19tw_scale
    loss_block_batch_normalize_19tw_variance_initializer_ones_initializer = tf.convert_to_tensor(value=np.asarray(ckpt['loss_block/batch_normalize_19tw/variance']), dtype=tf_dtypes['dtype'], dtype_hint=None)
    loss_block_batch_normalize_19tw_variance = tf.Variable(initial_value=loss_block_batch_normalize_19tw_variance_initializer_ones_initializer, trainable=False, caching_device=None, name='loss_block/batch_normalize_19tw/variance', variable_def=None, dtype=tf_dtypes['float32'], import_scope=None, constraint=None, synchronization=tf.VariableSynchronization.AUTO, shape=None)
    trainable_params['loss_block/batch_normalize_19tw/variance'] = loss_block_batch_normalize_19tw_variance
    loss_block_conv_21vm_filters_initializer_xavier_uniform = tf.convert_to_tensor(value=np.asarray(ckpt['loss_block/conv_21vm/filters']), dtype=tf_dtypes['dtype'], dtype_hint=None)
    loss_block_conv_21vm_filters = tf.Variable(initial_value=loss_block_conv_21vm_filters_initializer_xavier_uniform, trainable=False, caching_device=None, name='loss_block/conv_21vm/filters', variable_def=None, dtype=tf_dtypes['float32'], import_scope=None, constraint=None, synchronization=tf.VariableSynchronization.AUTO, shape=None)
    trainable_params['loss_block/conv_21vm/filters'] = loss_block_conv_21vm_filters
    loss_block_conv_23xc_filters_initializer_xavier_uniform = tf.convert_to_tensor(value=np.asarray(ckpt['loss_block/conv_23xc/filters']), dtype=tf_dtypes['dtype'], dtype_hint=None)
    loss_block_conv_23xc_filters = tf.Variable(initial_value=loss_block_conv_23xc_filters_initializer_xavier_uniform, trainable=False, caching_device=None, name='loss_block/conv_23xc/filters', variable_def=None, dtype=tf_dtypes['float32'], import_scope=None, constraint=None, synchronization=tf.VariableSynchronization.AUTO, shape=None)
    trainable_params['loss_block/conv_23xc/filters'] = loss_block_conv_23xc_filters
    return trainable_params


def model(data_block_input_data, trainable_params):
    model_block_conv_4eg = tf.nn.conv2d(input=data_block_input_data, filters=trainable_params['model_block/conv_4eg/filters'], strides=1, padding='SAME', data_format='NHWC', dilations=1, name='model_block/conv_4eg/filters')
    model_block_max_pool2d_6gw = tf.nn.max_pool(input=model_block_conv_4eg, ksize=3, strides=1, padding='VALID', data_format='NHWC', name='model_block/max_pool2d_6gw')
    model_block_max_pool2d_8im = tf.nn.max_pool(input=model_block_max_pool2d_6gw, ksize=3, strides=1, padding='VALID', data_format='NHWC', name='model_block/max_pool2d_8im')
    model_block_output = tf.reshape(tensor=model_block_max_pool2d_8im, shape=(-1, tf.math.reduce_prod(tf.convert_to_tensor([28, 28, 4]))), name='model_block/output')
    return model_block_output 


def get_loss(model_block_output, data_block_input_data, trainable_params):
    loss_block_conv_13na = tf.nn.conv2d(input=data_block_input_data, filters=trainable_params['loss_block/conv_13na/filters'], strides=1, padding='SAME', data_format='NHWC', dilations=1, name='loss_block/conv_13na/filters')
    loss_block_reluu = tf.nn.relu(name='loss_block/reluu', features=loss_block_conv_13na)
    loss_block_dropout_17rg = tf.nn.dropout(x=loss_block_reluu, rate=0.2, noise_shape=None, seed=None, name='loss_block/dropout_17rg')
    loss_block_batch_normalize_19tw = tf.nn.batch_normalization(x=loss_block_dropout_17rg, mean=trainable_params['loss_block/batch_normalize_19tw/mean'], variance=trainable_params['loss_block/batch_normalize_19tw/variance'], offset=trainable_params['loss_block/batch_normalize_19tw/offset'], scale=trainable_params['loss_block/batch_normalize_19tw/scale'], variance_epsilon=0.001, name='loss_block/batch_normalize_19tw/variance')
    loss_block_conv_21vm = tf.nn.conv2d(input=loss_block_batch_normalize_19tw, filters=trainable_params['loss_block/conv_21vm/filters'], strides=1, padding='VALID', data_format='NHWC', dilations=1, name='loss_block/conv_21vm/filters')
    loss_block_conv_23xc = tf.nn.conv2d(input=loss_block_conv_21vm, filters=trainable_params['loss_block/conv_23xc/filters'], strides=1, padding='VALID', data_format='NHWC', dilations=1, name='loss_block/conv_23xc/filters')
    loss_block_feature = tf.reshape(tensor=loss_block_conv_23xc, shape=(-1, tf.math.reduce_prod(tf.convert_to_tensor([28, 28, 64]))), name='loss_block/feature')
    loss_block_cross_0 = tf.keras.metrics.mean_squared_error(y_true=[loss_block_feature, model_block_output][0], y_pred=[loss_block_feature, model_block_output][1])
    loss_block_regularizer = 0.002*sum(list(map(lambda x: tf.nn.l2_loss(t=trainable_params[x], name='loss_block/regularizer'), ['model_block/conv_4eg/filters', 'loss_block/conv_13na/filters', 'loss_block/conv_21vm/filters', 'loss_block/conv_23xc/filters'])))
    loss_block_losses = tf.math.add(x=[loss_block_cross_0, loss_block_regularizer][0], y=[loss_block_cross_0, loss_block_regularizer][1], name='loss_block/losses')
    return loss_block_losses 


def get_optimizer():
    optimizer_block_solver_decay_exponential_decay = tf.keras.optimizers.schedules.ExponentialDecay(initial_learning_rate=0.0001, decay_steps=100000, decay_rate=0.96, staircase=True)
    optimizer_block_solver = tf.optimizers.Adam(learning_rate=optimizer_block_solver_decay_exponential_decay, beta_1=0.9, beta_2=0.999, epsilon=1e-08, name='optimizer_block/solver')
    return optimizer_block_solver 

from alex.alex.checkpoint import Checkpoint

C = Checkpoint("examples/configs/small1_linear.yml",
               ['checkpoints', 'test_code_gen_ckpt_trained.json'],
               None)

ckpt = C.load()

trainable_params = get_trainable_params(ckpt)

from alex.alex import registry
var_list = registry.get_trainable_params_list(trainable_params)

optimizer = get_optimizer()

probes = dict()

def inference(data_block_input_data, trainable_params):
    
    preds = model(data_block_input_data, trainable_params)
    
    return preds
    
def evaluation(data_block_input_data, trainable_params):
    
    preds = inference(data_block_input_data, trainable_params)
    
    loss = tf.reduce_mean(get_loss(preds, data_block_input_data, trainable_params))
    return loss
    
    
def train(var_list, data_block_input_data, trainable_params):
    
    with tf.GradientTape() as tape:
        preds = model(data_block_input_data, trainable_params)
        gradients = tape.gradient(get_loss(preds, data_block_input_data, trainable_params), var_list)
        optimizer.apply_gradients(zip(gradients, var_list))
    
    
def loop(var_list, val_inputs, trainable_params):
    
    for epoch in range(90):
        for i, batch in enumerate(trainloader):
            inputs = batch[0]
            labels = batch[1]
            train(var_list, inputs, trainable_params)
            if i % 500 == 499:
                results = evaluation(val_inputs, trainable_params)
                
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

loop(var_list, val_inputs, trainable_params)

