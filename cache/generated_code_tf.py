import tensorflow as tf
import numpy as np


tf_dtypes = {'float32': tf.float32, 'int8': tf.int8}


def get_trainable_params(ckpt=None):
    trainable_params = dict()
    conv_5fo_filters_initializer_xavier_uniform = tf.keras.initializers.glorot_uniform(seed=1)(shape=(3, 3, 3, 16))
    conv_5fo_filters = tf.Variable(initial_value=conv_5fo_filters_initializer_xavier_uniform, trainable=True, caching_device=None, name='conv_5fo/filters', variable_def=None, dtype=tf_dtypes['float32'], import_scope=None, constraint=None, synchronization=tf.VariableSynchronization.AUTO, shape=None)
    trainable_params['conv_5fo/filters'] = conv_5fo_filters
    batch_normalize_11lk_mean_initializer_zeros_initializer = tf.zeros_initializer()(shape=[16, ])
    batch_normalize_11lk_mean = tf.Variable(initial_value=batch_normalize_11lk_mean_initializer_zeros_initializer, trainable=False, caching_device=None, name='batch_normalize_11lk/mean', variable_def=None, dtype=tf_dtypes['float32'], import_scope=None, constraint=None, synchronization=tf.VariableSynchronization.AUTO, shape=None)
    trainable_params['batch_normalize_11lk/mean'] = batch_normalize_11lk_mean
    batch_normalize_11lk_offset_initializer_zeros_initializer = tf.zeros_initializer()(shape=[16, ])
    batch_normalize_11lk_offset = tf.Variable(initial_value=batch_normalize_11lk_offset_initializer_zeros_initializer, trainable=True, caching_device=None, name='batch_normalize_11lk/offset', variable_def=None, dtype=tf_dtypes['float32'], import_scope=None, constraint=None, synchronization=tf.VariableSynchronization.AUTO, shape=None)
    trainable_params['batch_normalize_11lk/offset'] = batch_normalize_11lk_offset
    batch_normalize_11lk_scale_initializer_ones_initializer = tf.ones_initializer()(shape=[16, ])
    batch_normalize_11lk_scale = tf.Variable(initial_value=batch_normalize_11lk_scale_initializer_ones_initializer, trainable=True, caching_device=None, name='batch_normalize_11lk/scale', variable_def=None, dtype=tf_dtypes['float32'], import_scope=None, constraint=None, synchronization=tf.VariableSynchronization.AUTO, shape=None)
    trainable_params['batch_normalize_11lk/scale'] = batch_normalize_11lk_scale
    batch_normalize_11lk_variance_initializer_ones_initializer = tf.ones_initializer()(shape=[16, ])
    batch_normalize_11lk_variance = tf.Variable(initial_value=batch_normalize_11lk_variance_initializer_ones_initializer, trainable=False, caching_device=None, name='batch_normalize_11lk/variance', variable_def=None, dtype=tf_dtypes['float32'], import_scope=None, constraint=None, synchronization=tf.VariableSynchronization.AUTO, shape=None)
    trainable_params['batch_normalize_11lk/variance'] = batch_normalize_11lk_variance
    conv_13na_filters_initializer_xavier_uniform = tf.keras.initializers.glorot_uniform(seed=1)(shape=(3, 3, 16, 16))
    conv_13na_filters = tf.Variable(initial_value=conv_13na_filters_initializer_xavier_uniform, trainable=True, caching_device=None, name='conv_13na/filters', variable_def=None, dtype=tf_dtypes['float32'], import_scope=None, constraint=None, synchronization=tf.VariableSynchronization.AUTO, shape=None)
    trainable_params['conv_13na/filters'] = conv_13na_filters
    conv_15pq_filters_initializer_xavier_uniform = tf.keras.initializers.glorot_uniform(seed=1)(shape=(3, 3, 16, 16))
    conv_15pq_filters = tf.Variable(initial_value=conv_15pq_filters_initializer_xavier_uniform, trainable=True, caching_device=None, name='conv_15pq/filters', variable_def=None, dtype=tf_dtypes['float32'], import_scope=None, constraint=None, synchronization=tf.VariableSynchronization.AUTO, shape=None)
    trainable_params['conv_15pq/filters'] = conv_15pq_filters
    dense_19tw_bias_initializer_zeros_initializer = tf.zeros_initializer()(shape=[1, ])
    dense_19tw_bias = tf.Variable(initial_value=dense_19tw_bias_initializer_zeros_initializer, trainable=True, caching_device=None, name='dense_19tw/bias', variable_def=None, dtype=tf_dtypes['float32'], import_scope=None, constraint=None, synchronization=tf.VariableSynchronization.AUTO, shape=None)
    trainable_params['dense_19tw/bias'] = dense_19tw_bias
    dense_19tw_weights_initializer_xavier_uniform = tf.keras.initializers.glorot_uniform(seed=2)(shape=[4096, 10])
    dense_19tw_weights = tf.Variable(initial_value=dense_19tw_weights_initializer_xavier_uniform, trainable=True, caching_device=None, name='dense_19tw/weights', variable_def=None, dtype=tf_dtypes['float32'], import_scope=None, constraint=None, synchronization=tf.VariableSynchronization.AUTO, shape=None)
    trainable_params['dense_19tw/weights'] = dense_19tw_weights
    return trainable_params


def model(input_data, trainable_params, training):
    conv_5fo = tf.nn.conv2d(input=input_data, filters=trainable_params['conv_5fo/filters'], strides=2, padding='SAME', data_format='NHWC', dilations=1, name='conv_5fo/filters')
    reluu = tf.nn.relu(name='reluu', features=conv_5fo)
    dropout_9ju = tf.nn.dropout(x=reluu, rate=0.2, noise_shape=None, seed=None, name='dropout_9ju')
    batch_normalize_11lk = tf.nn.batch_normalization(x=dropout_9ju, mean=trainable_params['batch_normalize_11lk/mean'], variance=trainable_params['batch_normalize_11lk/variance'], offset=trainable_params['batch_normalize_11lk/offset'], scale=trainable_params['batch_normalize_11lk/scale'], variance_epsilon=0.001, name='batch_normalize_11lk/variance')
    conv_13na = tf.nn.conv2d(input=batch_normalize_11lk, filters=trainable_params['conv_13na/filters'], strides=1, padding='SAME', data_format='NHWC', dilations=1, name='conv_13na/filters')
    conv_15pq = tf.nn.conv2d(input=conv_13na, filters=trainable_params['conv_15pq/filters'], strides=1, padding='SAME', data_format='NHWC', dilations=1, name='conv_15pq/filters')
    flatten_17rg = tf.reshape(tensor=conv_15pq, shape=(-1, tf.math.reduce_prod(tf.convert_to_tensor([16, 16, 16]))), name='flatten_17rg')
    dense_19tw = tf.add(x=tf.matmul(a=flatten_17rg, b=trainable_params['dense_19tw/weights']), y=trainable_params['dense_19tw/bias'], name='dense_19tw/weights')
    d_1 = tf.nn.softmax(logits=dense_19tw, name='d_1')
    return d_1 


def get_loss(trainable_params, inputs):
    cross_0 = tf.nn.softmax_cross_entropy_with_logits(labels=inputs[0], logits=inputs[1], axis=-1, name='cross_0')
    regularizer = 0.002*sum(list(map(lambda x: tf.nn.l2_loss(t=trainable_params[x], name='regularizer'), ['conv_5fo/filters', 'conv_13na/filters', 'conv_15pq/filters', 'dense_19tw/weights'])))
    losses = tf.math.add(x=[cross_0, regularizer][0], y=[cross_0, regularizer][1], name='losses')
    return losses 


def get_optimizer(trainable_params):
    solver_decay_exponential_decay = tf.keras.optimizers.schedules.ExponentialDecay(initial_learning_rate=0.0001, decay_steps=100000, decay_rate=0.96, staircase=True)
    solver = tf.optimizers.Adam(learning_rate=solver_decay_exponential_decay, beta_1=0.9, beta_2=0.999, epsilon=1e-08, name='solver')
    return solver 


from tensorflow import keras
from tensorflow.keras import datasets
import matplotlib.pyplot as plt
import numpy as np

num_classes = 10
(x_train, label_train), (x_test, label_test) = datasets.cifar10.load_data()

# Normalize pixel values to be between 0 and 1
# Scale images to the [0, 1] range
x_train = x_train.astype("float32") / 255
x_test = x_test.astype("float32") / 255

# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(label_train, num_classes)
y_test = keras.utils.to_categorical(label_test, num_classes)


class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']

plt.figure(figsize=(10,10))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(x_train[i])
    # The CIFAR labels happen to be arrays,
    # which is why you need the extra index
    plt.xlabel(class_names[label_train[i][0]])
plt.show()


# from alex.alex.checkpoint import Checkpoint

# C = Checkpoint("examples/configs/small1.yml",
#                ["cache",  "config_1622420349826577.json"],
#                ["checkpoints", None])

# ckpt = C.load()

trainable_variables = get_trainable_params()

from alex.alex import const
var_list = []
for tv in trainable_variables:
    if tv.split("/")[-1] in const.ALL_TRAINABLE_PARAMS:
        var_list.append(trainable_variables[tv])


optimizer = get_optimizer(trainable_variables)


def train(x, gt, trainable_variables, var_list, optimizer):
    with tf.GradientTape() as tape:
        prediction = model(x, trainable_variables, training=True)
        gradients = tape.gradient(get_loss(trainable_variables, [gt, prediction]), var_list)
        optimizer.apply_gradients(zip(gradients, var_list))




num_epochs = 90
batch_size = 100

train_loss_results = []
train_accuracy_results = []

train_ds = tf.data.Dataset.from_tensor_slices(
    (x_train, y_train)).shuffle(50000).batch(batch_size)

test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(10000)
for x_test, y_test in test_ds:
    break


for epoch in range(num_epochs):
    for i, (batch_x, batch_y) in enumerate(train_ds):
      train(batch_x, batch_y, trainable_variables, var_list, optimizer)

    preds = model(x_test, trainable_variables, training=False)

    matches_test  = tf.equal(tf.math.argmax(preds,1), tf.math.argmax(y_test,1))

    epoch_accuracy = tf.reduce_mean(tf.cast(matches_test,tf.float32))
    current_loss = get_loss(trainable_variables, [preds, y_test])
    epoch_loss_avg = tf.reduce_mean(current_loss)
    train_loss_results.append(epoch_loss_avg)
    train_accuracy_results.append(epoch_accuracy)

    print("--- On epoch %i ---" % epoch)
    tf.print("Accuracy: ", epoch_accuracy, "| Loss: ",epoch_loss_avg)
    print("\n")


