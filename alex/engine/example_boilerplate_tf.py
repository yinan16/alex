import keras
from tensorflow.keras import datasets
import matplotlib.pyplot as plt


num_classes = 10
(x_train, label_train), (x_test, label_test) = datasets.cifar10.load_data()

# Normalize pixel values to be between 0 and 1
# Scale images to the [0, 1] range
x_train = x_train.astype("float32") / 255
x_test = x_test.astype("float32") / 255
# Make sure images have shape (28, 28, 1)
x_train = np.expand_dims(x_train, -1)
x_test = np.expand_dims(x_test, -1)

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


from alex.alex.checkpoint import Checkpoint

C = Checkpoint("examples/configs/small1.yml",
               ["cache",  "config_1622420349826577.json"],
               ["checkpoints", None])

ckpt = C.load()

trainable_variables = get_trainable_params(ckpt)


var_list = list(trainable_variables.values())


optimizer = get_optimizer(trainable_variables)


def train(x, gt, trainable_variables, var_list, optimizer):
    with tf.GradientTape() as tape:
        prediction = model(x, trainable_variables, training=True)
        gradients = tape.gradient(trainable_variables, get_loss([gt, prediction]), var_list)
        optimizer.apply_gradients(zip(gradients, var_list))




num_epochs = 10
batch_size = 100

train_loss_results = []
train_accuracy_results = []

train_ds = tf.data.Dataset.from_tensor_slices(
    (x_train, y_train)).shuffle(10000).batch(batch_size)

test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(1000)
for x_test, y_test in test_ds:
    break


for epoch in range(num_epochs):
    for i, (batch_x, batch_y) in enumerate(train_ds):
      train(batch_x, batch_y, trainable_variables, var_list, optimizer)

    preds = model(x_test, trainable_variables, training=False)

    matches_test  = tf.equal(tf.math.argmax(preds,1), tf.math.argmax(y_test,1))

    epoch_accuracy = tf.reduce_mean(tf.cast(matches_test,tf.float32))
    current_loss = get_loss(preds, y_test, trainable_variables)
    epoch_loss_avg = tf.reduce_mean(current_loss)
    train_loss_results.append(epoch_loss_avg)
    train_accuracy_results.append(epoch_accuracy)

    print("--- On epoch %i ---" % epoch)
    tf.print("Accuracy: ", epoch_accuracy, "| Loss: ",epoch_loss_avg)
    print("\n")
