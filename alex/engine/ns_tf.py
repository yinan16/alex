# ----------------------------------------------------------------------
# Created: ons mar  3 23:40:02 2021 (+0100)
# Last-Updated:
# Filename: ns_tf.py
# Author: Yinan Yu
# Description:
# ----------------------------------------------------------------------
from alex.alex import const
from alex.engine import ns_alex


ENGINE = "tf"
DIM_ORDER = ["batch_size", "height", "width", "channel"]


def add_imports(additional_modules=[]):
    default_modules = [["tensorflow", "tf"],
                       ["numpy", "np"]]
    modules = default_modules + additional_modules
    imports = ""
    for m in modules:
        imports += "import %s " % m[0]
        if len(m) == 2:
            imports += "as %s\n" % m[1]
        elif len(m) == 1:
            imports += "\n"
    imports += "\n\n"
    dtypes = "tf_dtypes = {'float32': tf.float32, 'int8': tf.int8}\n"
    space = "\n\n"
    return imports + dtypes + space


def add_global_configs():
    return ""


def wrap_in_class(trainable_params_code, component_code, loss_code, optimizer_code, *args, **kargs):
    code = []
    code.append(trainable_params_code)
    code.append(component_code)
    code.append(loss_code)
    code.append(optimizer_code)
    code = "\n\n".join(code)
    return code


def instantiate(config, load_from, save_to):
    return """
from alex.alex.checkpoint import Checkpoint

C = Checkpoint("%s",
               %s,
               %s)

ckpt = C.load()

trainable_variables = get_trainable_params(ckpt)

from alex.alex import registry
var_list = []
for tv in trainable_variables:
    if tv.split('/')[-1] in registry.ALL_TRAINABLE_PARAMS:
        var_list.append(trainable_variables[tv])

optimizer = get_optimizer(trainable_variables)

""" % (config, str(load_from), str(save_to))


def training(save_to):
    if save_to:
        save_str = "C.save(model.trainable_params)"
    else:
        save_str = ""
    return """
def train(x, gt, trainable_variables, var_list, optimizer):
    with tf.GradientTape() as tape:
        prediction = model(x, trainable_variables, training=True)
        gradients = tape.gradient(get_loss(trainable_variables, [gt, prediction]), var_list)
        optimizer.apply_gradients(zip(gradients, var_list))

def loop(trainloader, val_inputs, val_labels):
    for epoch in range(90):
        for i, (inputs, labels) in enumerate(trainloader):
            train(inputs, labels, trainable_variables, var_list, optimizer)
            if i %% 500 == 499:
                accuracy, loss = validation(val_inputs, val_labels)
                %s
                tf.print("[", epoch, i, "500]", "accuracy: ", accuracy, ", loss: ", loss)
    print('Finished Training')

""" % save_str

def validation():
    return \
    """
def validation(inputs, labels):
    preds = model(inputs, trainable_variables, training=False)
    matches  = tf.equal(tf.math.argmax(preds,1), tf.math.argmax(labels,1))
    accuracy = tf.reduce_mean(tf.cast(matches,tf.float32))
    loss = tf.reduce_mean(get_loss(trainable_variables, [preds, labels]))
    return accuracy, loss

"""
