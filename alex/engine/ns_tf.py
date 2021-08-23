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


DEFINED = {"tf", "np", "tf_dtypes"}

LOOP_ARGS = ["trainloader", "val_inputs", "val_labels", "var_list"]

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


def instantiate(config, load_from, save_to, params_args, optimizer_args):
    return """
from alex.alex.checkpoint import Checkpoint

C = Checkpoint("%s",
               %s,
               %s)

ckpt = C.load()

trainable_params = get_trainable_params(%s)

from alex.alex import registry
var_list = registry.get_trainable_params_list(trainable_params)

optimizer = get_optimizer(%s)

""" % (config,
       str(load_from),
       str(save_to),
       ", ".join(params_args),
       ", ".join(optimizer_args))


def train(model_args, loss_args):
    func_name = "train"
    return func_name, """
with tf.GradientTape() as tape:
    preds = model(%s)
    gradients = tape.gradient(get_loss(%s), var_list)
    optimizer.apply_gradients(zip(gradients, var_list))
""" % (", ".join(model_args),
       ", ".join(loss_args).replace("inputs", "[labels, preds]"))


def loop(save_to, train_args, evaluation_args):
    if save_to:
        save_str = "C.save(model.trainable_params)"
    else:
        save_str = ""
    func_name = "loop"
    return func_name, """
for epoch in range(90):
    for i, (inputs, labels) in enumerate(trainloader):
        train(%s)
        if i %% 500 == 499:
            results = evaluation(%s)
            %s
            tf.print(results)
print('Finished Training')

""" % (", ".join(train_args).replace("data_block_input_data", "inputs"), ", ".join(evaluation_args).replace("data_block_input_data", "val_inputs").replace("labels", "val_labels"), save_str)


def inference(model_args, mode):
    if mode == "classification":
        code_str = """
preds = tf.math.argmax(model(%s), 1)
""" % (", ".join(model_args))
    elif mode == "regression":
        code_str = """
preds = model(%s)\n
""" % (", ".join(model_args))
    code_str += "return preds"
    func_name = "inference"

    return func_name, code_str


def evaluation(inference_args, loss_args, mode="classification"):
    if mode == "classification":
        evaluation_str = """
matches = tf.equal(preds, tf.math.argmax(labels, 1))
perf = tf.reduce_mean(tf.cast(matches, tf.float32))
"""
        return_str = "return perf, loss"
    else:
        evaluation_str = ""
        return_str = "return loss"

    func_name = "evaluation"
    code_str = """
preds = inference(%s)
%s
loss = tf.reduce_mean(get_loss(%s))
%s
""" % (", ".join(inference_args),
       evaluation_str,
       ", ".join(loss_args).replace("inputs", "[labels, preds]"),
       return_str)

    return func_name, code_str
