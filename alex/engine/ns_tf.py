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
