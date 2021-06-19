# ----------------------------------------------------------------------
# Created: mån mar 29 11:10:21 2021 (+0200)
# Last-Updated:
# Filename: ns_pytorch.py
# Author: Yinan Yu
# Description:
# ----------------------------------------------------------------------
from alex.alex import const
from alex.engine import ns_alex


ENGINE = "pytorch"
DIM_ORDER = ["batch_size", "channel", "height", "width"]


def add_imports(additional_modules=[]):
    default_modules = [["torch"],
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
    return imports


def add_global_configs():
    deterministic = "torch.backends.cudnn.deterministic = True\n"
    device = "device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')\n"
    dtypes = "torch_types = {'float32': torch.float32, 'int8': torch.int8}\n"
    space = "\n\n"
    return deterministic + device + dtypes + space


def wrap_in_class(trainable_params_code, component_code, loss_code, optimizer_code, scheduler_code=""):
    code = []
    code.append("class Model(torch.nn.Module):")
    code.append("")
    code.append(ns_alex.indent("def __init__(self, ckpt=None):", levels=1))
    code.append(ns_alex.indent("super(Model, self).__init__()", levels=2))
    code.append(ns_alex.indent("self.%s = self.get_trainable_params(ckpt)" % (const.ALEX_ARG_TRAINABLE_PARAMS), levels=2))
    code.append(ns_alex.indent("self.params = []", levels=2))
    code.append(ns_alex.indent("for var in self.%s:" % const.ALEX_ARG_TRAINABLE_PARAMS, levels=2))
    code.append(ns_alex.indent("self.register_parameter(var, self.%s[var])" % const.ALEX_ARG_TRAINABLE_PARAMS, levels=3))
    code.append(ns_alex.indent("self.params.append({'params': self.%s[var]})" % const.ALEX_ARG_TRAINABLE_PARAMS, levels=3))
    code.append("")
    code.append(ns_alex.indent("def forward(self, x, training):", levels=1))
    code.append(ns_alex.indent("x = self.model(x, self.%s, training)" % const.ALEX_ARG_TRAINABLE_PARAMS,
                               levels=2))
    code.append(ns_alex.indent("return x", levels=2))
    code.append("")
    code.append(ns_alex.indent("@staticmethod", levels=1))
    code.append(ns_alex.indent(trainable_params_code.replace("\n", "\n\t"), levels=1))
    code.append(ns_alex.indent("@staticmethod", levels=1))
    code.append(ns_alex.indent(component_code.replace("\n", "\n\t"), levels=1))
    code.append(ns_alex.indent("@staticmethod", levels=1))
    code.append(ns_alex.indent(loss_code.replace("\n", "\n\t"), levels=1))
    code.append(ns_alex.indent("@staticmethod", levels=1))
    code.append(ns_alex.indent(optimizer_code.replace("\n", "\n\t"), levels=1))
    if scheduler_code != "":
        code.append(ns_alex.indent("@staticmethod", levels=1))
        code.append(ns_alex.indent(scheduler_code.replace("\n", "\n\t"), levels=1))
    code = "\n".join(code)
    return code


def flatten():
    src_fn, src_args, annotated_args = ns_alex.flatten()
    trg_signature = "%s%s" % (ns_alex.get_component_fn("flatten", ENGINE),
                              "(input=%s, start_dim=1, end_dim=-1)" % const.ALEX_ARG_INPUTS)
    src = ns_alex.get_translation_code(trg_signature, src_args, src_fn)
    return src, annotated_args


# FIXME: refactor this
def initializer(initializer_type, component_type, trainable_params):
    channel = "list(%s.size())[1]" % const.ALEX_ARG_INPUTS
    if component_type == "conv" and trainable_params == "filters":
        shape = "n_filters, %s, kernel_size_h, kernel_size_w" % channel

    elif component_type == "dense":
        if trainable_params == "weights":
            shape = "n_units, %s" % channel
        elif trainable_params == "bias":
            shape = "dim"
    elif component_type == "batch_normalize":
        shape = channel

    src_fn, src_args, annotated_args = ns_alex.initializer(initializer_type, component_type, trainable_params)
    trg_signature = "%s(tensor=torch.empty(%s, dtype=torch_types[dtype]))" % (ns_alex.get_component_fn(initializer_type, ENGINE), shape)
    src = ns_alex.get_translation_code(trg_signature, src_args, src_fn)

    return src, annotated_args


def alex_trainable_params():
    src_fn, src_args, annotated_args = ns_alex.alex_trainable_params()
    trg_signature = "%s%s" % (const.CONSTRUCTORS["params"][ENGINE][0],
                              "(data=%s, requires_grad=requires_grad)" % (const.ALEX_ARG_INITIAL_VALUE))
    src = ns_alex.get_translation_code(trg_signature, src_args, src_fn)
    return src, annotated_args


def conv():
    src_fn, src_args, annotated_args = ns_alex.conv()
    trg_signature = "%s%s" % (ns_alex.get_component_fn("conv", ENGINE),
                              "(input=%s, weight=filters, bias=None, stride=strides, padding=padding, dilation=1, groups=1)" % (const.ALEX_ARG_INPUTS))
    src = ns_alex.get_translation_code(trg_signature, src_args, src_fn)
    return src, annotated_args


def dense():
    src_fn, src_args, annotated_args = ns_alex.dense()
    trg_fn = ns_alex.get_component_fn("dense", ENGINE)
    trg_signature = "%s(input=%s, weight=weights, bias=bias)" % (trg_fn,
                                                                 const.ALEX_ARG_INPUTS)
    src = ns_alex.get_translation_code(trg_signature, src_args, src_fn)
    return src, annotated_args


def batch_normalize():
    src_fn, src_args, annotated_args = ns_alex.batch_normalize()
    trg_fn = ns_alex.get_component_fn("batch_normalize", ENGINE)
    trg_signature = "%s(input=%s, running_mean=mean, running_var=variance, weight=scale, bias=offset, training=training, momentum=momentum, eps=epsilon)" % (trg_fn, const.ALEX_ARG_INPUTS)
    src = ns_alex.get_translation_code(trg_signature, src_args, src_fn)
    return src, annotated_args


def dropout():
    src_fn, src_args, annotated_args = ns_alex.dropout()
    trg_fn = ns_alex.get_component_fn("dropout", ENGINE)
    trg_signature = "%s(input=%s, p=%s, training=%s, inplace=False)" % (trg_fn,
                                                                        const.ALEX_ARG_INPUTS,
                                                                        "dropout_rate",
                                                                        "training")
    src = ns_alex.get_translation_code(trg_signature, src_args, src_fn)
    return src, annotated_args


def cross_entropy():
    src_fn, src_args, annotated_args = ns_alex.cross_entropy()
    trg_fn = ns_alex.get_component_fn("cross_entropy", ENGINE)
    trg_signature = "%s(weight=None, ignore_index=-100, reduction='mean', target=%s[0], input=%s[1])" % (trg_fn,
                                                                                                         const.ALEX_ARG_INPUTS,
                                                                                                         const.ALEX_ARG_INPUTS)
    src = ns_alex.get_translation_code(trg_signature, src_args, src_fn)
    return src, annotated_args


# TODO
def adam():
    src_fn, src_args, annotated_args = ns_alex.adam()
    trg_fn = ns_alex.get_component_fn("adam", ENGINE)
    trg_signature = "{'optimizer': %s(params=%s, lr=initial_learning_rate, betas=(beta1, beta2), eps=epsilon), 'learning_rate': lambda optimizer: learning_rate}" % (trg_fn, const.ALEX_ARG_TRAINABLE_PARAMS)
    src = ns_alex.get_translation_code(trg_signature, src_args, src_fn)
    return src, annotated_args


def exponential_decay():
    src_fn, src_args, annotated_args = ns_alex.exponential_decay()
    trg_fn = ns_alex.get_component_fn("exponential_decay", ENGINE)
    trg_signature = "%s(optimizer=optimizer, gamma=decay_rate, last_epoch=-1, verbose=False)" % (trg_fn)
    src = ns_alex.get_translation_code(trg_signature, src_args, src_fn)
    return src, annotated_args


def regularizer_l2():
    src_fn, src_args, annotated_args = ns_alex.regularizer_l2()
    trg_fn = ns_alex.get_component_fn("regularizer_l2", ENGINE)
    # trg_signature = "coeff*torch.sum(torch.vmap(lambda x: %s(trainable_params_name[x]))(inputs))" % (trg_fn)
    trg_signature = "coeff*sum(list(map(lambda x: %s(%s[x]), inputs_str)))" % (trg_fn, const.ALEX_ARG_TRAINABLE_PARAMS_NAME)
    src = ns_alex.get_translation_code(trg_signature, src_args, src_fn)
    return src, annotated_args


def zeros():
    src_fn, src_args, annotated_args = ns_alex.zeros()
    trg_signature = "%s%s" % (ns_alex.get_component_fn("zeros", ENGINE), "(%s)" % (const.ALEX_ARG_INPUT_SHAPE))
    src = ns_alex.get_translation_code(trg_signature, src_args, src_fn)
    return src, annotated_args


def relu():
    src_fn, src_args, annotated_args = ns_alex.relu()
    trg_signature = "%s%s" % (ns_alex.get_component_fn("relu", ENGINE), "(input=%s, inplace=False)" % (const.ALEX_ARG_INPUTS))
    src = ns_alex.get_translation_code(trg_signature, src_args, src_fn)
    return src, annotated_args


def softmax():
    src_fn, src_args, annotated_args = ns_alex.softmax()
    trg_signature = "%s%s" % (ns_alex.get_component_fn("softmax", ENGINE), "(input=%s)" % (const.ALEX_ARG_INPUTS))
    src = ns_alex.get_translation_code(trg_signature, src_args, src_fn)
    return src, annotated_args


def sigmoid():
    src_fn, src_args, annotated_args = ns_alex.sigmoid()
    trg_signature = "%s%s" % (ns_alex.get_component_fn("sigmoid", ENGINE), "(input=%s)" % (const.ALEX_ARG_INPUTS))
    src = ns_alex.get_translation_code(trg_signature, src_args, src_fn)
    return src, annotated_args


def add():
    src_fn, src_args, annotated_args = ns_alex.add()
    trg_signature = "%s%s" % (ns_alex.get_component_fn("add", ENGINE), "(input=%s[0], other=%s[1])" % (const.ALEX_ARG_INPUTS, const.ALEX_ARG_INPUTS))
    src = ns_alex.get_translation_code(trg_signature, src_args, src_fn)
    return src, annotated_args


# FIXME: fix hard coded padding=0 in pytorch
def max_pool2d():
    src_fn, src_args, annotated_args = ns_alex.max_pool2d()
    trg_signature = "%s%s" % (ns_alex.get_component_fn("max_pool2d", ENGINE),
                              "(input=%s, kernel_size=window_shape, stride=strides, padding=0)" % (const.ALEX_ARG_INPUTS))
    src = ns_alex.get_translation_code(trg_signature, src_args, src_fn)
    return src, annotated_args


FN_TABLE = {"conv": conv,
            "dense": dense,
            "trainable_params": alex_trainable_params,
            "initializer": initializer,
            "flatten": flatten,
            "adam": adam,
            "exponential_decay": exponential_decay,
            "cross_entropy": cross_entropy,
            "regularizer_l2": regularizer_l2,
            "zeros": zeros,
            "relu": relu,
            "batch_normalize": batch_normalize,
            "add": add,
            "dropout": dropout,
            "softmax": softmax,
            "max_pool2d": max_pool2d,
            "sigmoid": sigmoid}

# TODO: deconv
