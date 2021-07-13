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


## User defined layers:

### Linear regression 2D
def linear2d(x, weight, n_filters, k_w, k_h, dilation, strides):
    from math import floor
    import torch
    # Unfolding results in size [batch_size, k_w*k_h*channels, L]
    batch_size, channels, h, w = x.size()
    n_coeffs, L, n_filters = weight.size()
    x = torch.nn.functional.unfold(input=x,
                                   kernel_size=[k_w, k_h],
                                   dilation=dilation,
                                   padding=0,
                                   stride=strides)
    # weight has size [k_w*k_h*channels, L, n_filters]
    def lr2d(x, w):
        L = x.size()[-1]

        def _mult(_w):
            _out = torch.zeros((x.size()[0], L))
            for i in range(L):
                _out[:, i] = torch.inner(x[:, :, i], _w[:, i])
            return _out
        out = list(map(_mult, torch.unbind(w, 2)))
        out = torch.stack(out, 2)
        return out
    h_out = floor((h - dilation*(k_h-1) - 1)/strides[0] + 1)
    w_out = floor((w - dilation*(k_w-1) - 1)/strides[1] + 1)

    return lr2d(x, weight).transpose_(1, 2).view((batch_size,
                                                  n_filters,
                                                  h_out, w_out))
