# ----------------------------------------------------------------------
# Created: mån jul 12 00:52:56 2021 (+0200)
# Last-Updated:
# Filename: validation.py
# Author: Yinan Yu
# Description:
# ----------------------------------------------------------------------
import numpy as np
from math import floor, ceil
from pprint import pprint
import traceback
from copy import deepcopy
from jsonschema import validate
from alex.alex import const, util
import os


class Ingredient():
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.transformation = None

    def load_schema(self, component_type):
        schema_path = os.path.join(const.COMPONENT_BASE_PATH,
                                   "." + component_type + ".yml")
        try:
            schema = util.read_yaml(schema_path)["definitions"][component_type]
        except Exception:
            schema = None
            # TODO:
            # print("%s schema not implemented" % component_type)
        return schema

    def eval_input_channels(self, component, components):
        inputs = self.get_input_shape(component, components)
        if inputs is not None:
            util.replace_key(component["value"]["hyperparams"],
                             "input_shape",
                             inputs[0][-1])

    def get_input_shape(self, component, components):
        if component["meta"]["inputs"] is None or component["meta"]["inputs"] == "inputs":
            return None

        return list(map(lambda x: components[x]["meta"]["shape"] if x in components else None,
                        component["meta"]["inputs"]))

    def get_shape(self, component, components):
        return self.get_input_shape(component, components)[0]

    def get_transformation(self):
        return self.transformation

    def get_dimensions(self, component, components):
        input_shape = self.get_input_shape(component, components)
        return [input_shape, # input shape
                self.get_shape(component, components)] # output shape

    def validate_schema(self, component, schema):
        validate(instance=component["value"]["hyperparams"], schema=schema)

    def validate_connection(self, component, components):
        transformations = self.get_transformation()
        dims = self.get_dimensions(component, components)
        if transformations is None:
            return True
        else:
            valid = False
            msg = ""
            for trans in transformations:
                if valid:
                    return valid
                try:
                    if len(trans[0]) != len(dims[0]):
                        msg = "Number of inputs mismatch %s (ingredient: %s, dim %s)" % (component["meta"]["name"],
                                                                                         component["meta"]["type"],
                                                                         str(dims))
                        raise Exception(msg)
                    elif len(list(filter(lambda x: len(x[0]) != len(x[1]),
                                     zip(trans[0], dims[0]))))!=0:
                        msg = "Tensor rank mismatch %s (ingredient: %s, dim: %s)" % (component["meta"]["name"],
                                                                                    component["meta"]["type"],
                                                                                     str(dims))
                        raise Exception(msg)
                    else:
                        trans_ = [i for sub in trans[0] for i in sub] + trans[1]
                        dims_ = [i for sub in dims[0] for i in sub] + dims[1]
                        n = len(trans_)
                        pattern_required = [trans_[i]==trans_[i+j] for i in range(n-1) for j in range(1, n-i)]
                        pattern_data = [dims_[i]==dims_[i+j] for i in range(n-1) for j in range(1, n-i)]
                        for r in pattern_required:
                            if pattern_required and (not pattern_data):
                                msg = "Tensor dimension mismatch" % component["meta"]["name"]
                                raise Exception(msg)
                    valid = True
                except:
                    pass
                    # traceback.print_exc()
                    # raise Exception(msg)
            if not valid:
                print(msg)
                raise Exception(msg)


# ------------------------------------------------------------------------ #
class Dense(Ingredient):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.transformation = [[[["a"]], ["b"]]]

    def get_shape(self, component, *args):
        return [int(component["value"]["hyperparams"]["weights"]["shape"]["n_units"]), ]

class CrossEntropy(Ingredient):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.transformation = [[[["a"], ["a"]], ["c"]]]

    def get_shape(self, component, components):
        return [0, ]


class Regularizer(Ingredient):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def get_shape(self, component, components):
        return [0, ]

    def get_input_shape(self, component, components):
        # if component["meta"]["inputs"] is None:
        #     return None
        # return list(map(lambda x: components[x]["meta"]["shape"],
        #                 component["meta"]["inputs"]))
        return None

class Batchnorm(Ingredient):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.transformation = [[[["a", "b", "c"]], ["a", "b", "c"]]]


class Add(Ingredient):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.transformation = [[[["a", "b", "c"], ["a", "b", "c"]], ["a", "b", "c"]],
                               [[["a", "b"], ["a", "b"]], ["a", "b"]],
                               [[["a"], ["a"]], ["a"]]]

    def get_shape(self, component, components):
        input_shape = self.get_input_shape(component, components)
        if input_shape is None or input_shape[0] is None:
            return None
        return input_shape[0]


class Conv2D(Ingredient):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.transformation = [[[["a", "b", "c"]], ["d", "e", "f"]]]

    def get_shape(self, component, components):
        input_shape = self.get_input_shape(component, components)
        if input_shape is None or input_shape[0] is None:
            return None
        input_shape = input_shape[0]
        hyperparams = component["value"]["hyperparams"]
        strides = hyperparams["strides"]
        dilation = hyperparams["dilation"]
        padding = hyperparams["padding"]
        filters_shape = hyperparams["filters"]["shape"]
        k_h = int(filters_shape["kernel_size_h"])
        k_w = int(filters_shape["kernel_size_w"])
        n_filters = int(filters_shape["n_filters"])
        h = input_shape[0]
        w = input_shape[1]
        if padding == "SAME":
            hout = ceil(h/strides[0])
            wout = ceil(w/strides[1])
            shape = [hout, wout, n_filters]
        elif padding == "VALID":
            padding = [0, 0]
            shape = [int(floor((h + 2*padding[0] - dilation*(k_h-1) - 1)/strides[0] + 1)),
                     int(floor((w + 2*padding[1] - dilation*(k_w-1) - 1)/strides[1] + 1)),
                     int(n_filters)]
        return shape


class MaxPool2D(Ingredient):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.required_input_rank = 3
        self.transformation = [[[["a", "b", "c"]], ["d", "e", "f"]]]

    def get_shape(self, component, components):
        input_shape = self.get_input_shape(component, components)
        if input_shape is None or input_shape[0] is None:
            return None
        input_shape = input_shape[0]
        hyperparams = component["value"]["hyperparams"]
        padding = hyperparams["padding"]
        strides = hyperparams["strides"]
        dilation = hyperparams["dilation"]
        kernel_shape = hyperparams["window_shape"]

        k_h = int(kernel_shape[0])
        k_w = int(kernel_shape[1])
        h = input_shape[0]
        w = input_shape[1]
        if padding == "SAME":
            hout = ceil(h/strides[0])
            wout = ceil(w/strides[1])
            shape = [hout, wout, input_shape[-1]]
        elif padding == "VALID":
            padding = [0, 0]
            shape = [floor((h + 2*padding[0] - dilation*(k_h-1) - 1)/strides[0] + 1),
                     floor((w + 2*padding[1] - dilation*(k_w-1) - 1)/strides[1] + 1),
                     input_shape[-1]]
        return shape


# None trainable
class Flatten(Ingredient):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.required_input_rank = None
        self.transformation = [[[["a", "b", "c"]], ["d"]],
                               [[["a", "b"]], ["d"]]]

    def get_shape(self, component, components):
        shape = list(map(lambda x: components[x]["meta"]["shape"],
                         component["meta"]["inputs"]))
        return [int(np.prod(np.asarray(shape))), ]


class Data(Ingredient):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def get_shape(self, component, *args):
        shape = component["value"]["hyperparams"]["shape"]
        return [shape["height"], shape["width"], shape["channels"]]


class Label(Ingredient):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def get_shape(self, component, *args):
        shape = component["value"]["hyperparams"]["shape"]
        return [shape["dim"], ]


INGREDIENTS = {"ingredient": Ingredient(),
               "data": Data(),
               "cross_entropy": CrossEntropy(),
               "label": Label(),
               "dense": Dense(),
               "add": Add(),
               "batch_normalize": Batchnorm(),
               "regularizer_l2": Regularizer(),
               "conv": Conv2D(),
               "max_pool2d": MaxPool2D(),
               "flatten": Flatten()}


def get(component):
    value = component["meta"]["type"]
    # TODO: group ingredient by block
    if value not in INGREDIENTS:
        value = "ingredient"
    return INGREDIENTS[value]
