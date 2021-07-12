# ----------------------------------------------------------------------
# Created: mån maj 17 21:52:56 2021 (+0200)
# Last-Updated:
# Filename: param_count.py
# Author: Yinan Yu
# Description:
# ----------------------------------------------------------------------
import numpy as np
from math import floor, ceil
from pprint import pprint
import traceback
from copy import deepcopy
from alex.alex import core, const, node_interface
from alex.alex.annotator_interface import Annotator
from alex.engine import ns_alex


class Ingredient(node_interface.Ingredient):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.transformation = None

    def get_shape(self, input_shape, *args):
        return input_shape

    def get_trainable_params_count(self, node, annotated):
        name = node["name"]
        return sum(list(map(lambda x: x["count"],
                            core.get_children(name,
                                              annotated))))

    def get_dimensions(self, node):
        input_shape = node["meta"]["position"]["input_shape"]
        if input_shape is not None:
            input_shape = [input_shape] if not isinstance(input_shape[0], list) else input_shape  # FIXME: backward compatibility, remove (see same note below)
        return [input_shape, # input shape
                node["shape"]] # output shape

    def get_transformation(self):
        return self.transformation

    def validate_connection(self, node):
        transformations = self.get_transformation()
        dims = self.get_dimensions(node)
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
                        msg = "Number of inputs mismatch %s (ingredient: %s, dim %s)" % (node["name"],
                                                                                         node["value"],
                                                                         str(dims))
                        raise Exception(msg)
                    elif len(list(filter(lambda x: len(x[0]) != len(x[1]),
                                     zip(trans[0], dims[0]))))!=0:
                        msg = "Tensor rank mismatch %s (ingredient: %s, dim: %s)" % (node["name"],
                                                                                     node["value"],
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
                            msg = "Tensor dimension mismatch" % node["name"]
                            raise Exception(msg)
                    valid = True
                except:
                    # traceback.print_exc()
                    pass
            if not valid:
                traceback.print_exc()
                raise Exception


class Recipe(node_interface.Recipe):

    def get_shape(self, input_shape, *args):
        return input_shape

    def get_trainable_params_count(self, node, annotated):
        name = node["name"]
        return sum(list(map(lambda x: x["count"],
                            core.get_children(name,
                                              annotated))))

    def validate_connection(self, node):
        pass


class Hyperparam(node_interface.Hyperparam):

    def get_trainable_params_count(self, node, annotated):
        if self.trainable:
            return np.prod(np.asarray(node["shape"]))
        else:
            return 0

    def get_shape(self, input_shape, *args):
        return [0, ]

    def validate_connection(self, node):
        pass

# ------------------------------------------------------------------------ #
class Dense(Ingredient):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.transformation = [[[["a"]], ["b"]]]


class CrossEntropy(Ingredient):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.transformation = [[[["a"], ["a"]], ["c"]]]

    def get_shape(self, input_shape, name, annotated):
        return [0]


class Regularizer(Ingredient):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def get_shape(self, input_shape, name, annotated):
        return [0]


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

    def get_shape(self, input_shape, name, annotated):
        return input_shape[0]


class Conv2D(Ingredient):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.transformation = [[[["a", "b", "c"]], ["d", "e", "f"]]]

    def get_shape(self, input_shape, name, annotated):
        hyperparams = annotated[name]["meta"]["hyperparams"]
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
            shape = [floor((h + 2*padding[0] - dilation*(k_h-1) - 1)/strides[0] + 1),
                     floor((w + 2*padding[1] - dilation*(k_w-1) - 1)/strides[1] + 1),
                     n_filters]
        return shape


class MaxPool2D(Ingredient):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.required_input_rank = 3
        self.transformation = [[[["a", "b", "c"]], ["d", "e", "f"]]]

    def get_shape(self, input_shape, name, annotated):
        hyperparams = annotated[name]["meta"]["hyperparams"]
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


class Conv2DFilters(Hyperparam):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.trainable = True

    def get_shape(self, input_shape, name, annotated):
        shape = annotated[name]["meta"]["hyperparams"]["shape"]
        return [input_shape[-1],
                shape["kernel_size_h"],
                shape["kernel_size_w"],
                shape["n_filters"]]


class DenseWeights(Hyperparam):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.trainable = True

    def get_shape(self, input_shape, name, annotated):
        return [input_shape[-1],
                annotated[name]["meta"]["hyperparams"]["shape"]["n_units"]]


class DenseBias(Hyperparam):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.trainable = True

    def get_shape(self, input_shape, *args):
        return [1, ]


class BatchnormMean(Hyperparam):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def get_shape(self, input_shape, *args):
        return [input_shape[-1], ]


class BatchnormVariance(Hyperparam):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def get_shape(self, input_shape, *args):
        return [input_shape[-1], ]


class BatchnormOffset(Hyperparam):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.trainable = True

    def get_shape(self, input_shape, *args):
        return [input_shape[-1], ]


class BatchnormScale(Hyperparam):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.trainable = True

    def get_shape(self, input_shape, *args):
        return [input_shape[-1], ]


# None trainable
class Flatten(Ingredient):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.required_input_rank = None
        self.transformation = [[[["a", "b", "c"]], ["d"]],
                               [[["a", "b"]], ["d"]]]

    def get_shape(self, input_shape, *args):
        return [np.prod(np.asarray(input_shape)), ]


class Data(Ingredient):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def get_shape(self, shape):
        return [shape["height"], shape["width"], shape["channels"]]


class Label(Ingredient):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def get_shape(self, shape):
        return [shape["dim"], ]


def nodes(node):
    value = node["value"]
    # TODO: group ingredient by block
    nodes_ingredient = {"ingredient": Ingredient,
                        "data": Data,
                        "cross_entropy": CrossEntropy,
                        "label": Label,
                        "dense": Dense,
                        "add": Add,
                        "batch_normalize": Batchnorm,
                        "regularizer_l2": Regularizer,
                        "conv": Conv2D,
                        "max_pool2d": MaxPool2D,
                        "flatten": Flatten}
    nodes_hyperparam = {"hyperparam": Hyperparam}
    nodes_param = {"filters": Conv2DFilters,
                   "weights": DenseWeights,
                   "bias": DenseBias,
                   "means": BatchnormMean,
                   "vairance": BatchnormVariance,
                   "offset": BatchnormOffset,
                   "scale": BatchnormOffset}
    nodes_recipe = {"recipe": Recipe}

    _nodes = {**nodes_hyperparam,
              **nodes_ingredient,
              **nodes_param,
              **nodes_recipe}
    if value not in _nodes:
        value = node["type"]
    return _nodes[value](node["value"])


class ParamCount(Annotator):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.anno_name = "Trainable params count"
        self.passes = [[self.cache_input,
                        self.annotate_count]]

    def annotate_count(self, node):
        node = deepcopy(node)
        node["count"] = nodes(node).get_trainable_params_count(node=node,
                                                               annotated=self.annotated)
        return node

    def cache_input(self, node):
        """Cache the shape info in the node"""
        node = deepcopy(node)
        name = node["name"]
        input_nodes = self.get_input_nodes(name, self.annotated)
        if input_nodes is not None and input_nodes[0] is None:
            input_nodes = None
        if input_nodes is not None:
            input_shape = []
            for _input_node in input_nodes:
                input_shape.append(_input_node["shape"])
            if len(input_shape) == 1:
                input_shape = input_shape[0] # FIXME: backward compatibility, remove
            node["shape"] = nodes(node).get_shape(input_shape,
                                                  name,
                                                  self.annotated)
            if node["value"] == "channels":
                node["value"] = input_shape[-1]
                node["name"] = input_shape[-1]
        else:
            input_shape = None
            if name in self.components:
                node["type"] = "ingredient"
                component = self.components[name]
                hyperparams = component["value"]["hyperparams"]
                if "shape" in hyperparams:
                    node["shape"] = nodes(node).get_shape(hyperparams["shape"])
                else:
                    node["shape"] = None
        if input_nodes is None:
            inputs = None
        else:
            inputs = list(map(lambda x: x["name"], input_nodes))
        node["input_nodes"] = inputs
        node["meta"]["position"]["input_shape"] = input_shape
        node["ancestor"] = core.get_ancestor_ingredient_node(node, self.tree)
        if node["ancestor"]["name"] in self.components:
            node["dtype"] = self.components[node["ancestor"]["name"]]["meta"]["dtype"]
        else:
            node["dtype"] = None
        nodes(node).validate_connection(node)
        return node
