# ----------------------------------------------------------------------
# Created: mån maj 17 21:52:56 2021 (+0200)
# Last-Updated:
# Filename: param_count.py
# Author: Yinan Yu
# Description:
# ----------------------------------------------------------------------
import numpy as np
from math import floor
from pprint import pprint
from copy import deepcopy
from alex.alex import core, const, node_interface
from alex.alex.annotator_interface import Annotator
from alex.engine import ns_alex


class Ingredient(node_interface.Ingredient):

    def get_shape(self, input_shape, *args):
        return input_shape

    def get_trainable_params_count(self, node, annotated):
        name = node["name"]
        return sum(list(map(lambda x: x["count"],
                            core.get_children(name,
                                              annotated))))


class Recipe(node_interface.Recipe):

    def get_shape(self, input_shape, *args):
        return input_shape

    def get_trainable_params_count(self, node, annotated):
        name = node["name"]
        return sum(list(map(lambda x: x["count"],
                            core.get_children(name,
                                              annotated))))


class Hyperparam(node_interface.Hyperparam):

    def get_trainable_params_count(self, node, annotated):
        if self.trainable:
            return np.prod(np.asarray(node["shape"]))
        else:
            return 0

    def get_shape(self, input_shape, *args):
        return [0, ]


# ------------------------------------------------------------------------ #
class Conv2D(Ingredient):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

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
            padding = [1, 1] # FIXME
        elif padding == "VALID":
            padding = [0, 0]
        shape = [floor((h + 2*padding[0] - dilation*(k_h-1) - 1)/strides[0] + 1),
                 floor((w + 2*padding[1] - dilation*(k_w-1) - 1)/strides[1] + 1),
                 n_filters]
        return shape


class MaxPool2D(Ingredient):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

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
            padding = [1, 1] # FIXME
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
    _nodes = {"data": Data,
              "label": Label,
              "conv": Conv2D,
              "max_pool2d": MaxPool2D,
              "flatten": Flatten,
              "filters": Conv2DFilters,
              "weights": DenseWeights,
              "bias": DenseBias,
              "means": BatchnormMean,
              "vairance": BatchnormVariance,
              "offset": BatchnormOffset,
              "scale": BatchnormOffset,
              "ingredient": Ingredient,
              "hyperparam": Hyperparam,
              "recipe": Recipe}
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
        input_names = self.get_input_names(name, self.annotated)
        # FIXME: fix cache block info logic
        if input_nodes is None:
            node["block"] = "data"
        elif node["value"] in const.SCHEDULER_BLOCK:
            node["block"] = None
        elif node["value"] in const.OPTIMIZER_BLOCK:
            node["block"] = None
        elif node["value"] in const.LOSS_BLOCK:
            node["block"] = "loss"
        else:
            if input_nodes is not None:
                for _input_node in input_nodes:
                    if _input_node is not None and ("block" in _input_node) and (_input_node["block"] == "loss"):
                        node["block"] = "loss"
                        break
        # FIXME: for regularization
        if input_nodes is not None and input_nodes[0] is None:
            input_nodes = None
        if input_nodes is not None:
            input_shape = input_nodes[0]["shape"]
            node["shape"] = nodes(node).get_shape(input_shape,
                                                  name,
                                                  self.annotated)
            if node["value"] == "channels":
                node["value"] = input_shape[-1]
                node["name"] = input_shape[-1]
                # print(input_shape, name, node["shape"])
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
        node["input_names"] = input_names
        node["input_shape"] = input_shape
        node["ancestor"] = core.get_ancestor_ingredient_node(node,
                                                             self.components,
                                                             self.tree)
        if node["ancestor"]["name"] in self.components:
            node["dtype"] = self.components[node["ancestor"]["name"]]["meta"]["dtype"]
        else:
            node["dtype"] = None
        return node
