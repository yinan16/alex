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
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.required_input_rank = None
        self.required_input_num = None

    def get_shape(self, input_shape, *args):
        return input_shape

    def get_trainable_params_count(self, node, annotated):
        name = node["name"]
        return sum(list(map(lambda x: x["count"],
                            core.get_children(name,
                                              annotated))))

    def get_rank(self, node):
        return [len(node["meta"]["position"]["input_shape"]), # input rank
                len(node["shape"])] # output rank

    def get_required_input_rank(self):
        return self.required_input_rank


    def get_required_input_num(self):
        return self.required_input_num

    def validate_input_dims(self, node):
        # Validate if input dimensions are valid
        pass

    def validate_input_rank(self, node):
        required_input_rank = self.get_required_input_rank()
        if required_input_rank is not None:
            ranks = self.get_rank(node)
            if required_input_rank != ranks[0]:
                raise Exception("""Connection is invalid between
                %s (required rank %s) and %s (rank %s)""" % (node["name"],
                                                             str(required_input_rank),
                                                             str(node["meta"]["position"]["inputs"]),
                                                             str(ranks[0])))

    def validate_input_num(self, node):
        if self.get_required_input_num() is not None and node["meta"]["position"]["inputs"] is not None:
            if self.get_required_input_num() != len(node["meta"]["position"]["inputs"]):
                raise Exception("""Connection is invalud between
                %s (required inputs %s) and
                %s (inputs %s)""" % (node["name"],
                                     str(self.get_required_input_num()),
                                     str(node["meta"]["position"]["inputs"]),
                                     str(len(node["meta"]["position"]["inputs"]))))

    def validate_connection(self, node):
        self.validate_input_num(node)
        self.validate_input_rank(node)
        self.validate_input_dims(node)


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
        self.required_input_rank = 1


class CrossEntropy(Ingredient):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.required_input_num = 2


class Batchnorm(Ingredient):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.required_input_rank = 3


class Add(Ingredient):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.required_input_num = 2

    def validate_input_dims(self, node):
        # TODO
        pass


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
        self.required_input_rank = 3

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
        self.required_input_rank = None

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
    nodes_ingredient = {"ingredient": Ingredient,
                        "data": Data,
                        "cross_entropy": CrossEntropy,
                        "label": Label,
                        "dense": Dense,
                        "add": Add,
                        "batch_normalize": Batchnorm,
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
            # FIXME: do not handle validation for more than one inputs
            input_shape = input_nodes[0]["shape"]
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
