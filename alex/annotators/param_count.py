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

    def get_shape(self, node, *args, **kwargs):
        return node["meta"]["position"]["shape"]

    def get_trainable_params_count(self, node, annotated):
        name = node["name"]
        return sum(list(map(lambda x: x["count"],
                            core.get_children(name,
                                              annotated))))


class Recipe(node_interface.Recipe):

    def get_shape(self, node, annotated):
        last_component = annotated[node["children"][-1]]
        input_shape = last_component["meta"]["position"]["input_shape"][0]
        return input_shape

    def get_trainable_params_count(self, node, annotated):
        name = node["name"]
        return sum(list(map(lambda x: x["count"],
                            core.get_children(name,
                                              annotated))))


class Hyperparam(node_interface.Hyperparam):

    def get_trainable_params_count(self, node, annotated):
        if self.trainable:
            return np.prod(np.asarray(self.get_shape(node, annotated)))
        else:
            return 0

    def get_shape(self, *args, **kwargs):
        return [0, ]

    def get_input_shape(self, node):
        return node["meta"]["position"]["input_shape"]

# ------------------------------------------------------------------------ #
class Conv2DFilters(Hyperparam):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.trainable = True

    def get_shape(self, node, annotated):
        # input_shape = annotated[node["meta"]["position"]["component"]]["meta"]["position"]["input_shape"][0]
        input_shape = self.get_input_shape(node)[0]
        shape = node["meta"]["hyperparams"]["shape"]
        return [input_shape[-1],
                shape["kernel_size_h"],
                shape["kernel_size_w"],
                shape["n_filters"]]


class DenseWeights(Hyperparam):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.trainable = True

    def get_shape(self, node, annotated):
        # input_shape = annotated[node["meta"]["position"]["component"]]["meta"]["position"]["input_shape"][0]
        input_shape = self.get_input_shape(node)[0]
        return [input_shape[-1],
                node["meta"]["hyperparams"]["shape"]["n_units"]]


class DenseBias(Hyperparam):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.trainable = True

    def get_shape(self, *args, **kwargs):
        return [1, ]


class BatchnormMean(Hyperparam):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def get_shape(self, node, annotated):
        # input_shape = annotated[node["meta"]["position"]["component"]]["meta"]["position"]["input_shape"][0]
        input_shape = self.get_input_shape(node)[0]
        return [input_shape[-1], ]


class BatchnormVariance(Hyperparam):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def get_shape(self, node, annotated):
        # input_shape = annotated[node["meta"]["position"]["component"]]["meta"]["position"]["input_shape"][0]
        input_shape = self.get_input_shape(node)[0]
        return [input_shape[-1], ]


class BatchnormOffset(Hyperparam):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.trainable = True

    def get_shape(self, node, annotated):
        # input_shape = annotated[node["meta"]["position"]["component"]]["meta"]["position"]["input_shape"][0]
        input_shape = self.get_input_shape(node)[0]
        return [input_shape[-1], ]


class BatchnormScale(Hyperparam):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.trainable = True

    def get_shape(self, node, annotated):
        # input_shape = annotated[node["meta"]["position"]["component"]]["meta"]["position"]["input_shape"][0]
        input_shape = self.get_input_shape(node)[0]
        return [input_shape[-1], ]


def nodes(node):
    value = node["value"]
    # TODO: group ingredient by block
    nodes_ingredient = {"ingredient": Ingredient}
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
        self.passes = [[self.cache_shape,
                        self.annotate_count]]

    def annotate_count(self, node):
        node = deepcopy(node)
        node["count"] = nodes(node).get_trainable_params_count(node=node,
                                                               annotated=self.annotated)
        return node

    def cache_shape(self, node):
        """Cache the shape info in the node"""
        node = deepcopy(node)
        name = node["name"]

        node["shape"] = nodes(node).get_shape(node,
                                              self.annotated)
        input_nodes = self.get_input_nodes(name, self.annotated)
        if input_nodes is None:
            inputs = None
        else:
            inputs = list(map(lambda x: x["name"], input_nodes))
        node["input_nodes"] = inputs
        node["ancestor"] = core.get_ancestor_ingredient_node(node, self.tree)
        if node["ancestor"]["name"] in self.components:
            node["dtype"] = self.components[node["ancestor"]["name"]]["meta"]["dtype"]
        else:
            node["dtype"] = None
        return node
