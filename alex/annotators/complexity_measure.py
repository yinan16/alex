# ----------------------------------------------------------------------
# Created: ons maj 19 00:43:05 2021 (+0200)
# Last-Updated:
# Filename: complexity_measure.py
# Author: Yinan Yu
# Description:
# ----------------------------------------------------------------------
from abc import ABC, abstractmethod
from math import floor
from pprint import pprint
from copy import deepcopy
from alex.alex import core, const, node_interface
from alex.alex.annotator_interface import Annotator
from alex.engine import ns_alex
from math import floor
from pprint import pprint
from copy import deepcopy
from alex.engine import ns_alex
from jsonschema import validate
import collections
import numpy as np
import collections
import os
import numpy as np

# There are two main complexity measure we use here
# the Halstead complexity measure and the Cyclomatic measure
def cyclomatic(e, n, c):
    """
    e: the number of edges
    n: the number of nodes
    c: the connected components
    """
    return e - n + 2*c


def halstead(n_1, n_2, N_1, N_2):
    """
    n_1: the number of unique operators
    n_2: the number of unique operands
    N_1: the total number of operators
    N_2: the total number of operands

    """
    length = N_1 + N_2
    vocabulary = n_1 + n_2
    if N_1 == 0 or N_2 == 0:
        volume = 0
        difficulty = 0
    else:
        volume = round(length * np.log2(vocabulary), 2)
        difficulty = round(n_1/2 * N_2/n_2, 2)
    effort = difficulty * volume
    return {"vocabulary": vocabulary,
            "length": length,
            "volume": volume,
            "difficulty": difficulty,
            "effort": effort,
            "time": effort/18,
            "bugs": effort**(2/3)/3000}


def node_complextiy(node, tree, field):
    name = node["name"]
    descendants = core.get_descendants(name, tree)
    operators = list(map(lambda y: y["value"],
                         filter(lambda x: len(x["children"]) != 0,
                                descendants)))
    operands = list(map(lambda x: x["value"], descendants))
    n_1 = len(set(operators)) + 1
    n_2 = len(set(operands))
    N_1 = len(operators) + 1
    N_2 = len(operands)
    metric = halstead(n_1, n_2, N_1, N_2)
    if field is None:
        return metric
    else:
        return metric[field]


class Ingredient(node_interface.Ingredient):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @staticmethod
    def calculate_complexity(node, tree, field):
        """Halstead"""
        return node_complextiy(node, tree, field=field)


class Recipe(node_interface.Recipe):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def calculate_complexity(self, node, tree, *args, **kwargs):
        """Cyclomatic
        Default is sequential
        """
        if node["value"] != "root":
            n_nodes = self.get_node_count(node)
            n_edges = self.get_edge_count(node)
            # Assuming that all nodes are connected in a recipe
            metric = cyclomatic(n_edges, n_nodes, n_nodes)
        else:
            descendants = core.get_descendants(node["name"], tree, "complexity")
            metric = round(sum(descendants), 2)
        return metric


class Hyperparam(node_interface.Hyperparam):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @staticmethod
    def calculate_complexity(node, tree, field):
        """Halstead"""
        return node_complextiy(node, tree, field=field)


# ------------------------------------------------------------------------ #
def nodes(node):
    value = node["value"]
    _nodes = {"ingredient": Ingredient,
              "hyperparam": Hyperparam,
              "recipe": Recipe}
    if value not in _nodes:
        value = node["type"]
    return _nodes[value](node["value"])


class ComplexityMeasure(Annotator):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.anno_name = "complexity"
        self.passes = [[self.annotate]]

    def annotate(self, node):
        node = deepcopy(node)
        node[self.anno_name] = nodes(node).calculate_complexity(node,
                                                                self.annotated,
                                                                field="volume")
        return node
