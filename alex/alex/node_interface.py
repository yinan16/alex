# Created: mån maj 17 11:46:49 2021 (+0200)
# ----------------------------------------------------------------------
# Last-Updated:
# Filename: node_interface.py
# Author: Yinan Yu
# Description:
# ----------------------------------------------------------------------
from abc import ABC, abstractmethod
from math import floor
import collections
import os
import numpy as np
from pprint import pprint
from copy import deepcopy
from alex.alex import core, const, util, dsl_parser
from alex.engine import ns_alex
from jsonschema import validate


# These classes are only caching static info
class Node:
    @staticmethod
    def get_children_names(node):
        return node["children"]


class Ingredient(Node):

    def __init__(self, ingredient_name):
        self.name = ingredient_name

    def annotate(self):
        pass


class Recipe(Node):
    def __init__(self, recipe_name):
        self.name = recipe_name
        if self.name != "root":
            self.config_path = os.path.join(const.COMPONENT_BASE_PATH,
                                            self.name + ".yml")
            self.recipe = dsl_parser.parse(self.config_path, return_dict=False)
        else:
            self.recipe = None

    def get_node_count(self, node=None):
        if self.recipe is None:
            return len(node["children"])
        else:
            return len(self.recipe)

    def get_edge_count(self, node=None):
        if self.recipe is not None:
            return sum(list(map(lambda x: len(x["meta"][const.INPUTS]),
                                self.recipe)))
        else:
            return len(node["children"]) - 1

    def connection(self):
        """draw connection"""
        pass

    def annotate(self):
        pass


class Hyperparam(Node):

    def __init__(self, hyperparam_name):
        self.name = hyperparam_name
        self.trainable = False

    def is_trainable_params(self):
        pass

    def annotate(self):
        pass
