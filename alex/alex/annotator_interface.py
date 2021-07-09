# ----------------------------------------------------------------------
# Created: lör maj 15 18:31:09 2021 (+0200)
# Last-Updated:
# Filename: annotator_interface.py
# Author: Yinan Yu
# Description:
# ----------------------------------------------------------------------
from abc import ABC, abstractmethod
import collections
from pprint import pprint
from copy import deepcopy
from alex.alex import core, dsl_parser


class Annotator(ABC):

    def __init__(self, config_path, exclude_types=[], naive=True):
        if isinstance(config_path, str):
            self.config_path = config_path
            self.components_list = dsl_parser.parse(self.config_path)
        elif isinstance(config_path, list):
            self.components_list = config_path
        self.tree = core.alex_graph_to_tree(dsl_parser.list_to_graph(self.components_list),
                                            exclude_types=exclude_types,
                                            naive=naive)
        self.components = dsl_parser.list_to_dict(self.components_list)

        self.passes = [None]
        self.anno_name = "unnamed annotation"

    def get_input_names(self, name, tree):
        node = tree[name]
        if name == "root":
            return None
        elif node["type"] == "recipe":
            recipe_name = node["name"]
            first_ingredient_in_recipe = list(filter(lambda x: recipe_name in x,
                                                     self.components))[0]
            inputs = self.components[first_ingredient_in_recipe]["meta"]["inputs"]
        elif name in self.components:
            inputs = self.components[name]["meta"]["inputs"]
        else:
            parent = core.get_ancestor_ingredient_node(node, tree, "name")
            inputs = self.components[parent]["meta"]["inputs"]
        return inputs

    def get_input_nodes(self, name, tree):
        inputs = self.get_input_names(name, tree)
        if inputs is not None:
            inputs = core.get_nodes(tree, inputs)
        return inputs

    def annotate_tree(self):

        self.annotated = deepcopy(self.tree)

        for ipass, operation in enumerate(self.passes):
            if operation is None:
                raise Exception("Register an operation in each pass")
            if not isinstance(operation, list):
                operation = [operation]
            for sub_op in operation:
                for name in self.annotated:
                    node = self.annotated[name]
                    self.annotated[name] = sub_op(node)
        #     print("Annotation pass %i done" % ipass)

        print("Annotation %s done" % self.anno_name)

        return self.annotated
