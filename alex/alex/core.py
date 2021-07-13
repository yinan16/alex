# ----------------------------------------------------------------------
# Created: lör maj 15 17:21:15 2021 (+0200)
# Last-Updated:
# Filename: core.py
# Author: Yinan Yu
# Description:
# ----------------------------------------------------------------------

import uuid
import collections
from copy import deepcopy
import pydot
import matplotlib.image as mpimg
from io import BytesIO
from pprint import pprint
import json

from alex.alex import const, dsl_parser, util


# FIXME: use typing
def get_value_type(label):
    if label in const.INGREDIENT_TYPES \
       or label in const.INPUT_TYPES \
       or label in const.REGULARIZERS:
        node_type = "ingredient"
    elif label in const.ALL_RECIPES:
        node_type = "recipe"
    else:
        node_type = "hyperparam"
    return node_type


def get_node(tree, name):
    return tree[name]


def get_nodes(tree, names):
    return list(map(lambda x: tree[x] if x in tree else None, names))


def get_subtree(tree, name):
    tree = deepcopy(tree)
    subtree = collections.OrderedDict()
    descendants = tree[name]["descendants"]

    for node in descendants:
        subtree[node] = tree[node]
    subtree[name] = tree[name]
    return subtree


def get_children(root, tree, field=None):
    if field is not None:
        return list(map(lambda x: tree[x][field], tree[root]["children"]))
    else:
        return list(map(lambda x: tree[x], tree[root]["children"]))


def get_ancestor_param_node(node, tree, field=None):
    tree = deepcopy(tree)
    node = deepcopy(node)
    if node["value"] in const.ALL_PARAMS:
        parent_node = node
    elif node["name"] == "root" \
         or node["type"] == "ingredient" \
         or node["type"] == "recipe":
        parent_node = None
    else:
        parent = tree[node["parent"]]
        while parent["value"] not in const.ALL_PARAMS:
            _name = parent["name"]
            parent = tree[tree[_name]["parent"]]
            if parent["value"] == "root":
                parent = None
                break
        parent_node = parent
    if parent_node is None:
        return None
    elif field is None:
        return parent_node
    else:
        return parent_node[field]


def get_ancestor_ingredient_node(node, tree, field=None):
    name = node["meta"]["position"]["component"]
    if field is None:
        return tree[name]
    else:
        return tree[name][field]


def get_parent(root, tree, field=None):
    parent_node = tree[tree[root]["parent"]]
    if field is not None:
        parent_node = parent_node[field]
    return parent_node


def get_descendants(root, tree, field=None):
    if field is not None:
        return list(map(lambda x: tree[x][field], tree[root]["descendants"]))
    else:
        return list(map(lambda x: tree[x], tree[root]["descendants"]))


def get_labels(tree):
    return list(map(lambda x: tree[x]["label"], tree))


def name_to_index(names, tree):
    name_list = list(tree.keys())
    return list(map(lambda x: name_list.index(x), names))


def get_child_by_value(value, root, tree):
    for child in get_children(root, tree):
        if child["value"] == value:
            return child


def get_descendant_args_by_value(value, root, tree):
    for descendant in get_descendants(root, tree):
        if descendant["value"] == value:
            children = get_children(descendant["name"], tree, "value")
            return children


def instantiate_node(name, label, value, meta, parent, children, descendants):
    node = dict()
    node["name"] = name
    node["label"] = label
    node["value"] = value
    node["children"] = deepcopy(children)
    node["descendants"] = deepcopy(descendants)
    node["parent"] = parent
    node["lmd"] = descendants[0] if len(descendants)>=1 else name
    node["type"] = get_value_type(value)
    node["meta"] = meta
    return node


import ast
import traceback


def get_param(params, param):
    try:
        params = ast.literal_eval(params)
    except:
        # print("Error when parsing", params)
        # traceback.print_exc()
        # params is treated as a string literal
        pass
    if not isinstance(params, dict):
        if param == "name":
            return "%s_%s" % (str(params), str(uuid.uuid1()))
        else:
            return params
    if param not in params:
        raise Exception("param %s not recgonized" % str(param))
    return params[param]


def alex_graph_to_json(graph,
                       root_name="root",
                       json_obj=collections.OrderedDict(),
                       naive=True,
                       label_path=None,
                       position: dict = {"inputs": None,
                                         "component": None,
                                         "shape": None,
                                         "input_shape": None}):
    """
    This function converts alex graph into a json object.
    A node can be a recipe, an ingredient or a hyperparameter
    All info of a node is encoded into a json sstring

    """
    if isinstance(graph, dict) and ("visible" in graph) and not graph["visible"]:
        return graph, json_obj

    graph = deepcopy(graph)
    json_obj = collections.OrderedDict(sorted(json_obj.items()))
    if isinstance(graph, dict) \
       and "subgraph" in graph \
       and isinstance(graph["subgraph"], dict): # is a recipe
        _position = {"inputs": graph["inputs"],
                     "component": root_name,
                     "shape": None,
                     "input_shape": None}
        label = deepcopy(graph["type"])
        root_name = {"value": graph["type"],
                     "label": label,
                     "name": root_name,
                     "meta": {"hyperparams": {},
                              "position": _position,
                              "label_path": label_path}}
        root_name = str(root_name)
        json_obj[root_name] = collections.OrderedDict()
        for _name, _graph in graph["subgraph"].items():
            _position = {"inputs": _graph["inputs"],
                         "component": _name,
                         "shape": None,
                         "input_shape": None}
            _json_obj = alex_graph_to_json(_graph,
                                           _name,
                                           collections.OrderedDict(),
                                           naive=naive,
                                           label_path=None,
                                           position=_position)
            json_obj[root_name].update(_json_obj)
    else: # if is an ingredient or hyperparameters
        if isinstance(graph, dict) and "hyperparams" in graph:
            if len(graph["hyperparams"]) != 0: # if is an ingredient with hyperparameter

                if not naive:
                    hyperparam_str = str(uuid.uuid3(uuid.NAMESPACE_DNS,
                                                    json.dumps(graph["hyperparams"],
                                                               sort_keys=True)))
                    label = "%s#%s#" % (graph["type"], hyperparam_str)

                else:
                    label = graph["type"]
                _position = {"inputs": graph["inputs"],
                             "component": root_name,
                             "shape": graph["meta"]["shape"],
                             "input_shape": graph["meta"]["input_shape"]}
                _root_name = root_name
                root_name = {"value": graph["type"],
                             "label": label,
                             "name": root_name,
                             "meta": {"hyperparams": graph["hyperparams"],
                                      "position": _position,
                                      "label_path": label_path,
                                      "block": graph["meta"]["block"]}}

                root_name = str(root_name)
                json_obj[root_name] = collections.OrderedDict()
                for _name in sorted(graph["hyperparams"]):
                    _graph = graph["hyperparams"][_name]
                    _name = "%s/%s" % (_root_name, _name)
                    if "#" in label:
                        label_path = label.split("#")[0] + label.split("#")[2]
                    else:
                        label_path = label
                    _json_obj = alex_graph_to_json(_graph,
                                                   _name,
                                                   collections.OrderedDict(),
                                                   naive=naive,
                                                   label_path=label_path,
                                                   position=_position)
                    json_obj[root_name].update(_json_obj)

            else:
                _position = {"inputs": graph["inputs"],
                             "component": root_name,
                             "shape": graph["meta"]["shape"],
                             "input_shape": graph["meta"]["input_shape"]}

                root_name = {"value": graph["type"],
                             "label": graph["type"],
                             "name": root_name,
                             "meta": {"hyperparams": {},
                                      "position": _position,
                                      "label_path": label_path,
                                      "block": graph["meta"]["block"]}}
                root_name = str(root_name)

                json_obj[root_name] = None

        else: # hyperparam tree
            _root_name = deepcopy(root_name)
            _value = _root_name.split("/")[-1]
            label_path = "%s/%s" % (label_path, _value)
            root_name = {"value": _value,
                         "label": label_path,
                         "name":  _root_name,
                         "meta": {"hyperparams": graph,
                                  "position": position,
                                  "label_path": label_path}}
            root_name = str(root_name)
            if isinstance(graph, dict):

                json_obj[root_name] = collections.OrderedDict()
                _graph = sorted(graph)
                for _name in _graph:
                    __name = "%s/%s" % (_root_name, _name)
                    _json_obj = alex_graph_to_json(graph[_name],
                                                   __name,
                                                   collections.OrderedDict(),
                                                   naive=naive,
                                                   label_path=label_path,
                                                   position=position)
                    json_obj[root_name].update(_json_obj)

            elif isinstance(graph, list):
                json_obj[root_name] = []
                for i, _graph in enumerate(graph):
                    _label = _graph
                    _name = "%s/%s_%i" % (_root_name,
                                          _label, i)
                    label_path = "%s/%s" % (label_path, _label)

                    _json_value = {"value": _label,
                                   "label": label_path,
                                   "name": _name,
                                   "meta": {"hyperparams": graph,
                                            "position": position,
                                            "label_path": label_path}}
                    json_obj[root_name].append(str(_json_value))
            else:
                _label = graph
                _name = "%s/%s" % (_root_name,
                                   _label)
                label_path = "%s/%s" % (label_path, _label)
                _json_value = {"value": _label,
                               "label": label_path,
                               "name": _name,
                               "meta": {"hyperparams": graph,
                                        "position": position,
                                        "label_path": label_path}}

                json_obj[root_name] = str(_json_value)


    return json_obj


def json_to_tree(json,
                 root=None,
                 name_root="root",
                 parent_name=None,
                 tree=collections.OrderedDict(),
                 exclude_types=[]):
    json = deepcopy(json)
    tree = deepcopy(tree)

    children = []
    descendants = []
    if root is not None and get_value_type(get_param(root, "value")) in exclude_types:
        return tree

    if type(json) == dict:
        json = collections.OrderedDict(sorted(json.items()))
    for node in json:
        _json = json[node]
        name = get_param(node, "name")
        if get_value_type(get_param(node, "value")) in exclude_types:
            continue
        _descendants = []
        if isinstance(_json, dict):
            tree = json_to_tree(_json,
                                node,
                                name,
                                name_root,
                                tree,
                                exclude_types=exclude_types)
            _descendants = tree[name]["descendants"]
        else:
            if isinstance(_json, list):
                for _item in _json:
                    _name = get_param(str(_item), "name")
                    _label = get_param(str(_item), "label")
                    _meta = get_param(str(_item), "meta")
                    _value = get_param(str(_item), "value")
                    if get_value_type(_label) not in exclude_types:
                        tree[_name] = instantiate_node(_name,
                                                       _label,
                                                       _value,
                                                       _meta,
                                                       name,
                                                       [], [])
                        _descendants.append(_name)
            else:
                if _json is not None:
                    _name = get_param(str(_json), "name")
                    _label = get_param(str(_json), "label")
                    _meta = get_param(str(_json), "meta")
                    _value = get_param(str(_json), "value")
                    if get_value_type(_label) not in exclude_types:
                        tree[_name] = instantiate_node(_name,
                                                       _label,
                                                       _value,
                                                       _meta,
                                                       name,
                                                       [], [])
                        _descendants = [_name]

            tree[name] = instantiate_node(name,
                                          get_param(node, "label"),
                                          get_param(node, "value"),
                                          get_param(node, "meta"),
                                          name_root,
                                          _descendants,
                                          _descendants)
        if name in tree:
            descendants += _descendants + [name]
            children.append(name)
    if root is not None:
        tree[name_root] = instantiate_node(name_root,
                                           get_param(root, "label"),
                                           get_param(root, "value"),
                                           get_param(root, "meta"),
                                           parent_name,
                                           children,
                                           descendants)
    return tree


def alex_graph_to_tree(network_graph, exclude_types=[], naive=True):
    json_obj = alex_graph_to_json(network_graph, naive=naive)

    tree = json_to_tree(json_obj,
                        exclude_types=exclude_types)
    return tree


def draw(tree, graph_path='example.png', annotation=dict(), dpi=800, size=5, label_field="value", excluded_types=[]):
    graph = pydot.Dot(graph_type='digraph', rankdir='LR', dpi=dpi, size=size, fontsize=12)
    for component in tree:
        name = component
        if tree[component]["type"] in excluded_types:
            continue
        if name not in graph.obj_dict['nodes'].keys():
            if name in annotation:
                color = annotation[name]
            else:
                color = "white"
            if label_field not in tree[component]:
                label = str(tree[component]["label"])
            else:
                label = str(tree[component][label_field])
            node = pydot.Node(name,
                              label=label,
                              shape="box",
                              style='filled',
                              fillcolor=color)
            graph.add_node(node)
        children_list = tree[component]["children"].copy()
        for child in children_list:
            if tree[child]["type"] in excluded_types:
                continue

            if child not in graph.obj_dict['nodes'].keys():
                if child in annotation:
                    color = annotation[child]
                else:
                    color = "white"

                if label_field not in tree[child]:
                    label = str(tree[child]["label"])
                else:
                    label = str(tree[child][label_field])
                node = pydot.Node(child,
                                  label=label,
                                  shape="box",
                                  style='filled',
                                  fillcolor=color)
                # if isinstance(tree[child]["meta"], dict):
                #     if tree[child]["name"] != "root" and "block" in tree[child]["meta"]:
                #         block = tree[child]["meta"]["block"]
                #         blocks[block].add_node(node)
                #     else:
                          # graph.add_node(node)
                graph.add_node(node)

            edge = pydot.Edge(component, child)
            graph.add_edge(edge)

    if graph_path is None:
        png = graph.create_png(prog="dot")
        bio = BytesIO()
        bio.write(png)
        bio.seek(0)
        graph = mpimg.imread(bio)
    else:
        graph.write_png(graph_path)
    return graph
