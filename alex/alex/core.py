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

from alex.alex import const, dsl_parser


def get_input_components(component):
    return component[const.META][const.INPUT_COMPONENT]


# FIXME: use typing
def get_value_type(label):
    if label in const.INGREDIENT_TYPES or label in const.INPUTS:
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


def get_ancestor_ingredient_node(node, components, tree, field=None):
    tree = deepcopy(tree)
    components = deepcopy(components)
    node = deepcopy(node)
    if node["name"] in components \
       or node["name"] == "root" \
       or node["type"] == "recipe":
        parent_node = node
    else:
        parent = node["parent"]
        while parent not in components:
            _name = parent
            parent = tree[_name]["parent"]
        parent_node = tree[parent]
    if field is None:
        return parent_node
    else:
        return parent_node[field]


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
    node["type"] = get_value_type(label)
    node["meta"] = meta
    return node


def prepend_component_name(data, name):
    """
    Preppend the component name to hyperparams
    """
    data = deepcopy(data)
    new_data = deepcopy(data)
    for k, v in data.items():
        new_data["%s_%s" % (name, str(k))] = v
        new_data.pop(k, None)
        if isinstance(v, dict):
            new_data["%s_%s" % (name, str(k))], name = prepend_component_name(v, name)
        elif isinstance(v, list):
            new_data["%s_%s" % (name, str(k))] = []
            for i, _v in enumerate(v):
                new_data["%s_%s" % (name, str(k))].append("%s_%i_%s" % (name, i, str(_v)))
        else:
            new_data["%s_%s" % (name, str(k))] = "%s_%s" % (name, str(v))
    return new_data, name


import ast
def get_param(params, param):
    params = ast.literal_eval(params)
    if isinstance(params, str):
        if param == "name":
            return "%s_%s" % (params, str(uuid.uuid1()))
        else:
            return params
    if param not in params:
        raise Exception("param %s not recgonized" % str(param))
    return params[param]


def alex_graph_to_json(graph,
                       root_name="root",
                       json_obj=collections.OrderedDict(),
                       naive=True,
                       position: dict = {"input_component": None,
                                         "component": None,
                                         "input_shape": None}):
    """
    This function converts alex graph into a json object.
    A node can be a recipe, an ingredient or a hyperparameter
    All info of a node is encoded into a json sstring

    """
    if isinstance(graph, dict) and ("visible" in graph) and not graph["visible"]:
        return graph, json_obj

    graph = deepcopy(graph)
    json_obj = deepcopy(json_obj)
    if isinstance(graph, dict) and "subgraph" in graph and isinstance(graph["subgraph"], dict): # is recipe
        _position = {"input_component": graph["input_component"],
                     "component": root_name,
                     "input_shape": None}
        root_name = {"value": graph["type"],
                     "label": graph["type"],
                     "name": root_name,
                     "meta": {"hyperparams": {},
                              "position": _position}}
        root_name = str(root_name)
        json_obj[root_name] = collections.OrderedDict()
        for _name, _graph in graph["subgraph"].items():
            _position = {"input_component": _graph["input_component"],
                         "component": _name,
                         "input_shape": None}
            graph, _json_obj = alex_graph_to_json(_graph,
                                                  _name,
                                                  collections.OrderedDict(),
                                                  naive=naive,
                                                  position=_position)
            json_obj[root_name].update(_json_obj)
    else: # if is ingredient or hyperparameters
        if isinstance(graph, dict) and "hyperparams" in graph:
            if len(graph["hyperparams"]) != 0: # if is ingredient with hyperparameter
                hyperparam_str = str(graph["hyperparams"])
                if not naive:
                    # hyperparam_str = str(uuid.uuid3(uuid.NAMESPACE_DNS, json.dumps(graph["hyperparams"], sort_keys=True)))
                    label = "%s_uuid_%s" % (graph["type"], hyperparam_str)
                else:
                    label = graph["type"]

                _position = {"input_component": graph["input_component"],
                             "component": root_name,
                             "input_shape": None}
                root_name = {"value": graph["type"],
                             "label": label,
                             "name": root_name,
                             "meta": {"hyperparams": graph["hyperparams"],
                                      "position": _position}}
                root_name = str(root_name)
                json_obj[root_name] = collections.OrderedDict()

                for _name, _graph in graph["hyperparams"].items():
                    graph, _json_obj = alex_graph_to_json(_graph,
                                                          _name,
                                                          collections.OrderedDict(),
                                                          naive=naive,
                                                          position=_position)
                    json_obj[root_name].update(_json_obj)

            else:
                _position = {"input_component": graph["input_component"],
                             "component": root_name,
                             "input_shape": None}
                root_name = {"value": graph["type"],
                             "label": graph["type"],
                             "name": root_name,
                             "meta": {"hyperparams": {},
                                      "position": _position}}
                root_name = str(root_name)

                json_obj[root_name] = None

        else: # hyperparam tree
            _root_name = deepcopy(root_name)
            root_name = {"value": _root_name.split("/")[-1],
                         "label": _root_name,
                         "name": "%s/%s" % (position["component"], _root_name),
                         "meta": {"hyperparams": graph,
                                  "position": position}}
            root_name = str(root_name)
            if isinstance(graph, dict):

                json_obj[root_name] = collections.OrderedDict()
                for _name, _graph in graph.items():
                    _name = "%s/%s" % (_root_name, _name)
                    graph, _json_obj = alex_graph_to_json(_graph,
                                                          _name,
                                                          collections.OrderedDict(),
                                                          naive=naive,
                                                          position=position)
                    json_obj[root_name].update(_json_obj)

            elif isinstance(graph, list):
                json_obj[root_name] = []
                for _graph in graph:
                    _label = str(_graph)
                    _json_value = {"value": _label,
                                   "label": _label,
                                   "name": "%s/%s" % (position["component"], root_name),
                                   "meta": {"hyperparams": graph,
                                            "position": position}}
                    json_obj[root_name].append(str(_json_value))
            else:
                _label = str(graph)
                _json_value = {"value": _label,
                               "label": _label,
                               "name": "%s/%s" % (position["component"], root_name),
                               "meta": {"hyperparams": graph,
                                        "position": position}}

                json_obj[root_name] = str(_json_value)


    return graph, json_obj


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
        items = sorted(json.items())
    elif type(json) == collections.OrderedDict:
        items = json.items()
    for node, _json in items:
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
    _, json_obj = alex_graph_to_json(network_graph, naive=naive)

    tree = json_to_tree(json_obj,
                        exclude_types=exclude_types)
    return tree


def draw(tree, graph_path='example.png', annotation=dict(), dpi=800, size=5, label_field="label", excluded_types=[]):
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
            node = pydot.Node(name, shape="box", style = 'filled', fillcolor=color)

            if label_field not in tree[component]:
                label = str(tree[component]["label"]).split("_uuid_")[0]
            else:
                label = str(tree[component][label_field]).split("_uuid_")[0]
            node.set_label(label)
            graph.add_node(node)
        children_list = tree[component]["children"].copy()
        children_list.reverse()
        for child in children_list:
            if tree[child]["type"] in excluded_types:
                continue

            if child not in graph.obj_dict['nodes'].keys():
                if child in annotation:
                    color = annotation[child]
                else:
                    color = "white"
                node = pydot.Node(child, shape="box", style = 'filled', fillcolor=color)
                if label_field not in tree[child]:
                    label = str(tree[child]["label"]).split("_uuid_")[0]
                else:
                    label = str(tree[child][label_field]).split("_uuid_")[0]
                node.set_label(label)

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
