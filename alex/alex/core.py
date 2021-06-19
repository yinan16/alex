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


def instantiate_node(name, label, value, parent, children, descendants):
    node = dict()
    node["name"] = name
    node["label"] = label
    node["value"] = value
    node["children"] = deepcopy(children)
    node["descendants"] = deepcopy(descendants)
    node["parent"] = parent
    node["lmd"] = descendants[0] if len(descendants)>=1 else name
    node["type"] = get_value_type(label)
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


# TODO: consider remove list from hyperparam trees
def alex_graph_to_json(graph,
                       root_name="root",
                       json_obj=collections.OrderedDict(),
                       naive=True):

    if not graph["visible"]:
        return graph, json_obj

    graph = deepcopy(graph)
    json_obj = deepcopy(json_obj)
    if "hyperparams" in graph and len(graph["hyperparams"])!=0: # if is ingredient
        if not naive:
            root_name = "%s//%s_uuid_%s" % (graph["type"], str(uuid.uuid3(uuid.NAMESPACE_DNS, json.dumps(graph["hyperparams"], sort_keys=True))), root_name)
        else:
            root_name = "%s//%s" % (graph["type"], root_name)
    else:
        root_name = "%s//%s" % (graph["type"], root_name)

    json_obj[root_name] = collections.OrderedDict()

    if isinstance(graph["subgraph"], dict): # is recipe
        for _name, _graph in graph["subgraph"].items():
            graph, _json_obj = alex_graph_to_json(_graph,
                                                  _name,
                                                  collections.OrderedDict(),
                                                  naive=naive)
            json_obj[root_name].update(_json_obj)
    else: # is ingredient
        if len(graph["hyperparams"])!=0:
            json_obj[root_name] = graph["hyperparams"]
        else:
            json_obj[root_name] = None
    return graph, json_obj


def json_to_tree(json,
                 root=None,
                 name_root="root",
                 parent_name=None,
                 tree=collections.OrderedDict(),
                 get_name=lambda x: "%s_%s" % (x, str(uuid.uuid1())),
                 get_label=lambda x: x,
                 get_value=lambda x: x,
                 exclude_types=[]):
    json = deepcopy(json)
    tree = deepcopy(tree)

    children = []
    descendants = []
    if root is not None and get_value_type(get_value(root)) in exclude_types:
        return tree

    if type(json) == dict:
        items = sorted(json.items())
    elif type(json) == collections.OrderedDict:
        items = json.items()
    for node, _json in items:
        name = get_name(node)
        if get_value_type(get_value(node)) in exclude_types:
            continue
        _descendants = []
        if isinstance(_json, dict):
            tree = json_to_tree(_json,
                                node,
                                name,
                                name_root,
                                tree,
                                get_name=get_name,
                                get_label=get_label,
                                get_value=get_value,
                                exclude_types=exclude_types)
            _descendants = tree[name]["descendants"]
        else:
            if isinstance(_json, list):
                for _item in _json:
                    _name = get_name(str(_item))
                    _label = str(_item)
                    if get_value_type(_label) not in exclude_types:
                        tree[_name] = instantiate_node(_name,
                                                       _label,
                                                       _item,
                                                       name,
                                                       [], [])
                        _descendants.append(_name)
            else:
                if _json is not None:
                    _name = get_name(str(_json))
                    _descendants = [_name]
                    _label = str(_json)
                    if get_value_type(_label) not in exclude_types:
                        tree[_name] = instantiate_node(_name,
                                                       _label,
                                                       _json,
                                                       name,
                                                       [], [])
            tree[name] = instantiate_node(name,
                                          get_label(node),
                                          get_value(node),
                                          name_root,
                                          _descendants,
                                          _descendants)
        if name in tree:
            descendants += _descendants + [name]
            children.append(name)
    if root is not None:
        tree[name_root] = instantiate_node(name_root,
                                           get_label(root),
                                           get_value(root),
                                           parent_name,
                                           children,
                                           descendants)
    return tree


def alex_graph_to_tree(network_graph, exclude_types=[], naive=True):
    _, json_obj = alex_graph_to_json(network_graph, naive=naive)
    _get_name = lambda x: x.split("//")[1] if "//" in x else "%s_%s" % (x, str(uuid.uuid1()))
    _get_value = lambda x: x.split("//")[0] if "//" in x else str(x)
    if naive:
        _get_label = _get_value
    else:
        def _get_label(x):
            if "//" in x:
                _type = x.split("//")[0]
                _name = x.split("//")[1]
                if "_uuid_" in _name:
                    label =  "%s_uuid_%s" % (_type, _name.split("_uuid_")[0])
                else:
                    label = _type
            else:
                label = str(x)
            return label

    tree = json_to_tree(json_obj,
                        get_name = _get_name,
                        get_label = _get_label,
                        get_value = _get_value,
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
