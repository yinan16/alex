# Created: sön nov  12 10:11:23 2017 (+0100)
# ----------------------------------------------------------------------
# Last-Updated:
# Filename: parser.py
# Author: Yinan
# Description: This module parses block of components (recipes) and basic components (ingredients) from .yml files.
# ----------------------------------------------------------------------


import pydot
from copy import deepcopy
import os
import string
import collections.abc as collections
from collections import OrderedDict, Iterable
from pprint import pprint
import traceback

from typing import Union

from alex.alex import checkpoint, const, util
from jsonschema import validate


strs = list(string.ascii_lowercase)

# FIXME: find a better solution
# These are used to set up names
global global_count0, global_count1, component_count, prev_component
global_count0 = 0
global_count1 = 0
component_count = 0

prev_component = None

# Each keywords in the yml file is a function
"""
repeat: component -> [component]
share: component_name -> source_name -> [var_names]
"""

"""
The parsing process involves several passes in order to generate the final ast
At each pass, an intermediate data structure is generated
"""
Fns1pass = (const.REPEAT, const.NAME, const.INPUTS)
Fns2pass = (const.PARAMS_TRAINING, "share")
Fns = Fns2pass + Fns1pass
Training = (const.TYPE)
Values = (const.HYPERPARAMS, const.VAR, const.STATS, const.TENSOR) # output
Meta = (const.NAME, const.INPUTS, const.TYPE, const.DTYPE, const.SHAPE)


######## parse
def alex_reader(user_defined):

    def _read_inputs(component):
        component = clone(component)
        if const.INPUTS not in component:
            # If no inputs is specified, inputs is set to the previous component (sequential)
            component = {**{const.INPUTS: None}, **component}
        elif not isinstance(component[const.INPUTS], list):
            # Convert all inputs into list
            component[const.INPUTS] = [component[const.INPUTS]]
        return component

    def _read_hyperparams(component):
        component = clone(component)
        hyperparams = dict() if const.HYPERPARAMS not in component else clone(component[const.HYPERPARAMS])
        component_type = component[const.TYPE]
        if component_type in const.PARAMS:
            for param in const.PARAMS[component_type]:
                if param not in hyperparams:
                    hyperparams[param] = dict()
                if const.DTYPE not in hyperparams[param]:
                    hyperparams[param][const.DTYPE] = const.DEFAULT_DTYPE
        component[const.HYPERPARAMS] = hyperparams
        return component

    if "default" not in user_defined:
        default = {"dtype": const.DEFAULT_DTYPE}
    else:
        default = user_defined["default"]

    # Set up default value
    reader = {const.NAME: lambda x: {**{const.NAME: None}, **x} if const.NAME not in x else clone(x),
              const.INPUTS: _read_inputs,
              const.HYPERPARAMS: _read_hyperparams,
              "visible": lambda x: {**{"visible": True}, **x} if "visible" not in x else clone(x),
              "dtype": lambda x: {**{"dtype": default["dtype"]}, **x} if "dtype" not in x else clone(x)}
    return reader


def fill_mandatory(component, reader):
    for fn in reader:
        component = reader[fn](component)
    return component


def clone(component):
    return deepcopy(component)


def recipe_defined(rtype):
    # TODO: temporary. Use types instead
    return rtype not in list(const.DET_COMPONENTS_NO_HYPE.keys()) + [const.DATA,
                                                                 const.LABEL]

# FIXME
def _resolve_hyperparams(hyperparams):
    return hyperparams


def _parse_ingredient_hyperparams(hyperparams):
    hyperparams = clone(hyperparams)
    for hyperparam in hyperparams:
        if isinstance(hyperparams[hyperparam], dict):
            hyperparam_file = os.path.join(const.COMPONENT_BASE_PATH,
                                           hyperparam+".yml")
            if os.path.exists(hyperparam_file):
                _hyperparam = util.read_yaml(hyperparam_file)
                util.fill_dict(_hyperparam, hyperparams[hyperparam])
            _hyperparam = _parse_ingredient_hyperparams(hyperparams[hyperparam])
            hyperparams[hyperparam] = clone(_hyperparam)
    return hyperparams


def _config_to_inter_graph(user_defined, reader, config=None, recipe=None):
    if config is None:
        config = user_defined
    if recipe is None:
        recipe = const.ROOT
    # If configuration is not an ingredient
    if const.PARAMS_NETWORK in config:
        component = dict()
        if const.PARAMS_NETWORK not in user_defined:
            component = fill_mandatory(clone(user_defined), reader)
        component[const.HYPERPARAMS] = []
        component[const.TYPE] = recipe
        for subconfig in config[const.PARAMS_NETWORK]:
            component[const.HYPERPARAMS].append(dict())
            component[const.HYPERPARAMS][-1] = deepcopy(subconfig)
            if recipe_defined(subconfig[const.TYPE]):
                subconfig_def = util.read_yaml(os.path.join(const.COMPONENT_BASE_PATH,
                                                            subconfig[const.TYPE] + const.YAML))
            else:
                subconfig_def = {}
            component[const.HYPERPARAMS][-1] = _config_to_inter_graph(subconfig,
                                                                      reader,
                                                                      subconfig_def,
                                                                      subconfig[const.TYPE])
    else:
        # TODO: make it recursive
        component = fill_mandatory(clone(user_defined), reader)
        hyperparams = clone(config)
        util.fill_dict(hyperparams, component[const.HYPERPARAMS])
        component[const.HYPERPARAMS] = _parse_ingredient_hyperparams(component[const.HYPERPARAMS])
    return component


def _inter_graph_valid(inter_graph):
    # TODO: refine
    # check validity
    return True


def config_to_inter_graph(user_defined, reader):
    graph = _config_to_inter_graph(user_defined, reader, config=None, recipe=None)
    _inter_graph_valid(graph)
    return graph


###############################################################################
def _get_node(component):
    graph = dict()
    graph[const.VALUE] = dict()
    graph[const.META] = dict()
    for v in Values:
        if v in component:
            graph[const.VALUE][v] = clone(component[v])
        else:
            graph[const.VALUE][v] = dict()
    for m in Meta:
        if m in component:
            graph[const.META][m] = component[m]
    graph[const.META][const.NAME] = None
    graph[const.META]["visible"] = component["visible"]
    graph[const.META]["dtype"] = component["dtype"]
    graph[const.META][const.SCOPE] = component[const.SCOPE] if const.SCOPE in component else component[const.TYPE]
    graph[const.VALUE][const.VALUE] = dict()
    # if const.PARAMS_TRAINING in component: # not used for now... maybe for variables
    #     graph[const.VALUE][const.HYPERPARAMS][const.PARAMS_TRAINING] = component[const.PARAMS_TRAINING]
    return graph


def _parse_to_ast(component, args):
    ast = clone(args)
    for fn in Fns1pass:
        if fn in component:
            ast = (fn, component[fn], ast)
    return ast


def inter_graph_to_ast(component):
    component = clone(component)
    if isinstance(component[const.HYPERPARAMS], list): # not atomic
        args = []
        for _component in component[const.HYPERPARAMS]:
            _component[const.SCOPE] = component[const.TYPE]
            args.append(inter_graph_to_ast(_component))
    else:
        args = [_get_node(component)]
    ast = _parse_to_ast(component, args)
    return ast


######## eval
############################ Begin the first pass #######################
def _append(lst, elm):
    if isinstance(elm, list):
        return lst + elm
    else:
        return lst + [elm]


def _get_name(name, append=False):
    global global_count0, global_count1, component_count
    _name = str(component_count) \
        + strs[global_count0 % len(strs)] \
        + strs[global_count1 % len(strs)]
    if name:
        if append:
            _name = "%s_%s"%(name, _name)
        else:
            _name = name
    global_count0 += 1
    global_count1 += 8
    component_count += 1
    return _name


def _mark_repeat_idx(components, idx, times, count):
    components = clone(components)
    for i, component in enumerate(components):
        component[const.META][const.RPT_IDX] = {"block_repeat_idx": idx,
                                                "max_block_repeat": times-1,
                                                "component_idx_in_block": i,
                                                "component_count": count,
                                                "max_components": times*len(components)}
    return components


def eval_repeat(times, components):
    _components = clone(components)
    components = []
    count = 0
    for idx in range(times):
        count += 1
        __components = clone(_components)
        __components = _mark_repeat_idx(__components, idx, times, count)
        components = _append(components, __components)
    return components


def eval_name(name, components):
    components = clone(components)
    top_name = _get_name(name)
    if components[0][const.META][const.SCOPE] == const.ROOT:
        scope = ""
    else:
        scope = "%s_" % components[0][const.META][const.SCOPE]

    for component in components:
        is_atomic = False
        __name = clone(component[const.META][const.NAME])  # name resolved so far; might add prefix (path) to it
        _name = top_name  # suffix, e.g. "xyz123" in conv_xyz123
        if __name is None:  # it is the lowest level - component name has not been resolved yet
            is_atomic = True
            __name = _get_name(name)

        if const.RPT_IDX in component[const.META]:  # repeat
            rpt_info = component[const.META][const.RPT_IDX]
            rpt_idx = rpt_info["block_repeat_idx"]
            max_rpt = rpt_info["max_block_repeat"]
            if rpt_idx != max_rpt:
                if is_atomic:
                    __name = "%s_%i" % (__name, rpt_idx)
                else:
                    _name = "%s_%i" % (top_name, rpt_idx)
        if is_atomic:
            if name is None:  # if no user defined name
                __name = "%s_%s"%(component[const.META][const.TYPE], __name)  # const.TYPE + "_" + "id_string"
        else: # when component is not on the lowest level
            if name is None:
                __name = "%s%s/%s" % (scope, _name, __name)
            else: # user defined name in recipe
                __name = "%s/%s" % (_name, __name)
        component[const.META][const.NAME] = __name
    return components


def eval_inputs(inputs, components):
    global prev_component

    components = clone(components)
    user_defined = True
    if inputs is None:
        user_defined = False
        inputs = prev_component
    _prev_component = inputs

    for i, component in enumerate(components):
        component = clone(component)
        if component[const.META][const.TYPE] == const.DATA or component[const.META][const.TYPE] == "label":
            continue
        if i == 0 and user_defined:
            component[const.META][const.INPUTS] = inputs
        else:
            # If input components are not defined, then make it sequential
            if component[const.META][const.INPUTS] is None:
                component[const.META][const.INPUTS] = [_prev_component]
            else:
                if i != 0:
                    prev_dir = _prev_component.replace(_prev_component.split("/")[-1], "")
                    for _b, _base in enumerate(component[const.META][const.INPUTS]):
                        if isinstance(_base, dict):
                            pass
                        base_name = _base.split("/")[-1]
                        _in = prev_dir + base_name
                        component[const.META][const.INPUTS][_b] = _in

        if const.RPT_IDX in component[const.META]:
            rpt_info = component[const.META][const.RPT_IDX]
            _component_count = rpt_info["component_count"]
            component_idx_in_block = rpt_info["component_idx_in_block"]
            if _component_count != 1 and component_idx_in_block == 0:
                if isinstance(_prev_component, list):
                    component[const.META][const.INPUTS] = _prev_component
                else:
                    component[const.META][const.INPUTS] = [_prev_component]
            component[const.META].pop(const.RPT_IDX, None)
        _prev_component = component[const.META][const.NAME]
        components[i] = component

    prev_component = _prev_component
    return components


############################ The end of the first pass #######################
############################ Begin second pass ###############################
####### Second pass only deal with local dependencies

### TODO: share
def eval_share(share, components):

    # for component in components:
    #     if share is not None:

    #         if component[""][]SHARABLE
    return components


############################ The end of the second pass #######################
def std_env():
    env = {const.REPEAT: eval_repeat,
           const.NAME: eval_name,
           const.INPUTS: eval_inputs,
           "share": lambda t, x: deepcopy(x), # TODO: not implemented
    }
    return env


env = std_env()


def eval_ast_to_graph(ast, env=env):
    ast = clone(ast)
    if isinstance(ast, list):
        graph = []
        for _ast in ast:
            graph = _append(graph, eval_ast_to_graph(_ast, env=env))
    if isinstance(ast, tuple):
        if isinstance(ast[2], list):
            if all(isinstance(x, dict) for x in ast[2]):
                graph = env[ast[0]](ast[1], ast[2])
            else:
                graph = []
                for _ast in ast[2]:
                    _graph = eval_ast_to_graph(_ast, env=env)
                    graph = _append(graph, _graph)
                graph = eval_ast_to_graph((ast[0], ast[1], graph), env=env)
        elif isinstance(ast[2], tuple):
            graph = eval_ast_to_graph(ast[2], env=env)
            graph = eval_ast_to_graph((ast[0], ast[1], graph), env=env)
    return graph

# TODO: merge into the same module as schedulers
# Functions for inputs
def _parse_range(range):
    _range = range.split(":")
    lower = _range[0]
    upper = _range[1]
    if lower != "":
        lower = int(lower)
    if upper != "":
        upper = int(upper)
    return lower, upper


def find_key(data, key):
    if key in data: return data[key]
    found = None
    for _key, value in data.items():
        if _key == "connect_to":
            continue
        if isinstance(value, dict):
            found = find_key(value, key)
        if found is not None:
            break
    return found


def delete_key(data, key):
    if key in data:
        del data[key]

    for _key, value in data.items():
        if isinstance(value, dict):
            delete_key(value, key)


def connect_to(components, current_component_type, current_component_hyperparams=None, range=None, types=None, scope="all"):

    def _get_name_scope(_component):
        return "/".join(_component["meta"]["name"].split("/")[:-1])

    def _get_type(_component):
        return _component["meta"]["type"]

    if scope != "all" and (scope != "current"):
        raise Exception("scope has to be 'current' or 'all'!")
    if current_component_hyperparams is None:
        current_component_hyperparams = components[-1]["value"]["hyperparams"]
    current_component_name_scope = _get_name_scope(components[-1])
    if range is not None:
        lower, upper = _parse_range(range)
    else:
        lower, upper = -2, -1

    connected_component = []
    if lower == "" and upper != "":
        _components = components[:upper]
    elif upper == "" and lower != "":
        _components = components[lower:]
    elif upper != "" and lower != "":
        _components = components[lower:upper]
    else:
        _components = components

    for component in _components:
        type_match = True
        scope_match = True
        _component_type = _get_type(component)
        if types is not None and (_component_type not in types):
            type_match = False
        if scope == "current" and (current_component_name_scope != (_get_name_scope(component))):
            scope_match = False
        if type_match and scope_match:
            connected_component.append("%s/%s" % (component["meta"]["name"],
                                                  types[_component_type]))
            # Make regularization part of the hyperparam subtree
            # component["value"]["hyperparams"][types[_component_type]][current_component_type] = current_component_hyperparams
        delete_key(component["value"]["hyperparams"], "connect_to")
    delete_key(current_component_hyperparams, "connect_to")
    return connected_component


# TODO: implement
def _feature_maps_(range, type):
    pass

def flatten(lst):
    flattened = []
    for obj in lst:
        if not isinstance(obj, list):
            obj = [obj]
        flattened += obj
    return flattened


def global_update(components):
    """
    Here we resolve any functions defined for inputs
    """
    prev_dir = None
    prev_level = 0
    for icomponent, component in enumerate(components):
        _dir = component[const.META][const.NAME].split("/")
        current_level = len(_dir) - 1
        current_dir = component[const.META][const.NAME].replace(_dir[-1], "")
        if current_dir != prev_dir and prev_dir is not None:
            if prev_level <= current_level:
                inputs = component[const.META][const.INPUTS][0]
        if component[const.META][const.INPUTS] is not None:
            need_flatten = False
            for i, _input in enumerate(component[const.META][const.INPUTS]):
                if _input is None:
                    component[const.META][const.INPUTS][i] = const.INPUTS
                elif const.INPUTS in _input:
                    try:
                        component[const.META][const.INPUTS][i] = inputs
                    except: # if inputs is not defined
                        component[const.META][const.INPUTS][i] = const.INPUTS

                elif isinstance(_input, dict):
                    if "connect_to" in _input:
                        component[const.META][const.INPUTS][i] = connect_to(components[:icomponent+1],
                                                                            component[const.META][const.TYPE],
                                                                            **_input["connect_to"])
                    need_flatten = True
            if need_flatten:
                component[const.META][const.INPUTS] = flatten(component[const.META][const.INPUTS])
        component[const.VALUE][const.HYPERPARAMS] = _resolve_hyperparams(component[const.VALUE][const.HYPERPARAMS])
        for _key in component[const.VALUE][const.HYPERPARAMS]:
            hyperparam = component[const.VALUE][const.HYPERPARAMS][_key]
            # TODO: make this nested?
            if isinstance(hyperparam, dict) and ("connect_to" in hyperparam):
                connect_to(components[:icomponent+1],
                           component[const.META][const.TYPE],
                           hyperparam,
                           **hyperparam["connect_to"])
        prev_level = current_level
        prev_dir = current_dir

    return components



########### Params_Training ###############
def _user_fn(fn, kwargs):
    return lambda x: const.USER_FNS[fn](x, **kwargs)


######################################### Different views:
def draw_graph(graph_list, level=2, graph_path='example.png', show="name"):
    graph = pydot.Dot(graph_type='digraph', rankdir='LR', dpi=800, size=5, fontsize=18)

    added = []
    for component in graph_list:
        _dir = component[const.META][const.NAME].split("/")
        _dir = _dir[:level]
        name = "/".join(_dir)
        if name in added:
            continue
        added.append(name)
        node = pydot.Node(name, shape="box")
        if show == "name":
            label = name.split("/")[-1]# .split("_")[0]
        else:
            label = component["meta"]["type"]
        node.set_label(label)
        graph.add_node(node)

        if not component[const.META][const.TYPE] in const.INPUT_TYPES:
            inputs = component[const.META][const.INPUTS]
            for _input in inputs:
                __input = "/".join(_input.split("/")[:level])
                edge = pydot.Edge(_input, name)
                graph.add_edge(edge)

    graph.write_png(graph_path)
    return graph


def _get_dir_name(path):
    _dir = path.split("/")
    scope = path.replace(_dir[-1], "")
    return scope, _dir[-1]


def list_to_graph(components, parent_level=1, parent_scope=""):
    network = dict()
    _subgraph = OrderedDict()
    network[const.LEVEL] = parent_level-1
    network[const.META] = dict()
    # TODO: remove the following info (cached in meta)
    network[const.INPUTS] = None
    network["visible"] = True
    network[const.TYPE] = const.ROOT
    i = 0
    while i < len(components):
        component = components[i]
        hyperparams = component[const.VALUE][const.HYPERPARAMS]
        _dir = component[const.META][const.NAME].split("/")
        level = len(_dir)
        scope, _name = _get_dir_name(component[const.META][const.NAME])
        if scope == "":
            _scope = _name
        else:
            _scope = scope + component[const.META][const.NAME].replace(scope, "").split("/")[0]
        inputs = component[const.META][const.INPUTS]
        if parent_scope!=scope:
            if level>parent_level:
                _components = []
                for _component in components[i:]:
                    __scope, __ = _get_dir_name(_component[const.META][const.NAME])
                    if scope in __scope:
                        _components.append(_component)
                        i += 1
                    else:
                        break
                _network = list_to_graph(_components, level, scope)
                _network[const.INPUTS] = inputs
                _network[const.HYPERPARAMS] = dict() # hyperparams for recipes
                _network[const.TYPE] = component[const.META][const.SCOPE]
                _network["visible"] = component[const.META]["visible"]
                _network[const.META] = component[const.META]
                # if _network[const.META]["type"] in const.COMPONENT_RECIPES:
                #     _network[const.META]["block"] = "model"

                _subgraph[scope[:-1]] = _network
            else:
                break
        else:
            _network = dict()
            _network[const.SUBGRAPH] = component[const.META][const.NAME]
            _network[const.TYPE] = component[const.META][const.TYPE]
            _network["visible"] = component[const.META]["visible"]
            _network[const.META] = component[const.META]
            _network["dtype"] = component[const.META]["dtype"]
            _network[const.INPUTS] = inputs
            _network[const.LEVEL] = level
            _network[const.HYPERPARAMS] = hyperparams

            _subgraph[_scope] = _network
            i += 1

    network[const.SUBGRAPH] = _subgraph
    return network


def draw_hyperparam_tree(graph,
                         graph_path="hypertree.png"):

    def _is_component(node):
        return isinstance(node, dict) and (const.SUBGRAPH in node)

    def _is_recipe(node):
        return isinstance(node[const.SUBGRAPH], OrderedDict)

    def _is_simple(graph):
        return not isinstance(graph, Iterable) or (isinstance(graph, str))

    def _add_node(parent_node, label, name, graph):
        if label in const.COMPONENTS:
            color = colors["COMPONENTS"]
        elif label in const.DET_COMPONENTS_HYPE:
            color = colors["DET_COMPONENTS_HYPE"]
        elif label in const.DET_COMPONENTS_NO_HYPE:
            color = colors["DET_COMPONENTS_NO_HYPE"]
        elif label in const.INFERENCE:
            color = colors["INFERENCE"]
        elif label in const.ALL_TRAINABLE_PARAMS:
            color = colors["TRAINABLE_PARAMS"]
        elif label in const.ALL_OTHER_PARAMS:
            color = colors["NONTRAINABLE_PARAMS"]
        else:
            color = "white"
        node = pydot.Node(name,
                          shape="box",
                          style="filled",
                          fillcolor=color,
                          fontsize=16)
        node.set_label(label)
        graph.add_node(node)
        edge = pydot.Edge(parent_node, name)
        graph.add_edge(edge)
        return graph

    def _draw_hyperparam_tree(_graph,
                              parent_node: str,
                              graph):

        _graph = clone(_graph)
        if _is_simple(_graph):
            graph = _add_node(parent_node=parent_node,
                              label=str(_graph),
                              name="%s/%s" % (parent_node, str(_graph)),
                              graph=graph)
        else:
            for i, _node in enumerate(_graph):
                """ The assumption here is that _graph can be either a dict
                        or hyperparams and hyperparams have a very simple
                        structure, which means that _node is either a string
                        or a scalar. If _graph is a list and _node is a complex
                        data structure, it is not handled.
                """

                if _is_simple(_node):
                    label = str(_node) # node is simple
                else:
                    print(_node)
                    raise Exception("Unrecognized node type")

                name = "%s/%s" % (parent_node, label)
                if isinstance(_graph, list):
                    name = "%s_%i" % (name, i)

                if isinstance(_graph, dict) and _is_component(_graph[_node]):
                    # FIXME:
                    if not _graph[_node]["visible"]:
                        continue
                    label = clone(_graph[_node][const.TYPE])

                graph = _add_node(parent_node, label, name, graph)

                if isinstance(_graph, dict):
                    _node = clone(_graph[_node])
                    if _is_simple(_node):
                        graph = _add_node(parent_node=name,
                                          label=str(_node),
                                          name="%s/%s" % (name, str(_node)),
                                          graph=graph)
                    elif isinstance(_node, list):
                        for ii, __node in enumerate(_node):
                            graph = _add_node(parent_node=name,
                                              label=str(__node),
                                              name="%s/%s_%i" % (name, str(__node), ii),
                                              graph=graph)


                if isinstance(_node, dict):
                    # component or hyperparams
                    if _is_component(_node): # is a component (ingredient or recipe)
                        if _is_recipe(_node):
                            graph = _draw_hyperparam_tree(_node[const.SUBGRAPH],
                                                          parent_node=name,
                                                          graph=graph)
                        # go in the hyperparam tree
                        if recipe_defined(_node[const.TYPE]):
                            graph = _draw_hyperparam_tree(_node[const.HYPERPARAMS],
                                                          parent_node=name,
                                                          graph=graph)
                    else: # in the hyperparam tree
                        graph = _draw_hyperparam_tree(_node,
                                                      parent_node=name,
                                                      graph=graph)

        return graph

    colors = {"COMPONENTS": "red",
              "DET_COMPONENTS_HYPE": "green",
              "DET_COMPONENTS_NO_HYPE": "yellow",
              "INFERENCE": "cyan",
              "NONTRAINABLE_PARAMS": "#8ecae6",
              "TRAINABLE_PARAMS": "#ffafcc"}
    _graph = list_to_graph(graph)
    dot_graph = pydot.Dot(graph_type='digraph', rankdir='LR', dpi=100, size=100, fontsize=18)
    parent_node = const.ROOT
    dot_graph = _draw_hyperparam_tree(_graph[const.SUBGRAPH], parent_node, dot_graph)
    dot_graph.write(graph_path, format="png")


def list_to_dict(components):
    components = deepcopy(components)
    components = OrderedDict(map(lambda x:
                                 (components[x]["meta"]["name"],
                                  components[x]), range(len(components))))

    return components


### d3
### FIXME: test this
def graph_to_d3_ast(graph):
    graph = clone(graph)
    children = []
    for node_name in graph:
        _node = graph[node_name]
        node = dict()
        node["name"] = node_name

        if isinstance(_node, dict) and "level" in _node:
            node["level"] = _node["level"]
            node["type"] = _node["type"]
            if isinstance(_node["subgraph"], dict):
                node["children"] = graph_to_d3_ast(_node["subgraph"])
            if len(_node["hyperparams"]) != 0:
                node["children"] = graph_to_d3_ast(_node["hyperparams"])
        elif isinstance(_node, dict): # is part of hyperparams
            node["type"] = "hyperparams"
            node["children"] = graph_to_d3_ast(_node)
        elif isinstance(_node, list):
            if "type" not in node:
                node["type"] = "hyperparams"
            node["children"] = []
            for __node in _node:
                node["children"].append({"name": __node, "type": "hyperparams"})
        elif _node is None:
            if "type" not in node:
                node["type"] = "hyperparams"
            node["children"] = [{"name": "none", "type": "hyperparams"}]
        else:
            node["children"] = [{"name": _node, "type": "hyperparams"}]
            if "type" not in node:
                node["type"] = "hyperparams"
        children.append(node)
    return children


def annotate(components):
    components = list_to_dict(components)
    for name in components:
        component = components[name]
        try:
            inputs = component["meta"]["inputs"]
            # Annotate block: data, model, loss, optimizer
            if component["meta"]["type"] in const.INPUT_TYPES:
                component["meta"]["block"] = "data"
            elif component["meta"]["type"] not in const.OPTIMIZER_BLOCK \
                 and (component["meta"]["type"] in const.LOSS_BLOCK \
                      or len(list(filter(lambda x: components[x]["meta"]["block"] == "loss",
                                         inputs)))!=0):
                component["meta"]["block"] = "loss"
            elif component["meta"]["type"] in const.OPTIMIZER_BLOCK \
                 or len(list(filter(lambda x: components[x]["meta"]["block"] == "optimizer",
                                    inputs)))!=0:
                component["meta"]["block"] = "optimizer"
            else:
                component["meta"]["block"] = "model"
        except Exception:
            print("------------- Component %s annotation failed ----------" % (name))
            traceback.print_exc()
            raise
    return list(components.values())


def validate_components(components):
    schemas = dict()
    for component in components:
        component_type = component["meta"]["type"]
        if component_type not in schemas:
            schema_path = os.path.join(const.COMPONENT_BASE_PATH,
                                       "." + component_type + ".yml")
            try:
                schemas[component_type] = util.read_yaml(schema_path)["definitions"][component_type]
            except Exception:
                schemas[component_type] = None
                # TODO:
                # print("%s schema not implemented" % component_type)
        if schemas[component_type] is not None:
            validate(instance=component["value"]["hyperparams"], schema=schemas[component_type])

def config_to_type(yml_file):
    return yml_file.split("/")[-1].split(".")[0]


################################################################################
# Public API
################################################################################
def parse(yml_file, return_dict=False):
    global global_count0, global_count1, component_count, prev_component
    global_count0 = 0
    global_count1 = 0
    component_count = 0
    prev_component = None
    try:
        # Read the yml file
        user_defined = util.read_yaml(yml_file)
        reader = alex_reader(user_defined)
        # Step 1: transform yml into a (recursive) dict
        # Each node in the dict has the following keys:
        # hyperparams: list or dict, default: {}
        # type: string
        # inputs: list, default: None
        # name: string, default: None
        # (optional) dtype: data type
        # (optional) shape: list
        _graph:dict = config_to_inter_graph(user_defined, reader)
        # Step 2: transform _graph into an intermediate ast, where each key is a function
        # name
        # inputs
        # repeat
        # e.g. ("inputs", inputs_list, ("name", name_str, ("repeat", n, [])))
        ast:list = inter_graph_to_ast(_graph)

        # Step 3: first pass
        # A list of dicts
        # The reason of having a list of dicts as the main underlying data structure is due to its simplicity. This data structure can be transformed into the final ast
        # Each dict has the following structure:
        # "meta": {"inputs": None or list,
        #          "name": string (scope/.../name),
        #          "scope": the scope of the node,
        #          "type": type of the deep learning component}
        # "value": {"hyperparams": dict of hyperparameters,
        #           "stats": {}
        #           "tensor": library specific tensor object (not easily serializable),
        #           "value": a serializable object that contains the same info as tensor,
        #           "var": }
        graph:list = eval_ast_to_graph(ast)
        # Step 4: second pass: same data structure; populate inputs
        graph:list = global_update(graph)
        if config_to_type(yml_file) not in const.ALL_RECIPES:
            graph = annotate(graph)
        validate_components(graph) # json schema
        # Step 5: load graph and states from checkpoint; check if the structure matches the one defined in the DSL
    except Exception:
        message = "-------------- Error during parsing configuration %s ----------" % yml_file
        traceback.print_exc()
        raise
    if return_dict:
        graph = list_to_dict(graph)
    return graph


def load(config_yml, ckpt_name=None, checkpoint_dir=const.CHECKPOINT_BASE_PATH):
    graph = parse(config_yml)
    graph, states = checkpoint.load(graph, checkpoint_dir, ckpt_name)
    return graph, states


## Util
def dict_to_arg_str(d):
    out = ""
    for key in d:
        if isinstance(d[key], str):
            val = "'%s'"%str(d[key])
        else:
            val = str(d[key])
        out += "%s=%s" % (str(key), val)
    return out


def make_graph_from_yml(yml_file, png_file, level, show="label"):
    graph = parse(yml_file)
    draw_graph(graph, level, png_file, show=show)


def make_ast_from_yml(yml_file, png_file):
    graph = parse(yml_file)
    draw_hyperparam_tree(graph,
                         graph_path=png_file)
