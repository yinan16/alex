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

from alex.alex import checkpoint, const, util, validation, registry
from alex.alex.logger import logger


strs = list(string.ascii_lowercase)

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
        if component_type in registry.PARAMS:
            for param in registry.PARAMS[component_type]:
                if param not in hyperparams:
                    hyperparams[param] = dict()
                if const.DTYPE not in hyperparams[param]:
                    hyperparams[param][const.DTYPE] = registry.DEFAULT_DTYPE
        component[const.HYPERPARAMS] = hyperparams
        return component

    if "default" not in user_defined:
        default = {"dtype": registry.DEFAULT_DTYPE}
    else:
        default = user_defined["default"]

    # Set up default value
    reader = {const.NAME: lambda x: {**{const.NAME: None}, **x} if const.NAME not in x else clone(x),
              const.INPUTS: _read_inputs,
              const.HYPERPARAMS: _read_hyperparams,
              "visible": lambda x: {**{"visible": True}, **x} if "visible" not in x else clone(x),
              "trainable": lambda x: {**{"trainable": None}, **x} if "trainable" not in x else clone(x),
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
    return rtype not in list(registry.STATELESS_INGREDIENTS_WITHOUT_HYPE.keys()) + [const.DATA,
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


def _config_to_inter_graph(user_defined, reader, block, parent_type="root", config=None, recipe=None, inputs=None):
    if config is None and "params_network" not in user_defined:
        config = user_defined
        recipe = const.ROOT
        component = dict()
        component[const.TYPE] = recipe
        component[const.HYPERPARAMS] = []
        for block in user_defined:
            _user_defined = {"type": block,
                             "hyperparams": deepcopy(user_defined[block])}
            _config = {"params_network": deepcopy(user_defined[block])}
            component[const.HYPERPARAMS].append(_config_to_inter_graph(_user_defined,
                                                                       reader,
                                                                       block,
                                                                       parent_type,
                                                                       config=_config,
                                                                       recipe=block))

    else:
        if config is None:
            config = deepcopy(user_defined)
        # If configuration is not an ingredient
        if const.PARAMS_NETWORK in config:
            component = dict()
            component[const.TYPE] = recipe
            if const.PARAMS_NETWORK not in user_defined:
                component = fill_mandatory(clone(user_defined), reader)

            component[const.HYPERPARAMS] = []
            component["block"] = block
            component[const.SCOPE] = parent_type
            for subconfig in config[const.PARAMS_NETWORK]:
                if recipe_defined(subconfig[const.TYPE]):
                    subconfig_def = util.read_yaml(os.path.join(const.COMPONENT_BASE_PATH,
                                                                subconfig[const.TYPE] + const.YAML))
                else:
                    subconfig_def = {}

                if "inputs" in subconfig:
                    _inputs = subconfig["inputs"]
                else:
                    _inputs = None
                component[const.HYPERPARAMS].append(_config_to_inter_graph(subconfig,
                                                                           reader,
                                                                           block,
                                                                           recipe,
                                                                           subconfig_def,
                                                                           subconfig[const.TYPE], _inputs))
            if inputs is not None:
                component[const.HYPERPARAMS][0]["inputs"] = inputs
        else:
            component = fill_mandatory(clone(user_defined), reader)
            hyperparams = clone(config)
            component["block"] = block
            util.fill_dict(hyperparams, component[const.HYPERPARAMS])
            component[const.HYPERPARAMS] = _parse_ingredient_hyperparams(component[const.HYPERPARAMS])
            component["scope"] = parent_type
    return component


def _inter_graph_valid(inter_graph):
    # TODO: refine
    # check validity
    return True


def config_to_inter_graph(user_defined, reader):
    graph = _config_to_inter_graph(user_defined,
                                   reader,
                                   block="root",
                                   config=None,
                                   recipe=None)
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
    graph[const.META]["trainable"] = component["trainable"]
    graph[const.META]["block"] = component["block"]
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


def _is_block(scope):
    return scope in registry.BLOCKS


def eval_name(name, components):
    components = clone(components)
    top_name = _get_name(name)
    __scope = components[0][const.META][const.SCOPE]
    __type =  components[0][const.META][const.TYPE]
    for component in components:
        if "dir" in component[const.META] and component[const.META]["dir"] != "":
            continue
        __scope = component[const.META][const.SCOPE]
        break
    if __scope == const.ROOT:
        scope = ""
    else:
        scope = __scope
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
            if _is_block(scope):
                name_cache = __name
                __name = "%s/%s" % (scope, __name)
            else:
                if name is None:
                    __name = "%s_%s/%s" % (scope, _name, __name)
                else:  # user defined name in recipe
                    __name = "%s/%s" % (_name, __name)
        component[const.META][const.NAME] = __name
        component[const.META]["dir"] = "/".join(__name.split("/")[:-1])

        if "recipes" not in component[const.META]:
            component[const.META]["recipes"] = []
        component[const.META]["recipes"].append(scope)
        if len(component[const.META]["recipes"]) > len(__name.split("/")[:-1]):
            component[const.META]["recipes"] = component[const.META]["recipes"][1:]
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
                            continue
                        # TO_BE_TESTED
                        # current_scope = component[const.META]["scope"]
                        # if current_scope in registry.BLOCKS:
                        #     if "/" not in _base:
                        #         _in = "%s/%s" % (current_scope, _base)
                        if _base.split("/")[0] in registry.BLOCKS:
                            _in = _base
                        else:
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
        logger.error("Scope has to be 'current' or 'all'!")
        raise Exception
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
    return lambda x: registry.USER_FNS[fn](x, **kwargs)


######################################### Different views:
def draw_graph(graph_list, level=2, graph_path='example.png', show="name"):
    graph = pydot.Dot(graph_type='digraph', rankdir='LR', dpi=800, size=5, fontsize=18)
    prev = None
    for component in graph_list:
        _dir = component[const.META][const.NAME].split("/")
        _dir = _dir[:level]
        name = "/".join(_dir)
        if prev is None:
            prev = name

        if name == prev:
            continue
        else:
            prev = name
        component_scope = component[const.META][const.SCOPE]
        component_type = component[const.META][const.TYPE]
        node = pydot.Node(name, shape="box")
        if show == "name":
            label = name.split("/")[-1]# .split("_")[0]
        else:
            if show not in component["meta"] or show=="type":
                logger.error("Lable '%s' is not available" % show)
                raise Exception
            label = str(component["meta"][show])
        node.set_label(label)
        graph.add_node(node)

        _type = component[const.META][const.TYPE]
        if not _type in registry.DATA_BLOCK:
            inputs = component[const.META][const.INPUTS]
            for _input in inputs:
                if _type not in registry.REGULARIZERS:
                    _input = "/".join(_input.split("/")[:level])
                _edge = pydot.Edge(_input, name)
                graph.add_edge(_edge)

    graph.write_png(graph_path)
    return graph


def _get_dir_name(path, level):
    _dir = path.split("/")
    scope = path.replace("/%s"%_dir[-1], "")
    scope = "/".join(scope.split("/")[0:level])
    return scope, _dir[-1]


def _register_component(component, level):
    subgraph = dict()
    path = component[const.META][const.NAME]
    subgraph[const.SUBGRAPH] = component[const.META][const.NAME]
    subgraph[const.TYPE] = component[const.META][const.TYPE]
    subgraph[const.META] = component[const.META]
    subgraph[const.LEVEL] = level
    subgraph[const.HYPERPARAMS] = component[const.VALUE][const.HYPERPARAMS]
    return subgraph


def _register_graph(level, node_type="root", meta={"inputs": None,
                                                   "name": "root"}):
    graph = dict()
    graph[const.SUBGRAPH] = OrderedDict()
    graph[const.LEVEL] = level
    graph[const.HYPERPARAMS] = dict()
    graph[const.TYPE] = node_type
    graph[const.META] = meta
    return graph


def list_to_graph(components):
    graph = _register_graph(0)
    i = 0

    for component in components:
        path = component[const.META][const.NAME]
        _dir = path.split("/")
        levels = len(_dir)
        _graph = graph
        for _level in range(1, levels+1):
            scope = "/".join(_dir[0:_level])
            if _level == levels: # is leaf
                _graph["subgraph"][scope] = _register_component(component, _level)
            elif scope not in _graph["subgraph"]:
                _graph["subgraph"][scope] = _register_graph(_level,
                                                            component["meta"]["recipes"][levels-_level-1],
                                                            component["meta"])
            _graph = _graph["subgraph"][scope]
    return graph


def draw_hyperparam_tree(graph,
                         graph_path="hypertree.png"):

    def _is_component(node):
        return isinstance(node, dict) and (const.SUBGRAPH in node)

    def _is_recipe(node):
        return isinstance(node[const.SUBGRAPH], OrderedDict)

    def _is_simple(graph):
        return not isinstance(graph, Iterable) or (isinstance(graph, str))

    def _add_node(parent_node, label, name, graph):
        if label in registry.STATEFUL_INGREDIENTS:
            color = colors["STATEFUL_INGREDIENTS"]
        elif label in registry.STATELESS_INGREDIENTS_WITH_HYPE:
            color = colors["STATELESS_INGREDIENTS_WITH_HYPE"]
        elif label in registry.STATELESS_INGREDIENTS_WITHOUT_HYPE:
            color = colors["STATELESS_INGREDIENTS_WITHOUT_HYPE"]
        elif label in registry.INFERENCES:
            color = colors["INFERENCE"]
        elif label in registry.ALL_TRAINABLE_PARAMS:
            color = colors["TRAINABLE_PARAMS"]
        elif label in registry.ALL_OTHER_PARAMS:
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
                    pprint(_node)
                    logger.error("Unrecognized node type")
                    raise Exception

                name = "%s/%s" % (parent_node, label)
                if isinstance(_graph, list):
                    name = "%s_%i" % (name, i)

                if isinstance(_graph, dict) and _is_component(_graph[_node]):
                    # FIXME:
                    if not _graph[_node]["meta"]["visible"]:
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

    colors = {"STATEFUL_INGREDIENTS": "red",
              "STATELESS_INGREDIENTS_WITH_HYPE": "green",
              "STATELESS_INGREDIENTS_WITHOUT_HYPE": "yellow",
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


def annotate(components, lazy=True):
    schemas = dict()
    components = list_to_dict(components)
    for name in components:
        component = components[name]
        component_type = component["meta"]["type"]
        validator = validation.get(component)
        try:
            if component_type not in schemas:
                schemas[component_type] = validator.load_schema(component_type)
            if schemas[component_type] is not None:
                validator.validate_schema(component, schemas[component_type])

            component["meta"]["input_shape"] = validator.get_input_shape(component, components)
            component["meta"]["shape"] = validator.get_shape(component, components)
            if not lazy:
                validator.eval_input_channels(component, components)
                validator.validate_connection(component, components)

            inputs = component["meta"]["inputs"]
        except Exception:
            logger.error("Component %s annotation failed" % (name))
            raise Exception
        components[name] = component
    return list(components.values())


def config_to_type(yml_file):
    return yml_file.split("/")[-1].split(".")[0]


################################################################################
# Public API
################################################################################
def parse(yml_file, return_dict=False, lazy=True):
    global global_count0, global_count1, component_count, prev_component
    global_count0 = 0
    global_count1 = 0
    component_count = 0
    # print("parsing, yml_file", yml_file)
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
        # if config_to_type(yml_file) not in const.RECIPE_TYPES:
        graph = annotate(graph, lazy=lazy) # json schema
        # Step 5: load graph and states from checkpoint; check if the structure matches the one defined in the DSL
    except Exception:
        message = "Error during parsing configuration %s" % yml_file
        logger.error(message)
        raise Exception
    if return_dict:
        graph = list_to_dict(graph)
    return graph


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


def make_graph_from_yml(yml_file, png_file, level, show="name", lazy=True):
    graph = parse(yml_file, return_dict=False, lazy=lazy)
    draw_graph(graph, level, png_file, show=show)


def make_ast_from_yml(yml_file, png_file, lazy=True):
    graph = parse(yml_file, return_dict=False, lazy=lazy)
    draw_hyperparam_tree(graph,
                         graph_path=png_file)
