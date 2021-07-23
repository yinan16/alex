# ----------------------------------------------------------------------
# Created: fre maj 21 20:16:37 2021 (+0200)
# Last-Updated:
# Filename: code_gen.py
# Author: Yinan Yu
# Description:
# Steps for code generation:
# Step 1: generate translation code
# e.g.
# def xavier_uniform(seed, shape, dtype, name):
#    return torch.nn.init.xavier_uniform_(tensor=torch.empty(*shape), dtype=torch_types[dtype])
# Step 2: Traverse tree; generate function calls
# e.g.
# xavier_uniform_id = xavier_uniform(name="some_name", seed=1)
# Step 3: Nomralize function calls
# e.g.
# xavier_uniform_id = xavier_uniform(name="some_name", seed=1, dtype="float32", shape=(64, 3, 3, 3))
# Step 4: Inline
# ----------------------------------------------------------------------
import numpy as np
import os
import collections
from abc import ABC, abstractmethod
from math import floor
from pprint import pprint
from copy import deepcopy
from typing import TypeVar, Union
import rope
from rope.base.project import Project
from rope.refactor.inline import create_inline
import warnings
import traceback

from alex.alex import core, const, node_interface, checkpoint, util
from alex.alex.checkpoint import Checkpoint
from alex.alex.annotator_interface import Annotator
from alex.engine import ns_alex, ns_tf, ns_pytorch
from alex.annotators import param_count


NAMESPACES = {"tf": ns_tf,
              "pytorch": ns_pytorch}


VALUE = TypeVar('VALUE', str, list, int, float)
FUNCTION = TypeVar('FUNCTION')
IDENTIFIER = TypeVar('IDENTIFIER')

runtime_keys = ["channels", "batch_size"]


def get_node_type(node):
    if node["value"] in ["input_shape"]:
        return IDENTIFIER # FIXME: temporary for backward compatibility
    elif node["value"] in const.FNS + ["shape"] + const.ALL_PARAMS_LIST:
        return FUNCTION
    elif len(node["children"])==0:
        return VALUE
    elif len(node["children"]) == len(node["descendants"]):
        return IDENTIFIER
    else:
        return IDENTIFIER


def _name_strs_to_names(name_strs):
    need_unpack = False
    if isinstance(name_strs, str):
        name_strs = [name_strs]
        need_unpack = True
    names = list(map(lambda x: x.replace("/", "_").replace("-", "_"),
                     name_strs))
    if need_unpack:
        names = names[0]
    return names


def _parse_str(value, node=None, literal=True):
    if not isinstance(value, str):
        return str(value)
    if value in runtime_keys:
        if value == "channels":
            parsed = node["input_shape"][-1]
        elif value == "batch_size":
            parsed = "batch_size"
            literal = False
    if value in const.ALL_COMPONENTS:
        parsed = "%s()" % value
    elif literal:
        parsed = "'%s'" % value
    else:
        parsed = "%s" % value
    return parsed


def value_to_kwarg_str(key, value, node=None, literal=True):
    if isinstance(value, list) and len(value)==1:
        value = value[0]
    if key != "":
        key = "%s=" % key
    if isinstance(value, str):
        value = _parse_str(value, node=node, literal=literal)
        kwarg_str = "%s%s" % (key, value)
    elif isinstance(value, list):
        _kwarg_str = []
        for _value in value:
            _kwarg_str.append(_parse_str(_value,
                                         node=node,
                                         literal=literal))
        _kwarg_str = ", ".join(_kwarg_str)
        kwarg_str = "%s[%s]" % (key, _kwarg_str)
    else:
        kwarg_str = "%s%s" % (key, value)
    return kwarg_str


def cache_fn(node: dict,
             fn: str,
             args: dict,
             return_symbol: str,
             name: str,
             node_type,
             inline: bool = False,
             extra: Union[str, None]=None):
    """
    return_symbol = fn(args["key"]=args["value"], ...)
    """
    node = deepcopy(node)
    if "code" not in node:
        node["code"] = collections.OrderedDict()

    arg_str = []
    for _arg in args:
        arg = args[_arg]
        if arg["value"] is not None:
            if arg["type"] == VALUE:
                _arg_str = value_to_kwarg_str("",
                                              arg["value"])
            elif arg["type"] == IDENTIFIER:
                if node_type == FUNCTION:
                    if arg["str"] == "":
                        continue
                    _arg_str = "%s=%s" % (arg["key"], arg["str"])
                else:
                    if arg["value"] == "":
                        continue
                    _arg_str = arg["value"]
            else:
                if arg["value"] == "":
                    continue
                if node_type == FUNCTION:
                    _arg_str = "%s=%s" % (arg["key"], arg["value"])
                else:
                    _arg_str = arg["value"]

            arg_str.append(_arg_str)

    if node_type == IDENTIFIER and len(arg_str)>1:
        arg_str = "[%s]" % ", ".join(arg_str)
    else:
        arg_str = ", ".join(arg_str)
    if node_type == FUNCTION:
        if len(arg_str) != "":
            arg_str = "(%s)" % (arg_str)
        code_str = "%s%s" % (fn,
                             arg_str)
    else:
        code_str = arg_str
    node["code"] = {"fn": fn,
                    "args": args,
                    "return_symbol": return_symbol,
                    "type": node_type,
                    "name": name,
                    "str": code_str,
                    "inline": inline,
                    "extra": extra}
    return node


class Ingredient(param_count.Ingredient):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def generate_code(self, node, annotated, engine):
        children = core.get_children(node["name"], annotated)
        args = dict()
        if node["input_nodes"] is not None:
            inputs = _name_strs_to_names(node["input_nodes"])
            if len(inputs) == 1:
                inputs = inputs[0]
        else:
            inputs = None

        args[const.ALEX_ARG_INPUTS] = {"key": const.ALEX_ARG_INPUTS,
                                       "value": inputs,
                                       "ref": inputs,
                                       "type": IDENTIFIER,
                                       "str": value_to_kwarg_str("", inputs,
                                                                 literal=False)}
        args[const.ALEX_ARG_NAME] = {"key": const.ALEX_ARG_NAME,
                                     "value": _name_strs_to_names(node["name"]),
                                     "ref": None,
                                     "type": IDENTIFIER,
                                     "str": "'%s'" % (node["name"])}
        if node["input_nodes"] is not None:
            input_nodes = annotated[node["input_nodes"][0]]
            args["shape"] = {"key": "shape",
                             "value": input_nodes,
                             "ref": None,
                             "type": IDENTIFIER,
                             "str": "%s" % str(input_nodes["shape"])}
        args["dtype"] = {"key": "dtype",
                         "value": node["dtype"],
                         "ref": None,
                         "type": IDENTIFIER,
                         "str": "'%s'" % (node["dtype"])}
        return_symbol = _name_strs_to_names(node["name"])

        for child in children:
            if child["value"] in args:
                continue
            _arg = child["code"]["args"]
            for _key in _arg:
                if _arg[_key]["type"] == IDENTIFIER:
                    args = {**args, **{_key: _arg[_key]}}
            if child["code"]["type"] != VALUE:
                if child["value"] in const.ALL_PARAMS:
                    _name = "%s/%s" % (core.get_parent(child["name"],
                                                       annotated,
                                                       "name"),
                                       child["value"])

                    args = {**args,
                            **{child["value"]: {"key": child["value"],
                                                "value": "trainable_params['%s']" % _name,
                                                "ref": child["name"],
                                                "type": child["code"]["type"],
                                                "str": child["code"]["str"]}}}
                else:
                    _child = {"key": child["value"],
                              "value": child["code"]["return_symbol"],
                              "ref": child["name"],
                              "type": child["code"]["type"],
                              "str": child["code"]["str"]}
                    if child["value"]=="decay":
                        if engine == "tf":
                            _child["key"] = "learning_rate"
                            args["learning_rate"] = _child
                    else:
                        args = {**args,
                                **{child["value"]: _child}}
        node = cache_fn(node,
                        node["value"],
                        args,
                        return_symbol,
                        node["name"],
                        FUNCTION)
        return node


class Regularizer(param_count.Ingredient):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def generate_code(self, node, annotated, engine):
        children = core.get_children(node["name"], annotated)
        args = dict()
        args[const.ALEX_ARG_INPUTS] = {"key": const.ALEX_ARG_INPUTS,
                                       "value": node["meta"]["position"]["inputs"],
                                       "ref": None,
                                       "type": IDENTIFIER,
                                       "str": value_to_kwarg_str("",
                                                                 node["meta"]["position"]["inputs"],
                                                                 literal=True)}
        args[const.ALEX_ARG_NAME] = {"key": const.ALEX_ARG_NAME,
                                     "value": _name_strs_to_names(node["name"]),
                                     "ref": None,
                                     "type": IDENTIFIER,
                                     "str": "'%s'" % (node["name"])}
        args["dtype"] = {"key": "dtype",
                         "value": node["dtype"],
                         "ref": None,
                         "type": IDENTIFIER,
                         "str": "'%s'" % (node["dtype"])}
        return_symbol = _name_strs_to_names(node["name"])

        for child in children:
            _arg = child["code"]["args"]
            for _key in _arg:
                if _arg[_key]["type"] == IDENTIFIER:
                    args = {**args, **{_key: _arg[_key]}}
            if child["code"]["type"] != VALUE:
                if child["value"] in const.ALL_PARAMS:
                    _name = "%s/%s" % (core.get_parent(child["name"],
                                                       annotated,
                                                       "name"),
                                       child["value"])

                    args = {**args,
                            **{child["value"]: {"key": child["value"],
                                                "value": "trainable_params['%s']" % _name,
                                                "ref": child["name"],
                                                "type": child["code"]["type"],
                                                "str": child["code"]["str"]}}}
                else:
                    args = {**args,
                            **{child["value"]: {"key": child["value"],
                                                "value": child["code"]["return_symbol"],
                                                "ref": child["name"],
                                                "type": child["code"]["type"],
                                                "str": child["code"]["str"]}}}
        node = cache_fn(node,
                        node["value"],
                        args,
                        return_symbol,
                        node["name"],
                        FUNCTION)
        return node


class Recipe(param_count.Recipe):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def generate_code(self, node, annotated, engine):
        node = deepcopy(node)
        node["code"] = dict()
        node["code"]["type"] = IDENTIFIER
        return node


class Function(param_count.Hyperparam):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def generate_code(self, node, annotated, engine):
        node = deepcopy(node)
        args = dict()
        return_symbol = _name_strs_to_names(node["name"])
        children = core.get_children(node["name"], annotated)
        for child in children:
            _node_type = child["code"]["type"]
            if _node_type == VALUE:
                continue
            elif _node_type == IDENTIFIER:
                _value = child["code"]["str"]
            elif _node_type == FUNCTION:
                _value = child["code"]["return_symbol"]
            args = {**args,
                    **{child["value"]: {"key": child["value"],
                                        "value": _value,
                                        "ref": child["name"],
                                        "type": child["code"]["type"],
                                        "str": child["code"]["str"]}}}
        node = cache_fn(node,
                        node["value"],
                        args,
                        return_symbol,
                        "",
                        FUNCTION)
        return node


class Value(param_count.Hyperparam):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def generate_code(self, node, annotated, engine):
        if "padding" in node["parent"] and engine=="pytorch":
            ingredient = annotated[node["meta"]["position"]["component"]]
            if ingredient["value"] in ["conv", "max_pool2d"]:
                if node["value"] == "SAME":
                    _value = "same"
                    # _value = [1, 1]
                    # inputs = annotated[ingredient["meta"]["position"]["inputs"][0]]
                    # input_shape = inputs["shape"]
                    # h, w = input_shape[0], input_shape[1]
                    strides = ingredient["meta"]["hyperparams"]["strides"]
                    if strides[0]!=1 or strides[1]!=1:
                        raise Exception("Pytorch does not support strided ops  for padding='same'")
                    # if ingredient["value"] == "conv":
                    #     k_h = ingredient["meta"]["hyperparams"]["filters"]["shape"]["kernel_size_h"]
                    #     k_w = ingredient["meta"]["hyperparams"]["filters"]["shape"]["kernel_size_w"]
                    # else:
                    #     [k_h, k_w] = ingredient["meta"]["hyperparams"]["window_shape"]
                    # padding_h_min = floor(((ingredient["shape"][0]-1)*strides[0] - h + k_h)/2)
                    # padding_w_min = floor(((ingredient["shape"][1]-1)*strides[1] - w + k_w)/2)
                    # _value = [padding_h_min, padding_w_min]
                elif node["value"] == "VALID":
                    # _value = [0, 0]
                    _value = "valid"
        else:
            _value = node["value"]
        node = cache_fn(node=deepcopy(node),
                        fn=_value,
                        args={},
                        return_symbol=_value,
                        name="",
                        node_type=VALUE)
        return node


class Shape(param_count.Hyperparam):

    def generate_code(self, node, annotated, engine):
        node = deepcopy(node)
        node_type = get_node_type(node)
        # FIXME: refactor dispatch; for now it's experimental
        args = dict()
        fn = "%s_%s_shape" % (node["ancestor"]["value"],
                              core.get_parent(node["name"],
                                              annotated,
                                              "value"))
        return_symbol = _name_strs_to_names(node["name"])

        children = core.get_children(node["name"], annotated)
        for child in children:
            _node_type = child["code"]["type"]
            if _node_type == VALUE:
                continue
            elif _node_type == IDENTIFIER:
                _value = child["code"]["str"]
            elif _node_type == FUNCTION:
                _value = child["code"]["return_symbol"]
            args = {**args,
                    **{child["value"]: {"key": child["value"],
                                        "value": _value,
                                        "ref": child["name"],
                                        "type": child["code"]["type"],
                                        "str": child["code"]["str"]}}}
        node = cache_fn(node,
                        fn,
                        args,
                        return_symbol,
                        "",
                        node_type,
                        True)
        return node


class Identifier(param_count.Hyperparam):

    def generate_code(self, node, annotated, engine):
        node = deepcopy(node)
        node_type = get_node_type(node)

        children = core.get_children(node["name"], annotated)

        args = dict()
        for child in children:
            # FIXME:
            if "code" not in child:
                continue
            args[child["value"]] = {"key": child["value"],
                                    "value": child["code"]["return_symbol"],
                                    "ref": child["name"],
                                    "type": child["code"]["type"],
                                    "str": child["code"]["str"]}
        node = cache_fn(node=node,
                        fn=node["value"],
                        args=args,
                        return_symbol=_name_strs_to_names(node["name"]),
                        name="",
                        node_type=node_type)
        return node


class Initializer(param_count.Hyperparam):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def generate_code(self, node, annotated, engine):
        node = deepcopy(node)
        node_type = get_node_type(node)

        args = dict()

        param_name = core.get_ancestor_param_node(node, annotated, field="value")
        shape_name = _name_strs_to_names("%s_%s_shape" % (node["ancestor"]["name"],
                                                          param_name))

        _name = node["name"]
        return_symbol = _name_strs_to_names(_name)

        args["shape"] = {"key": "shape",
                         "value": shape_name,
                         "ref": None,
                         "type": IDENTIFIER,
                         "str": "%s" % shape_name}

        args[const.ALEX_ARG_NAME] = {"key": const.ALEX_ARG_NAME,
                                     "value": _name_strs_to_names(_name),
                                     "ref": None,
                                     "type": IDENTIFIER,
                                     "str": "'%s'" % (_name)}

        children = core.get_children(node["name"], annotated)
        for child in children:
            _node_type = child["code"]["type"]
            if _node_type == VALUE:
                continue
            elif _node_type == IDENTIFIER:
                _value = child["code"]["str"]
            elif _node_type == FUNCTION:
                _value = child["code"]["return_symbol"]
            args = {**args,
                    **{child["value"]: {"key": child["value"],
                                        "value": _value,
                                        "ref": child["name"],
                                        "type": child["code"]["type"],
                                        "str": child["code"]["str"]}}}
        fn = node["value"]
        node = cache_fn(node,
                        fn,
                        args,
                        return_symbol,
                        _name,
                        node_type)

        return node


class TrainableParams(param_count.Hyperparam):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def generate_code(self, node, annotated, engine):
        node = deepcopy(node)
        node_type = get_node_type(node)
        args = dict()
        _name = "%s/%s" % (core.get_parent(node["name"],
                                           annotated,
                                           "name"),
                           node["value"])
        return_symbol = _name_strs_to_names(_name)
        node_type = FUNCTION

        ingredient = core.get_parent(node["name"],
                                     annotated,
                                     "value")
        is_trainable = const.PARAMS[ingredient][node["value"]]["derivative"]
        args["is_trainable"] = {"key": "is_trainable",
                                "value": is_trainable,
                                "ref": None,
                                "type": IDENTIFIER,
                                "str": "%s" % str(is_trainable)}
        fn = "%s_%s" % (core.get_parent(node["name"],
                                        annotated,
                                        "value"),
                        node["value"])

        args[const.ALEX_ARG_NAME] = {"key": const.ALEX_ARG_NAME,
                                     "value": _name_strs_to_names(_name),
                                     "ref": None,
                                     "type": IDENTIFIER,
                                     "str": "'%s'" % (_name)}
        children = core.get_children(node["name"], annotated)
        for child in children:
            if "code" not in child:
                continue
            _node_type = child["code"]["type"]
            if _node_type == VALUE:
                continue
            elif _node_type == IDENTIFIER:
                _value = child["code"]["str"]
            elif _node_type == FUNCTION:
                _value = child["code"]["return_symbol"]
            args = {**args,
                    **{child["value"]: {"key": child["value"],
                                        "value": _value,
                                        "ref": child["name"],
                                        "type": child["code"]["type"],
                                        "str": child["code"]["str"]}}}
        node = cache_fn(node=node,
                        fn=fn,
                        args=args,
                        return_symbol=return_symbol,
                        name=_name,
                        node_type=node_type,
                        inline=False,
                        extra="%s['%s'] = %s\n" % (const.ALEX_ARG_TRAINABLE_PARAMS,
                                                   _name,
                                                   return_symbol))
        return node


def nodes(node):
    value = node["value"]
    if value in const.REGULARIZERS:
        value = "regularizer"
    elif value in const.ALL_INITIALIZERS:
        value = "initializer_fn"
    elif node["value"] in const.ALL_PARAMS:
        value = "trainable_params"
    elif node["value"] in const.INGREDIENT_TYPES:
        value = "ingredient"
    elif node["value"] in const.SCHEDULER_BLOCK:
        value = FUNCTION
    _nodes = {"ingredient": Ingredient,
              "recipe": Recipe,
              "initializer_fn": Initializer,
              "trainable_params": TrainableParams,
              "shape": Shape,
              "regularizer": Regularizer,
              FUNCTION: Function,
              VALUE: Value,
              IDENTIFIER: Identifier}
    if value not in _nodes:
        if node["type"] == "hyperparam":
            value = get_node_type(node)
        else:
            value = node["type"]
    return _nodes[value](node["value"])


def generate_python(output_file,
                    config_path,
                    engine,
                    dirname=const.CACHE_BASE_PATH,
                    load_ckpt=None,
                    save_ckpt=None):

    if load_ckpt is None or load_ckpt[1] is None:
        load_from_ckpt = False
    else:
        load_from_ckpt = True

    ckpt = checkpoint.Checkpoint(config_path,
                                 load_ckpt,
                                 save_ckpt)
    util.clear_file(output_file)

    param_str = ParamCodeBlock(output_file,
                               config_path,
                               engine,
                               ckpt,
                               load_from_ckpt=load_from_ckpt,
                               dirname=const.CACHE_BASE_PATH)()

    component_str = ModelCodeBlock(output_file,
                                   config_path,
                                   engine,
                                   ckpt,
                                   load_from_ckpt=load_from_ckpt,
                                   dirname=const.CACHE_BASE_PATH)()

    loss_str = LossCodeBlock(output_file,
                             config_path,
                             engine,
                             ckpt,
                             load_from_ckpt=load_from_ckpt,
                             dirname=const.CACHE_BASE_PATH)()

    optimizer_str = OptimizerCodeBlock(output_file,
                                       config_path,
                                       engine,
                                       ckpt,
                                       load_from_ckpt=load_from_ckpt,
                                       dirname=const.CACHE_BASE_PATH)()

    scheduler_str = SchedulerCodeBlock(output_file,
                                       config_path,
                                       engine,
                                       ckpt,
                                       load_from_ckpt=load_from_ckpt,
                                       dirname=const.CACHE_BASE_PATH)()

    boiler_str = cache_boiler_plate(engine)

    code = NAMESPACES[engine].wrap_in_class(param_str,
                                            component_str,
                                            loss_str,
                                            optimizer_str,
                                            scheduler_str)
    util.write_to(boiler_str, output_file)
    util.write_to(code, output_file)


class CodeBlock(param_count.ParamCount):
    def __init__(self,
                 output_file,
                 config_path,
                 engine,
                 ckpt,
                 load_from_ckpt=False,
                 dirname=const.CACHE_BASE_PATH):
        super().__init__(config_path)

        self.load = load_from_ckpt
        self.ckpt = ckpt
        self.anno_name = "code generation"
        self.engine = engine
        self.dirname = dirname
        self.alex_cache_code_path = const.ALEX_CACHE_BASE_PATH
        os.makedirs(self.alex_cache_code_path, exist_ok=True)
        self.block_name = self.get_block_name()
        self.code_registry = self.get_code_registry()
        self.blocks = self.get_blocks()
        self.cache_code_path = os.path.join(self.alex_cache_code_path,
                                            "%s.py" % self.block_name)
        self.cache_def_path = os.path.join(self.alex_cache_code_path,
                                           "%s_def.py" % self.block_name)

        self.inline_index_fns = []
        self.inline_index_python = []

        self.translation_code = []
        self.alex_code = []
        self.alex_inline = []

        self.output_file = output_file
        util.clear_file(self.cache_def_path)

        self.passes = [[self.cache_shape],
                       [self._cache_alex_function_calls]]


    def __call__(self):
        return self.generate_dl_code()

    def get_blocks(self):
        return [self.block_name]

    def _write_cache_to_file(self):
        util.clear_file(self.cache_code_path)
        write_list_to_file(self.translation_code,
                           self.cache_def_path)

    def _cache_translation_code(self):
        cached = []
        for fn in self.code_registry:
            code = self.code_registry[fn]
            if fn in cached:
                continue
            cached.append(fn)
            code = self.code_registry[fn]
            try:
                if "alex" not in code:
                    continue
                src_fn = get_src_fn(code)
                src_args_str = get_src_args_str(code)
                trg_fn = get_trg_fn(code, self.engine)
                trg_args_str = get_trg_args_str(code, self.engine)

                code_str = get_translation(src_fn,
                                           src_args_str,
                                           trg_fn,
                                           trg_args_str)
                self.translation_code.append(code_str)
                self.inline_index_fns.append(src_fn)

            except Exception as error:
                traceback.print_exc()
                print("%s not implemented" % fn)

    def _in_block(self, node):
        if "code_block" not in node["meta"]:
            return False

        code_block = node["meta"]["code_block"]
        return code_block in self.blocks

    def _cache_alex_function_calls(self, node):
        try:
            _node = nodes(node).generate_code(node,
                                              self.annotated,
                                              self.engine)
            if _node is None:
                return node
            else:
                node = _node
            if not self._in_block(node):
                return node
            if node["code"]["type"] == FUNCTION:
                _param_node = self.annotated[self.annotated[node["parent"]]["parent"]]

                if node["value"] in const.ALL_INITIALIZERS \
                   and self.load \
                   and _param_node["parent"] in self.ckpt.matched:
                    _name = "%s/%s" % (_param_node["parent"],
                                       _param_node["value"])
                    code_str = "%s = %s(inputs=%s, dtype='%s', device=device)\n" % (node["code"]["return_symbol"],
                                                                                  "as_tensor",
                                                                                  "ckpt['%s']" % _name,
                                                                                  node["dtype"])
                else:
                    code_str = "%s = %s\n" % (node["code"]["return_symbol"],
                                              node["code"]["str"])

                if node["code"]["inline"]:
                    self.inline_index_python.append(_name_strs_to_names(node["code"]["return_symbol"]))
                    self.alex_inline.append(code_str)
                else:
                    self.alex_code.append(code_str)
                    if node["code"]["extra"]:
                        self.alex_code.append(node["code"]["extra"])

        except Exception as err:
            traceback.print_exc()
        return node

    def get_dl_code(self, fn_name, manual_args=[], return_str=None, prefix=None):

        util.concatenate_files([self.cache_def_path],
                               self.cache_code_path)
        if prefix is not None:
            write_list_to_file(prefix, self.cache_code_path)

        write_list_to_file(self.alex_inline, self.cache_code_path)
        write_list_to_file(self.alex_code, self.cache_code_path)
        filename = self.cache_code_path.split("/")[-1]
        dirname = "/".join(self.cache_code_path.split("/")[:-1])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            warnings.warn("deprecated", DeprecationWarning)
            inline(self.inline_index_fns, dirname, filename)
            inline(self.inline_index_python, dirname, filename)
        with open(self.cache_code_path, "r") as f:
            src_code = f.readlines()
        # TODO: automatically detect args
        if len(src_code) != 0:
            src_code = ns_alex.wrap_in_function(src_code, fn_name, manual_args, return_str)
        else:
            src_code = ""
        return src_code


class ParamCodeBlock(CodeBlock):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.anno_name = "Parameter code generation"
        self.cache_param_translation_code()
        self._cache_translation_code()
        self.annotate_tree()
        self._write_cache_to_file()

    @staticmethod
    def get_block_name():
        return "param_block"

    def get_code_registry(self):
        return {**const.ALL_PARAMS, **const.ALL_INITIALIZERS, **const.AS_TENSOR}

    def generate_dl_code(self):
        return self.get_dl_code(fn_name="get_trainable_params",
                                return_str="trainable_params",
                                manual_args=["ckpt=None"], # if self.load else [],
                                prefix="trainable_params = dict()\n")


    def cache_param_translation_code(self):
        cached = []
        for component in self.ckpt.components_list:
            _type = component["meta"]["type"]
            if _type not in const.PARAMS or _type in cached:
                continue
            for param in const.PARAMS[_type]:
                shape = const.PARAMS[_type][param]["shape"]
                src_fn = get_src_fn(shape)
                self.inline_index_fns.append(src_fn)

                src_args_str = get_src_args_str(shape)
                trg_fn = get_trg_fn(shape, self.engine)
                trg_args_str = get_trg_args_str(shape, self.engine)

                shape_code_str = get_translation(src_fn,
                                                 src_args_str,
                                                 trg_fn,
                                                 trg_args_str)
                self.translation_code.append(shape_code_str)
                constructor = const.CONSTRUCTORS["params"]
                src_fn = "%s_%s" % (_type, param)

                src_args_str = get_src_args_str(constructor)
                trg_fn = get_trg_fn(constructor, self.engine)
                trg_args_str = get_trg_args_str(constructor, self.engine)
                constructor_code_str = get_translation(src_fn,
                                                       src_args_str,
                                                       trg_fn,
                                                       trg_args_str)
                self.translation_code.append(constructor_code_str)
                self.inline_index_fns.append(src_fn)
            cached.append(_type)


class DataCodeBlock(CodeBlock):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.annotate_tree()

    @staticmethod
    def get_block_name():
        return "data_block"

    def get_code_registry(self):
        return const.INPUT_TYPES


class ModelCodeBlock(CodeBlock):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.anno_name = "Model code generation"
        self._cache_translation_code()
        self.annotate_tree()
        self._write_cache_to_file()

    def get_code_registry(self):
        return {**const.MODEL_BLOCK}

    @staticmethod
    def get_block_name():
        return "model_block"

    def generate_dl_code(self):
        return self.get_dl_code(fn_name="model",
                                manual_args=["data_block_input_data", # FIXME:
                                             "trainable_params",
                                             "training"])


class LossCodeBlock(CodeBlock):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.anno_name = "Loss code generation"
        self._cache_translation_code()
        self.annotate_tree()
        self._write_cache_to_file()

    def generate_dl_code(self):
        return self.get_dl_code(fn_name="get_loss",
                                manual_args=["trainable_params", "inputs"])

    def get_code_registry(self):
        return {**const.LOSS_BLOCK,
                **const.MODEL_BLOCK}

    @staticmethod
    def get_block_name():
        return "loss_block"


class OptimizerCodeBlock(CodeBlock):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.anno_name = "Optimizer code generation"
        self._cache_translation_code()
        self.annotate_tree()
        self._write_cache_to_file()

    def generate_dl_code(self):
        return self.get_dl_code(fn_name="get_optimizer",
                                manual_args=["trainable_params", ])

    def get_code_registry(self):
        if self.engine == "tf":
            code_registry = {**const.MODEL_BLOCK,
                             **const.OPTIMIZER_BLOCK,
                             **const.SCHEDULER_BLOCK}
        elif self.engine == "pytorch":
            code_registry = const.OPTIMIZER_BLOCK
        return code_registry

    def get_blocks(self):
        if self.engine == "tf":
            blocks = ["scheduler_block", "optimizer_block"]
        elif self.engine == "pytorch":
            blocks = ["optimizer_block"]
        return blocks

    @staticmethod
    def get_block_name():
        return "optimizer_block"


class SchedulerCodeBlock(CodeBlock):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.anno_name = "Scheduler code generation"
        self._cache_translation_code()
        self.annotate_tree()
        self._write_cache_to_file()

    def generate_dl_code(self):
        if len(self.code_registry) != 0:
            scheduler_str = self.get_dl_code(fn_name="get_scheduler",
                                             manual_args=["optimizer"])
        else:
            scheduler_str = ""
        return scheduler_str

    def get_code_registry(self):
        if self.engine == "tf":
            code_registry = {}
        elif self.engine == "pytorch":
            code_registry = const.SCHEDULER_BLOCK
        return code_registry

    def get_blocks(self):
        if self.engine == "tf":
            blocks = []
        elif self.engine == "pytorch":
            blocks = ["scheduler_block"]
        return blocks

    @staticmethod
    def get_block_name():
        return "scheduler_block"


# ------------------------ Helper function for translation ------------------- #
def get_trg_fn(component, engine):
    return component[engine][0]


def get_trg_args_str(component, engine):
    trg_args = list(map(lambda x: "%s=%s" % (x,
                                             component[engine][1][x]),
                        component[engine][1]))
    return "(%s)" % (", ".join(trg_args)) if len(trg_args) != 0 else ""


def get_src_fn(component):
    fn = component["alex"][0]
    return fn


def get_src_args_str(component):
    src_args = list(component["alex"][1].keys())
    return ", ".join(src_args)


def get_translation(src_fn, src_args, trg_fn, trg_args):
    code_str = "def %s(%s):\n" % (src_fn, src_args)
    code_str += "\treturn %s%s\n\n" % (trg_fn, trg_args)
    return code_str

def cache_boiler_plate(engine):
    imports = NAMESPACES[engine].add_imports()
    configs = NAMESPACES[engine].add_global_configs()
    return imports + configs


# -------------------------- Inline -------------------------------
def write_list_to_file(lst, filename):
    util.write_to("".join(lst), filename)
    # print("Written to %s" % filename)


def inline(inline_fns, dirname, filename):
    project = Project(dirname)
    resource = project.get_resource(filename)
    for inline_fn in inline_fns:
        try: # FIXME: temporarily handling not implemented error
            changes = create_inline(project,
                                    resource,
                                    resource.read().index(inline_fn)).get_changes(remove=True)
            project.do(changes)
        except Exception as e:
            pass
            # print("%s not implemented" % (inline_fn))
            # print("Error: %s", e)
