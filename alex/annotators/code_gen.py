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
import ast, typing
from collections.abc import Iterable

from alex.alex import core, const, node_interface, checkpoint, util, registry
from alex.alex.checkpoint import Checkpoint
from alex.alex.annotator_interface import Annotator
from alex.engine import ns_alex, ns_tf, ns_pytorch
from alex.annotators import param_count


NAMESPACES = {"tf": ns_tf,
              "pytorch": ns_pytorch}


# Tags: value, function, identifier
VALUE = TypeVar('VALUE', str, list, int, float)
FUNCTION = TypeVar('FUNCTION')
IDENTIFIER = TypeVar('IDENTIFIER')

runtime_keys = ["channels", "batch_size"]


def get_node_tag(node):
    if node["value"] in ["input_shape"]:
        tag = IDENTIFIER # FIXME: temporary for backward compatibility
    elif node["value"] in registry.TAG_FUNCTIONS:
        tag = FUNCTION
    elif len(node["children"])==0: # is leaf
        tag = VALUE
    elif len(node["children"]) == len(node["descendants"]):
        tag = IDENTIFIER
    else:
        tag = IDENTIFIER
    return tag


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
    if literal:
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
             node_tag,
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
            if arg["tag"] == VALUE:
                _arg_str = value_to_kwarg_str("",
                                              arg["value"])
            elif arg["tag"] == IDENTIFIER:
                if node_tag == FUNCTION:
                    # node is function and arg is identifier
                    if arg["str"] == "":
                        continue
                    # FIXME: arg["value"] is dict
                    _arg_str = "%s=%s" % (arg["key"], arg["str"])

                elif node_tag == IDENTIFIER:
                    # node is identifier and arg is identifier
                    if arg["value"] == "":
                        continue
                    _arg_str = arg["value"]
            elif arg["tag"] == FUNCTION:
                if arg["value"] == "":
                    continue
                if node_tag == FUNCTION:
                    # node is function and arg is function
                    _arg_str = "%s=%s" % (arg["key"], arg["value"])

                elif node_tag == IDENTIFIER:
                    # node is identifier and arg is function
                    _arg_str = arg["value"]

            arg_str.append(_arg_str)

    if node_tag == IDENTIFIER and len(arg_str)>1:
        arg_str = "[%s]" % ", ".join(arg_str)
    else:
        arg_str = ", ".join(arg_str)
    if node_tag == FUNCTION:
        if len(arg_str) != "":
            arg_str = "(%s)" % (arg_str)
        code_str = "%s%s" % (fn,
                             arg_str)
    else:
        code_str = arg_str
    node["code"] = {"fn": fn,
                    "args": args,
                    "return_symbol": return_symbol,
                    "tag": node_tag,
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
                                       "tag": IDENTIFIER,
                                       "str": value_to_kwarg_str("", inputs,
                                                                 literal=False)}
        args[const.ALEX_ARG_NAME] = {"key": const.ALEX_ARG_NAME,
                                     "value": _name_strs_to_names(node["name"]),
                                     "ref": None,
                                     "tag": IDENTIFIER,
                                     "str": "'%s'" % (node["name"])}
        if node["input_nodes"] is not None:
            input_nodes = annotated[node["input_nodes"][0]]
            args["shape"] = {"key": "shape",
                             "value": "%s" % str(input_nodes["shape"]), # input_nodes,
                             "ref": None,
                             "tag": IDENTIFIER,
                             "str": "%s" % str(input_nodes["shape"])}
        args["dtype"] = {"key": "dtype",
                         "value": node["dtype"],
                         "ref": None,
                         "tag": IDENTIFIER,
                         "str": "'%s'" % (node["dtype"])}
        return_symbol = _name_strs_to_names(node["name"])

        for child in children:
            if child["value"] in args:
                continue
            # _arg = child["code"]["args"]
            # for _key in _arg:
            #     if _arg[_key]["tag"] == IDENTIFIER:
            #         args = {**args, **{_key: _arg[_key]}}
            if child["code"]["tag"] != VALUE:
                if child["value"] in registry.ALL_PARAMS:
                    _name = "%s/%s" % (core.get_parent(child["name"],
                                                       annotated,
                                                       "name"),
                                       child["value"])
                    args = {**args,
                            **{child["value"]: {"key": child["value"],
                                                "value": "trainable_params['%s']" % _name,
                                                "ref": child["name"],
                                                "tag": child["code"]["tag"],
                                                "str": child["code"]["str"]}}}
                else:
                    _child = {"key": child["value"],
                              "value": child["code"]["return_symbol"],
                              "ref": child["name"],
                              "tag": child["code"]["tag"],
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
                                       "tag": IDENTIFIER,
                                       "str": value_to_kwarg_str("",
                                                                 node["meta"]["position"]["inputs"],
                                                                 literal=True)}
        args[const.ALEX_ARG_NAME] = {"key": const.ALEX_ARG_NAME,
                                     "value": _name_strs_to_names(node["name"]),
                                     "ref": None,
                                     "tag": IDENTIFIER,
                                     "str": "'%s'" % (node["name"])}
        args["dtype"] = {"key": "dtype",
                         "value": node["dtype"],
                         "ref": None,
                         "tag": IDENTIFIER,
                         "str": "'%s'" % (node["dtype"])}
        return_symbol = _name_strs_to_names(node["name"])

        for child in children:
            _arg = child["code"]["args"]
            for _key in _arg:
                if _arg[_key]["tag"] == IDENTIFIER:
                    args = {**args, **{_key: _arg[_key]}}
            if child["code"]["tag"] != VALUE:
                if child["value"] in registry.ALL_PARAMS:
                    _name = "%s/%s" % (core.get_parent(child["name"],
                                                       annotated,
                                                       "name"),
                                       child["value"])

                    args = {**args,
                            **{child["value"]: {"key": child["value"],
                                                "value": "trainable_params['%s']" % _name,
                                                "ref": child["name"],
                                                "tag": child["code"]["tag"],
                                                "str": child["code"]["str"]}}}
                else:
                    args = {**args,
                            **{child["value"]: {"key": child["value"],
                                                "value": child["code"]["return_symbol"],
                                                "ref": child["name"],
                                                "tag": child["code"]["tag"],
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
        node["code"]["tag"] = IDENTIFIER
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
            _node_tag = child["code"]["tag"]
            if _node_tag == VALUE:
                continue
            elif _node_tag == IDENTIFIER:
                _value = child["code"]["str"]
            elif _node_tag == FUNCTION:
                _value = child["code"]["return_symbol"]
            args = {**args,
                    **{child["value"]: {"key": child["value"],
                                        "value": _value,
                                        "ref": child["name"],
                                        "tag": child["code"]["tag"],
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
                # FIXME:
                if node["value"] == "SAME":
                    _value = [1, 1]

                elif node["value"] == "VALID":
                    _value = [0, 0]

                # "valid" and "same" are only available for pytorch > 1.9
                # if node["value"] == "SAME":
                #     _value = "same"
                #     # _value = [1, 1]
                #     # inputs = annotated[ingredient["meta"]["position"]["inputs"][0]]
                #     # input_shape = inputs["shape"]
                #     # h, w = input_shape[0], input_shape[1]
                #     strides = ingredient["meta"]["hyperparams"]["strides"]
                #     if strides[0]!=1 or strides[1]!=1:
                #         raise Exception("Pytorch does not support strided ops  for padding='same'")
                #     # if ingredient["value"] == "conv":
                #     #     k_h = ingredient["meta"]["hyperparams"]["filters"]["shape"]["kernel_size_h"]
                #     #     k_w = ingredient["meta"]["hyperparams"]["filters"]["shape"]["kernel_size_w"]
                #     # else:
                #     #     [k_h, k_w] = ingredient["meta"]["hyperparams"]["window_shape"]
                #     # padding_h_min = floor(((ingredient["shape"][0]-1)*strides[0] - h + k_h)/2)
                #     # padding_w_min = floor(((ingredient["shape"][1]-1)*strides[1] - w + k_w)/2)
                #     # _value = [padding_h_min, padding_w_min]
                # elif node["value"] == "VALID":
                #     _value = "valid"
        else:
            _value = node["value"]
        node = cache_fn(node=deepcopy(node),
                        fn=_value,
                        args={},
                        return_symbol=_value,
                        name="",
                        node_tag=VALUE)
        return node


class Shape(param_count.Hyperparam):

    def generate_code(self, node, annotated, engine):
        node = deepcopy(node)
        node_tag = get_node_tag(node)
        args = dict()
        fn = "%s_%s_shape" % (node["ancestor"]["value"],
                              core.get_parent(node["name"],
                                              annotated,
                                              "value"))
        return_symbol = _name_strs_to_names(node["name"])

        children = core.get_children(node["name"], annotated)
        for child in children:
            _node_tag = child["code"]["tag"]
            if _node_tag == VALUE:
                continue
            elif _node_tag == IDENTIFIER:
                _value = child["code"]["str"]
            elif _node_tag == FUNCTION:
                _value = child["code"]["return_symbol"]
            args = {**args,
                    **{child["value"]: {"key": child["value"],
                                        "value": _value,
                                        "ref": child["name"],
                                        "tag": child["code"]["tag"],
                                        "str": child["code"]["str"]}}}
        node = cache_fn(node,
                        fn,
                        args,
                        return_symbol,
                        "",
                        node_tag,
                        True)
        return node


class Identifier(param_count.Hyperparam):

    def generate_code(self, node, annotated, engine):
        node = deepcopy(node)
        node_tag = get_node_tag(node)

        children = core.get_children(node["name"], annotated)

        args = dict()
        for child in children:
            # FIXME:
            if "code" not in child:
                continue
            args[child["value"]] = {"key": child["value"],
                                    "value": child["code"]["return_symbol"],
                                    "ref": child["name"],
                                    "tag": child["code"]["tag"],
                                    "str": child["code"]["str"]}
        node = cache_fn(node=node,
                        fn=node["value"],
                        args=args,
                        return_symbol=_name_strs_to_names(node["name"]),
                        name="",
                        node_tag=node_tag)
        return node


class Initializer(param_count.Hyperparam):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def generate_code(self, node, annotated, engine):
        node = deepcopy(node)
        node_tag = get_node_tag(node)

        args = dict()

        param_name = core.get_ancestor_param_node(node, annotated, field="value")
        shape_name = _name_strs_to_names("%s_%s_shape" % (node["ancestor"]["name"],
                                                          param_name))
        _name = node["name"]
        return_symbol = _name_strs_to_names(_name)

        args["shape"] = {"key": "shape",
                         "value": shape_name,
                         "ref": None,
                         "tag": IDENTIFIER,
                         "str": "%s" % shape_name}

        args[const.ALEX_ARG_NAME] = {"key": const.ALEX_ARG_NAME,
                                     "value": _name_strs_to_names(_name),
                                     "ref": None,
                                     "tag": IDENTIFIER,
                                     "str": "'%s'" % (_name)}

        children = core.get_children(node["name"], annotated)
        for child in children:
            _node_tag = child["code"]["tag"]
            if _node_tag == VALUE:
                continue
            elif _node_tag == IDENTIFIER:
                _value = child["code"]["str"]
            elif _node_tag == FUNCTION:
                _value = child["code"]["return_symbol"]
            args = {**args,
                    **{child["value"]: {"key": child["value"],
                                        "value": _value,
                                        "ref": child["name"],
                                        "tag": child["code"]["tag"],
                                        "str": child["code"]["str"]}}}
        fn = node["value"]
        node = cache_fn(node,
                        fn,
                        args,
                        return_symbol,
                        _name,
                        node_tag)

        return node


class TrainableParams(param_count.Hyperparam):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def generate_code(self, node, annotated, engine):
        node = deepcopy(node)
        node_tag = get_node_tag(node)
        args = dict()
        _name = "%s/%s" % (core.get_parent(node["name"],
                                           annotated,
                                           "name"),
                           node["value"])
        return_symbol = _name_strs_to_names(_name)
        node_tag = FUNCTION

        ingredient = core.get_parent(node["name"],
                                     annotated)
        if ingredient["meta"]["trainable"] is not None:
            is_trainable = ingredient["meta"]["trainable"]
        else:
            is_trainable = registry.PARAMS[ingredient["value"]][node["value"]]["derivative"]
        args["is_trainable"] = {"key": "is_trainable",
                                "value": is_trainable,
                                "ref": None,
                                "tag": IDENTIFIER,
                                "str": "%s" % str(is_trainable)}
        fn = "%s_%s" % (core.get_parent(node["name"],
                                        annotated,
                                        "value"),
                        node["value"])

        args[const.ALEX_ARG_NAME] = {"key": const.ALEX_ARG_NAME,
                                     "value": _name_strs_to_names(_name),
                                     "ref": None,
                                     "tag": IDENTIFIER,
                                     "str": "'%s'" % (_name)}
        children = core.get_children(node["name"], annotated)
        for child in children:
            if "code" not in child:
                continue
            _node_tag = child["code"]["tag"]
            if _node_tag == VALUE:
                continue
            elif _node_tag == IDENTIFIER:
                _value = child["code"]["str"]
            elif _node_tag == FUNCTION:
                _value = child["code"]["return_symbol"]
            args = {**args,
                    **{child["value"]: {"key": child["value"],
                                        "value": _value,
                                        "ref": child["name"],
                                        "tag": child["code"]["tag"],
                                        "str": child["code"]["str"]}}}
        node = cache_fn(node=node,
                        fn=fn,
                        args=args,
                        return_symbol=return_symbol,
                        name=_name,
                        node_tag=node_tag,
                        inline=False,
                        extra="%s['%s'] = %s\n" % (const.ALEX_ARG_TRAINABLE_PARAMS,
                                                   _name,
                                                   return_symbol))
        return node


def nodes(node):
    value = node["value"]
    if value in registry.REGULARIZERS:
        value = "regularizer"
    elif value in registry.ALL_INITIALIZERS:
        value = "initializer_fn"
    elif node["value"] in registry.ALL_PARAMS:
        value = "trainable_params"
    elif node["value"] in registry.INGREDIENT_TYPES:
        value = "ingredient"
    elif node["value"] in registry.SCHEDULER_BLOCK:
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
            value = get_node_tag(node)
        else:
            value = node["type"]
    return _nodes[value](node["value"])


def generate_python(output_file,
                    config_path,
                    engine,
                    dirname=const.CACHE_BASE_PATH,
                    load_ckpt=None,
                    save_ckpt=None,
                    def_only=True):

    if load_ckpt is None or load_ckpt[1] is None:
        load_from_ckpt = False
    else:
        load_from_ckpt = True

    ckpt = checkpoint.Checkpoint(config_path,
                                 engine,
                                 load_ckpt,
                                 save_ckpt)
    util.clear_file(output_file)

    param_str, params_args = ParamCodeBlock(output_file,
                                            config_path,
                                            engine,
                                            ckpt,
                                            load_from_ckpt=load_from_ckpt,
                                            dirname=dirname)()

    model_str, model_args = ModelCodeBlock(output_file,
                                           config_path,
                                           engine,
                                           ckpt,
                                           load_from_ckpt=load_from_ckpt,
                                           dirname=dirname)()
    loss = LossCodeBlock(output_file,
                         config_path,
                         engine,
                         ckpt,
                         load_from_ckpt=load_from_ckpt,
                         dirname=dirname)
    loss_str, loss_args = loss()
    loss_mode = loss.loss_mode
    optimizer_str, optimizer_args = OptimizerCodeBlock(output_file,
                                                       config_path,
                                                       engine,
                                                       ckpt,
                                                       load_from_ckpt=load_from_ckpt,
                                                       dirname=dirname)()

    scheduler_str, scheduler_args = SchedulerCodeBlock(output_file,
                                                       config_path,
                                                       engine,
                                                       ckpt,
                                                       load_from_ckpt=load_from_ckpt,
                                                       dirname=dirname)()

    boiler_str = cache_boiler_plate(engine)

    code = NAMESPACES[engine].wrap_in_class(param_str,
                                            model_str,
                                            loss_str,
                                            optimizer_str,
                                            scheduler_str,
                                            load_from_ckpt=load_from_ckpt,
                                            model_args=model_args,
                                            trainable_params_args=params_args)
    util.write_to(boiler_str, output_file)
    util.write_to(code, output_file)
    if not def_only:
        instanstiate_str = NAMESPACES[engine].instantiate(config=config_path,
                                                          engine=engine,
                                                          load_from=load_ckpt,
                                                          save_to=save_ckpt,
                                                          params_args=params_args,
                                                          optimizer_args=optimizer_args)

        inference_func_name, inference_str = NAMESPACES[engine].inference(model_args,
                                                                          loss_mode)
        inference_str, inference_args = assemble_func_src_code(inference_str,
                                                               inference_func_name,
                                                               "",
                                                               exclude_args=NAMESPACES[engine].DEFINED)
        evaluation_func_name, evaluation_str = NAMESPACES[engine].evaluation(inference_args,
                                                                             loss_args,
                                                                             loss_mode)
        evaluation_str, evaluation_args = assemble_func_src_code(evaluation_str,
                                                                 evaluation_func_name,
                                                                 "",
                                                                 exclude_args=NAMESPACES[engine].DEFINED)

        train_func_name, train_str = NAMESPACES[engine].train(model_args, loss_args)
        train_str, train_args = assemble_func_src_code(train_str,
                                                       train_func_name,
                                                       "", exclude_args=NAMESPACES[engine].DEFINED)

        loop_func_name, loop_str = NAMESPACES[engine].loop(save_ckpt,
                                                           train_args,
                                                           evaluation_args)
        loop_str, loop_args = assemble_func_src_code(loop_str,
                                                     loop_func_name,
                                                     "",
                                                     exclude_args=NAMESPACES[engine].DEFINED)
        # TODO: change this
        data_src = os.path.join(const.ENGINE_PATH, "data_%s.py" % engine)
        data_write_src = os.path.join(const.ENGINE_PATH, "example_data_%s.py" % engine)
        util.clear_file(data_write_src)
        with open(data_write_src, "a") as f:
            with open(data_src, "r") as fr:
                lines = fr.readlines()
            f.write("".join(lines) + "\n" + loop_str.split("\n")[0].replace("def ", "").replace(":", ""))

        src = [instanstiate_str, inference_str, evaluation_str, train_str, loop_str]
        src_str = "\n".join(src)
        util.write_to(src_str, output_file)


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

        self.passes = [[self.cache_shape_and_block],
                       [self._cache_alex_function_calls]]
        self.loss_mode = None # FIXME

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

            if node["code"]["tag"] == FUNCTION:
                _param_node = self.annotated[self.annotated[node["parent"]]["parent"]]

                if node["value"] in registry.ALL_INITIALIZERS \
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
                if node["value"] in registry.CLASSIFICATION_LOSSES:
                    self.loss_mode = "classification"
                elif node["value"] in registry.REGRESSION_LOSSES:
                    self.loss_mode = "regression"
                if self.block_name=="model_block" \
                   and "probe" in node["meta"] \
                   and node["meta"]["probe"]:
                    code_str += "probes['%s'] = %s\n" % (node["name"],
                                                         node["code"]["return_symbol"])
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

    def get_dl_code(self, fn_name, exclude_args=[], manual_args=[], return_str=None, prefix=None):

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
            __src_code = f.readlines()
        # TODO: automatically detect args
        if len(__src_code) != 0:
            src_code, args = assemble_func_src_code(__src_code, fn_name, return_str, exclude_args=exclude_args, manual_args=manual_args)
        else:
            src_code = ""
            args = []
        return src_code, args


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
        return {**registry.ALL_PARAMS,
                **registry.ALL_INITIALIZERS,
                **registry.AS_TENSOR}

    def generate_dl_code(self):
        if self.load:
            manual_args = ["ckpt"]
        else:
            manual_args = []
        return self.get_dl_code(fn_name="get_trainable_params",
                                return_str="trainable_params",
                                exclude_args=NAMESPACES[self.engine].DEFINED,
                                manual_args=manual_args,
                                prefix="trainable_params = dict()\n")


    def cache_param_translation_code(self):
        cached = []
        for component in self.ckpt.components_list:
            _type = component["meta"]["type"]
            if _type not in registry.PARAMS or _type in cached:
                continue
            for param in registry.PARAMS[_type]:
                shape = registry.PARAMS[_type][param]["shape"]
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
                constructor = registry.PARAM_CONSTRUCTORS["params"]
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
        return registry.DATA_BLOCK


class ModelCodeBlock(CodeBlock):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.anno_name = "Model code generation"
        self._cache_translation_code()
        self.annotate_tree()
        self._write_cache_to_file()

    def get_code_registry(self):
        return {**registry.MODEL_BLOCK}

    @staticmethod
    def get_block_name():
        return "model_block"

    def generate_dl_code(self):
        return self.get_dl_code(fn_name="model",
                                return_str="model_block_output",
                                exclude_args=NAMESPACES[self.engine].DEFINED)


class LossCodeBlock(CodeBlock):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.anno_name = "Loss code generation"
        self._cache_translation_code()
        self.annotate_tree()
        self._write_cache_to_file()

    def generate_dl_code(self):
        return self.get_dl_code(fn_name="get_loss",
                                exclude_args=NAMESPACES[self.engine].DEFINED)

    def get_code_registry(self):
        return {**registry.LOSS_BLOCK,
                **registry.MODEL_BLOCK}

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
                                exclude_args=NAMESPACES[self.engine].DEFINED)

    def get_code_registry(self):
        if self.engine == "tf":
            code_registry = {**registry.OPTIMIZER_BLOCK,
                             **registry.SCHEDULER_BLOCK}
        elif self.engine == "pytorch":
            code_registry = registry.OPTIMIZER_BLOCK
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
            scheduler_str, args = self.get_dl_code(fn_name="get_scheduler",
                                                   exclude_args=NAMESPACES[self.engine].DEFINED)
        else:
            scheduler_str = ""
            args = []
        return scheduler_str, args

    def get_code_registry(self):
        if self.engine == "tf":
            code_registry = {}
        elif self.engine == "pytorch":
            code_registry = registry.SCHEDULER_BLOCK
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


def assemble_func_src_code(code_body, fn_name, return_str, exclude_args=[], manual_args=[]):
    src_code = ns_alex.wrap_in_function(code_body,
                                        fn_name,
                                        args=[],
                                        return_val_str=return_str)
    local_symbols, defined_symbols = get_symbols_from_func_def_literal(src_code)
    args = set(local_symbols).difference(set(defined_symbols))
    args = sorted(list(args.difference(exclude_args)))
    args += manual_args
    args = list(set(args))
    src_code = ns_alex.wrap_in_function(code_body, fn_name, args, return_str)
    return src_code, args


def get_symbols_from_func_def_literal(code_str):

    local_symbols = []
    defined_symbols = []

    if isinstance(code_str, str):
        parsed = ast.parse(code_str).body[0].body
    elif isinstance(code_str, ast.Call):
        parsed = code_str.keywords + code_str.args
    elif isinstance(code_str, (ast.Tuple, ast.List)):
        parsed = code_str.elts
    elif not isinstance(code_str, Iterable):
        parsed = [code_str]
    else:
        parsed = code_str
    for i, line in enumerate(parsed):
        if isinstance(line, (int, float, bool, str, ast.Return, ast.UnaryOp, ast.Constant,
                             ast.Str, ast.Num, # deprecated in 3.8
        )):
            continue
        elif isinstance(line, ast.Name):
            local_symbols.append(line.id)
        elif isinstance(line, ast.Assign):
            # left hand side:
            targets = line.targets
            if isinstance(targets, list):
                for var in targets:
                    if isinstance(var, ast.Subscript): # mutating an object
                        local_symbols.append(var.value.id)
                    elif isinstance(var, ast.Name):
                        defined_symbols.append(var.id)
                    else:
                        try: # FIXME
                            defined_symbols.append(var.value.id)
                        except:
                            _local_symbols, _defined_symbols = get_symbols_from_func_def_literal(var)
                            local_symbols += _local_symbols
                            defined_symbols += _defined_symbols
            else:
                defined_symbols.append(targets.value.id)

            value = line.value
            if isinstance(value, ast.Name):
                local_symbols.append(value.id)
            else:
                _local_symbols, _defined_symbols = get_symbols_from_func_def_literal(value)
                local_symbols += _local_symbols
                defined_symbols += _defined_symbols
        else:
            if line is None:
                continue
            if isinstance(line, (ast.With, ast.For, ast.If)):
                obj = line.body
                if isinstance(line, ast.For):
                    defined_symbols.append(line.target.id)
            elif isinstance(line, ast.Lambda):
                obj = line.body
            elif isinstance(line, (ast.Tuple, ast.List, ast.Call)):
                obj = line
            elif isinstance(line, ast.UnaryOp):
                obj = line.operand
            elif isinstance(line, ast.BinOp):
                obj = [line.left, line.right]
            else:
                obj = line.value
            _local_symbols, _defined_symbols = get_symbols_from_func_def_literal(obj)
            local_symbols += _local_symbols
            defined_symbols += _defined_symbols

    return local_symbols, defined_symbols


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
