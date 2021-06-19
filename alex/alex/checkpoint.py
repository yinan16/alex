# ----------------------------------------------------------------------
# Created: mån jun  3 15:56:29 2019 (+0200)
# Last-Updated:
# Filename: checkpoint.py
# Author: Yinan Yu
# Description:
# ----------------------------------------------------------------------
import json
import os
from copy import deepcopy
from pprint import pprint
import numpy as np
from pathlib import Path
from datetime import datetime
from alex.alex import const, util, dsl_parser, compare


def get_checkpoint(graph_list, states, trainable_params):
    network = dict()
    network["state"] = states
    network["components"] = []

    for component_block in graph_list:
        component = dict()
        component["value"] = dict()
        name = component_block["meta"]["name"]
        names_trainable_params = list(filter(lambda x: name+"/" in x, trainable_params))
        component["meta"] = deepcopy(component_block["meta"])
        for _key in component_block["value"]:
            if _key == const.TENSOR:
                component["value"][_key] = None
            elif _key == const.VAR:
                component["value"][_key] = dict()
                for _param in names_trainable_params:
                    # TODO: need to be framework dependent
                    # in both tensorflow and pytorch it happens to be .tolist()
                    component["value"][_key][_param] = trainable_params[_param].tolist()
            else:
                component["value"][_key] = deepcopy(component_block["value"][_key])
        network["components"].append(component)
    return network


def dump(graph_list, states, trainable_params, path):
    _network = get_checkpoint(graph_list, states, trainable_params)
    _json_cache = "/".join(path.split("/")[0:-1] + ["."+path.split("/")[-1]])
    with open(_json_cache, "w") as write_file:
        json.dump(_network, write_file, indent=4)
    os.rename(_json_cache, path)


def load_json(checkpoint_dir, json_file=None):
    json_file = get_load_path(checkpoint_dir, json_file)
    data = util.read_json(json_file)
    return data


def load(graph_list,
         checkpoint_dir,
         ckpt_name=None,
         matched=None):
    graph_list = deepcopy(graph_list)
    states = init_states()
    ckpt = load_json(checkpoint_dir, ckpt_name)
    trainable_params = dict()
    if ckpt is not None:
        states = ckpt["state"]
        ckpt_components = dsl_parser.list_to_dict(ckpt["components"])
        for i, component in enumerate(graph_list):
            name_in_new_config = component["meta"]["name"]
            if matched is None or name_in_new_config in matched:
                name_in_ckpt = matched[component["meta"]["name"]]
                _params = deepcopy(ckpt_components[name_in_ckpt]["value"]["var"])
                component["value"]["var"] = dict()
                for _param in _params:
                    _param_name = _param.split("/")[-1]
                    param_name = "%s/%s" % (name_in_new_config, _param_name)
                    trainable_params[param_name] = _params[_param]
                    graph_list[i]["value"]["var"][param_name] = _params[_param]

    return graph_list, trainable_params, states


############ Setup data:
def init_states(train_info={"meta": {"epoch": 0}},
                valid_info={"meta": dict(), "logs": {"loss": [],
                                                     "accuracy": []}},
                test_info=dict()):
    return {"valid": valid_info,
            "train": train_info,
            "test": test_info}


def setup_variable(component_value, init, init_value, var_name, **kwargs):
    """
    component_value (dict):
        "hyperparams" (dict)
        "var" (dict): {"variable name": a framework specific varialbel}
        "stats" (dict): {"variable name": statistics and additional information summarized from the training data}
        "tensor" (dict): {"variable name": framework specific tensor that contains the values of the parameters}
        "value" (dict): {"variable name": numpy array that contains the values of the parameters (or other serializable container)}
    init (callable): a framework specific callable to return a variable, e.g. tf.Variable
    init_value (numpy array): initial value for the variable; returned by an initializer
    var_name (str): name of the variable
    """
    if const.VAR not in component_value:
        component_value[const.VAR] = dict()
    if var_name not in component_value[const.VAR]:
        component_value[const.VAR][var_name] = None
    if component_value[const.VALUE] == {} or (var_name not in component_value[const.VALUE]):
        var = init(initial_value=init_value, name=var_name, trainable=True, **kwargs)
    else:
        var = init(initial_value=np.asarray(component_value[const.VALUE][var_name]),
                   name=var_name,
                   trainable=True,
                   **kwargs)
    component_value[const.VAR][var_name] = var
    return var


def setup(component_info, var_def, var_init, var_name, **kwargs):
    if var_name not in component_info[const.VAR]:
        component_info[const.VAR][var_name] = None
    if component_info[const.VALUE] == {} or (var_name not in component_info[const.VALUE]):
        var = var_def(initial_value=var_init, name=var_name, **kwargs)
    else:
        var = var_def(initial_value=np.asarray(component_info[const.VALUE][var_name], dtype=np.float32), name=var_name, **kwargs)
    component_info[const.VAR][var_name] = var
    return var


def save(graph_list, states, trainable_params, path):
    os.makedirs(path.split("/")[0], exist_ok=True)
    dump(graph_list, states, trainable_params, path)


def get_checkpoint_path(ckpt_dir, ckpt_name=None, checkpoint_ext=".json"):
    if ckpt_name is None:
        _id = "_" + str(datetime.timestamp(datetime.now())).replace(".", "")
        ckpt_name = "config" + _id
    if checkpoint_ext not in ckpt_name:
        ckpt_name += checkpoint_ext
    return os.path.join(ckpt_dir,
                        ckpt_name)


def get_load_path(ckpt_dir, ckpt_name=None, ext=".json"):
    json_file = None
    if ckpt_name is not None:
        if ckpt_name.split(".")[0] == "latest":
            json_file = util.get_latest_file(ckpt_dir,
                                             ext=ext)
        else:
            if ext not in ckpt_name:
                ckpt_name += ext
            json_file = os.path.join(ckpt_dir,
                                     ckpt_name)
    if json_file is None or not Path(json_file).is_file():
        return None
    else:
        return json_file


class Checkpoint():

    def __init__(self,
                 config,
                 load=["log", None],
                 save=["log", None]):

        self.components_list = dsl_parser.parse(config)
        self.load_path = get_load_path(ckpt_dir=load[0],
                                       ckpt_name=load[1])
        self.save_path = get_checkpoint_path(ckpt_dir=save[0],
                                             ckpt_name=save[1])
        self.save_dir = "/".join(self.save_path.split("/")[:-1])
        self.save_name = self.save_path.split("/")[-1]

        if self.load_path is not None:
            self.load_dir = "/".join(self.load_path.split("/")[:-1])
            self.load_name = self.load_path.split("/")[-1]
            # Check the diff between config and load:
            loaded_component_list = load_json(self.load_dir,
                                              self.load_name)["components"]
            self.matched = compare.matched_ingredients(self.components_list,
                                                       loaded_component_list,
                                                       os.path.join(self.save_dir,
                                                                    "ckpt_diff.png"))
        else:
            self.matched = None


    def load(self):
        if self.load_path is None:
            trainable_params = None
            stats = None
        else:
            self.components_list, trainable_params, stats = load(self.components_list,
                                                                 self.load_dir,
                                                                 ckpt_name=self.load_name,
                                                                 matched=self.matched)
        return trainable_params

    def save(self, trainable_params, stats={}):
        save(self.components_list, stats, trainable_params, self.save_path)
        print("Saving to path", self.save_path)
