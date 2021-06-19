import os
import queue
import random
import string
import threading
import time
import uuid
import yaml
import json
import numpy as np
from copy import deepcopy
from contextlib import contextmanager
from datetime import datetime
from glob import glob
import types
import io
from collections import OrderedDict, Iterable
import warnings
from copy import deepcopy
import shutil

from alex.alex import const


################## Data structure related
def flatten_subdict_keys(tree, parent=[], key_lst=[]):
    key_lst = deepcopy(key_lst)
    tree = deepcopy(tree)
    for k, v in tree.items():
        new_key = "/".join(parent+[k])
        if new_key in key_lst:
            duplicated_key = "Key %s exists" % new_key
            warnings.warn(duplicated_key)
        key_lst.append(new_key)
        if isinstance(v, dict):
            _, key_lst = flatten_subdict_keys(v, parent=[k], key_lst=key_lst)
    return parent, key_lst


def flatten_dict_keys(tree, ancestors=[], key_lst=[]):
    key_lst = deepcopy(key_lst)
    tree = deepcopy(tree)
    for k, v in tree.items():
        _ancestors = deepcopy(ancestors)
        new_key = "/".join(_ancestors+[k])
        if new_key in key_lst:
            duplicated_key = "Key %s exists" % new_key
            raise Exception(duplicated_key)
        key_lst.append(new_key)
        if isinstance(v, dict):
            _ancestors.append(k)
            _, key_lst = flatten_dict_keys(v, ancestors=_ancestors, key_lst=key_lst)
    return ancestors, key_lst


def keys_unique(tree):
    _, key_lst = flatten_subdict_keys(tree)
    keys = list(map(lambda x: x.split("/")[-1], key_lst))
    if len(keys) > len(list(set(keys))):
        return False
    else:
        return True


def fill_dict(dict1, dict2):
    # dict1 is new info, dict2 is original
    if not isinstance(dict2, dict):
        dict2 = dict()
    for k, v in dict1.items():
        if k not in dict2:
            dict2[k] = deepcopy(v)
        if isinstance(v, dict):
            fill_dict(v, dict2[k])


def update_ordereddict(new_dict, original_dict):
    for k, v in new_dict.items():
        if k not in original_dict:
            original_dict[k] = deepcopy(v)
        else:
            if isinstance(v, dict):
                update_ordereddict(v, original_dict[k])
            elif isinstance(v, list):
                original_dict[k] = list(set(deepcopy(v)).union(set(deepcopy(original_dict[k]))))
            elif isinstance(v, set):
                original_dict[k] = set(deepcopy(v)).union(set(deepcopy(original_dict[k])))
            else:
                original_dict[k] = deepcopy(v)


def replace_key(data, key, new_value):
    if isinstance(data, dict):
        if key in data:
            data[key] = deepcopy(new_value)
        for _key in data:
            if isinstance(data[_key], dict):
                for __key in data[_key]:
                    if isinstance(data[_key][__key], dict):
                        replace_key(data[_key][__key], key, new_value)
                    else:
                        if __key == key:
                            data[_key][__key] = deepcopy(new_value)


def get_value(data, key, found=False):
    if not keys_unique(data):
        print("Warning: %s is not unique" %key)
    data = deepcopy(data)
    if found:
        return data, found
    for k, v in data.items():
        if k == key:
            return v, True
        if type(v) is dict:
            _v, found = get_value(v, key, found)
            if found:
                return _v, found
    return None, False


################## Arg checks
def arg_list(lst, exp_len):
    if not isinstance(lst, list):
        lst = [lst]
    if len(lst)!=exp_len:
        raise Exception("Input length not expected! %i!=%i" % (len(lst),
                                                               exp_len))
    return lst


################## File related
def get_component_path(component):
    return os.path.join(const.COMPONENT_BASE_PATH, component+".yml")


def get_component_spec_path(component):
    return os.path.join(const.COMPONENT_BASE_PATH, "."+component+".yml")


def read_yaml(yml_path):
    with open(yml_path, 'r') as params:
        try:
            config = yaml.load(params, Loader=yaml.SafeLoader)
        except yaml.YAMLError as exc:
            print(exc)
    return config


def read_json(json_path):
    with open(json_path) as _file:
        data = json.load(_file)
    return data


def generate_random_string(string_length):
    return '_' + (''.join(random.choice(string.ascii_lowercase
                                        + string.digits)
                          for _ in range(string_length)))


def get_latest_file(path, ext):
    return sorted(glob(os.path.join(path, "*"+ext)), key=os.path.getmtime)[-1]


def concatenate_files(filenames, outfile):
    lines = []
    for filename in filenames:
        with open(filename, "r") as stream:
            lines += stream.readlines()
        lines.append("\n")
        lines.append("\n")
    lines = "".join(lines)
    with open(outfile, "w") as stream:
        stream.write(lines)


def write_to(exp, filename, mode="a"):
    s = exp
    if filename is None:
        return s # TODO: write to IO
    else:
        with open(filename, mode) as stream:
            stream.write(s)


def clear_file(filename):
    open(filename, 'w').close()


def delete_dir(dir_name):
    try:
        shutil.rmtree(dir_name)
    except OSError as e:
        print("Error: %s - %s." % (e.filename, e.strerror))
