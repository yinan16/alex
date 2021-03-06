# ----------------------------------------------------------------------
# Created: fre mar 12 19:58:21 2021 (+0100)
# Last-Updated:
# Filename: ns_alex.py
# Author: Yinan Yu
# Description: alex namespace
# ----------------------------------------------------------------------
import inspect
import math
import collections.abc as collections
from collections import OrderedDict
from copy import deepcopy
import os
from pprint import pprint
from alex.alex import const, util
# ---------------------------------------------------------------------------- #


def indent(src: str, levels=1):
    return ("\t"*levels + src).expandtabs(4)


def wrap_in_function(src, fn, args, return_val_str=None):
    if not isinstance(args, list):
        args = [args]
    fn_src = "def %s(%s):\n" % (fn, ", ".join(args))
    if isinstance(src, str):
        src = src.split("\n")
    for line in src:
        _line = indent(line)
        if "\n" not in _line:
            _line += "\n"
        fn_src += _line
    if return_val_str == "":
        return_str = ""
    else:
        if return_val_str is None:
            return_val_str = line.split("=")[0]
        return_str = "return %s\n" % return_val_str
    fn_src += indent(return_str)
    return fn_src
