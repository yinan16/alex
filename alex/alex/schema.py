# ----------------------------------------------------------------------
# Created: sön apr 25 14:43:00 2021 (+0200)
# Last-Updated:
# Filename: schema.py
# Author: Yinan Yu
# Description:
# ----------------------------------------------------------------------

from glob import glob
from pprint import pprint
import os
from copy import deepcopy

from alex.alex import const, util

from jsonschema import validate


if __name__ == "__main__":

    schema = util.read_yaml(util.get_component_spec_path("conv"))["definitions"]["conv"]
    conv2d = util.read_yaml(util.get_component_path("conv"))
    conv2d["invalid_key"] = None
    validate(instance=conv2d, schema=schema)
