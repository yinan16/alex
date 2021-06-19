# ----------------------------------------------------------------------
# Created: sön apr 25 15:24:51 2021 (+0200)
# Last-Updated:
# Filename: test_schema.py
# Author: Yinan Yu
# Description:
# ----------------------------------------------------------------------
import unittest
from pprint import pprint
import os
from copy import deepcopy
from glob import glob
from jsonschema import validate

from alex.alex import (schema, util, const)


class TestSchema(unittest.TestCase):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.conv_expected = {'filters':
                              {'shape':
                               {'kernel_size_w': 3,
                                'kernel_size_h': 3,
                                'input_shape': 'channels',
                                'n_filters': 64},
                               'initializer':
                               {'xavier_uniform': {'seed': None}}},
                              'padding': [1, 1],
                              'strides': [2, 2],
                              'use_bias': False,
                              'dilation': 1}


    def test_expand_ingredient(self):
        conv = util.read_yaml(util.get_component_path("conv"))
        conv_schema = util.read_yaml(util.get_component_spec_path("conv"))["definitions"]["conv"]
        validate(conv, conv_schema)
        conv_invalid = deepcopy(self.conv_expected)
        conv_invalid["filters"]["shape"]["input_shape"] = "not_valid_value"
        with self.assertRaises(Exception) as context:
            validate(conv_invalid, conv_schema)

        conv_invalid = deepcopy(self.conv_expected)
        conv_invalid["invalid_key"] = None
        with self.assertRaises(Exception) as context:
            validate(conv_invalid, conv_schema)


    def test_schemas(self):
        """A fuzzy test to check if the schema is reasonable.
           The assumption is that the component is well formed."""
        all_schemas = sorted(glob(os.path.join(const.COMPONENT_BASE_PATH, ".*.yml")))
        for _schema in all_schemas:
            component_name = _schema.split("/")[-1].split(".")[1]
            component = util.read_yaml(util.get_component_path(component_name))
            component_schema = util.read_yaml(util.get_component_spec_path(component_name))["definitions"][component_name]
            validate(component, component_schema)
            print(component_name, "schema checking")


if __name__ == '__main__':
    unittest.main()
