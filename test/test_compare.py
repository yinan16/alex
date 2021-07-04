# ----------------------------------------------------------------------
# Created: mån maj 10 01:52:20 2021 (+0200)
# Last-Updated:
# Filename: test_compare.py
# Author: Yinan Yu
# Description:
# ----------------------------------------------------------------------
import unittest
import os
from pprint import pprint
import collections
from copy import deepcopy
from alex.alex import core, compare, const, util


class TestCompare(unittest.TestCase):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.conv = util.read_yaml(util.get_component_path("conv"))
        self.modified_conv = deepcopy(self.conv)
        self.modified_conv["padding"] = "WRONG"
        self.modified_conv["filters"]["nfilters"] = dict()
        self.modified_conv["filters"]["nfilters"]["terrible"] = 5
        self.modified_conv["filters"]["shape"]["added"] = None
        self.modified_conv["filters"]["initializer"]["xavier_uniform"] = "changed"
        self.toy_json = collections.OrderedDict()
        a_subtree = collections.OrderedDict()
        a_subtree["c"] = "l"
        a_subtree["h"] = None
        self.toy_json["f"] = collections.OrderedDict()
        self.toy_json["f"]["a"] = a_subtree
        self.toy_json["f"]["e"] = None
        self.toy_json2 = collections.OrderedDict()
        a_subtree2 = collections.OrderedDict()
        a_subtree2["d"] = "i"
        a_subtree2["c"] = collections.OrderedDict()
        a_subtree2["c"]["j"] = None
        a_subtree2["c"]["b"] = collections.OrderedDict()
        a_subtree2["c"]["b"]["m"] = 5
        self.toy_json2["f"] = collections.OrderedDict()
        self.toy_json2["f"]["a"] = a_subtree2
        self.toy_json2["f"]["k"] = None

    def setUp(self):
        pass

    def test_get_children(self):
        conv_tree = core.json_to_tree(self.conv, "conv")
        conv_key = list(conv_tree.keys())[-1]
        expected_children = ["dilation",
                             "filters",
                             "padding",
                             "strides"]
        children = core.get_children(conv_key, conv_tree, "label")
        self.assertListEqual(children, expected_children)

    def test_json_to_tree(self):
        conv_tree = core.json_to_tree(self.conv, "conv")
        toy_tree = core.json_to_tree(self.toy_json)

    def test_get_keyroots(self):
        toy_tree = core.json_to_tree(self.toy_json)
        keyroots = compare.get_keyroots(toy_tree)
        self.assertListEqual(keyroots, [2, 4, 5])

    def test_ted(self):
        toy_tree = core.json_to_tree(self.toy_json)
        toy_tree2 = core.json_to_tree(self.toy_json2)
        costs, operations = compare.ted(toy_tree, toy_tree2)
        annotation = compare.annotate_ops(operations)
        self.assertEqual(costs, 8)
        core.draw(toy_tree, os.path.join(const.CACHE_BASE_PATH,
                                         "toytree1.png"),
                  annotation[1])
        core.draw(toy_tree2, os.path.join(const.CACHE_BASE_PATH,
                                          "toytree2.png"),
                  annotation[2])

    def test_ted_on_conv(self):
        conv_tree1 = core.json_to_tree(self.conv, "conv")
        conv_tree2 = core.json_to_tree(self.modified_conv, "conv")
        costs, operations = compare.ted(conv_tree1, conv_tree2)
        self.assertEqual(costs, 7)
        annotation = compare.annotate_ops(operations)
        core.draw(conv_tree1,
                  os.path.join(const.CACHE_BASE_PATH,
                               "conv1.png"),
                  annotation[1])
        core.draw(conv_tree2,
                  os.path.join(const.CACHE_BASE_PATH,
                               "conv2.png"),
                  annotation[2])

    def test_dist(self):
        matched = compare.matched_ingredients("examples/configs/small1.yml",
                                              "examples/configs/small2.yml",
                                              render_to="./cache/subtree.png")
        print("Matched ingredients")
        pprint(matched)


if __name__ == '__main__':
    unittest.main()
