# ----------------------------------------------------------------------
# Created: tor jul 22 11:27:31 2021 (+0200)
# Last-Updated:
# Filename: test_dsl_parser.py
# Author: Yinan Yu
# Description:
# ----------------------------------------------------------------------


import unittest
from alex.alex import dsl_parser
from pprint import pprint
import collections


class TestDslParser(unittest.TestCase):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.config = "./examples/configs/small3.yml"

    def test_make_graph_from_yml(self):
        dsl_parser.make_graph_from_yml(yml_file=self.config,
                                       png_file="./cache/dsl_parser_graph.png",
                                       level=2, show="name")

    def test_make_ast_from_yml(self):
        dsl_parser.make_ast_from_yml(yml_file=self.config,
                                     png_file="./cache/dsl_parser_ast.png")


    def test_list_to_graph(self):
        glist = dsl_parser.parse(self.config, return_dict=False)
        graph = dsl_parser.list_to_graph(glist)
        # pprint(graph)


    # def test_list_to_graph_simple(self):
    #     glist = [{"type": "root/block1/conv", "meta": {"scope": "block1", "name": "root/block1/conv"}},
    #              {"type": "root/block1/conv2", "meta": {"scope": "block1", "name": "root/block1/conv2"}},
    #              {"type": "root/block2/conv", "meta": {"scope": "block2", "name": "root/block2/conv"}}]
    #     # tree = collections.OrderedDict({"root": {"meta": {},
    #     #                                          "subgraph": collections.OrderedDict(
    #     #                                              {"root/block1": {"meta": {},
    #     #                                                               "subgraph": collections.OrderedDict({"root/block1/conv": {"meta": {"name": "conv"},
    #     #                                                                                                                         "subgraph": "conv"}})}})}})
    #     graph = dsl_parser.list_to_graph(glist)


if __name__ == '__main__':
    unittest.main()
