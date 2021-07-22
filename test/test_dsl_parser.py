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


class TestDslParser(unittest.TestCase):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.config = "./examples/configs/small4.yml"

    def test_make_graph_from_yml(self):
        dsl_parser.make_graph_from_yml(yml_file=self.config,
                                       png_file="./cache/dsl_parser_graph.png",
                                       level=1, show="shape")

    def test_make_ast_from_yml(self):
        dsl_parser.make_ast_from_yml(yml_file=self.config,
                                     png_file="./cache/dsl_parser_ast.png")


    def test_parse(self):
        glist = dsl_parser.parse(self.config, return_dict=False)
        graph = dsl_parser.list_to_graph(glist)
        pprint(graph)


if __name__ == '__main__':
    unittest.main()
