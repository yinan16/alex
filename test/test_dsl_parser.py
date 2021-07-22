# ----------------------------------------------------------------------
# Created: tor jul 22 11:27:31 2021 (+0200)
# Last-Updated:
# Filename: test_dsl_parser.py
# Author: Yinan Yu
# Description:
# ----------------------------------------------------------------------


import unittest
from alex.alex import dsl_parser


class TestDslParser(unittest.TestCase):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.config = "./examples/configs/small1.yml"

    def test_make_graph_from_yml(self):
        dsl_parser.make_graph_from_yml(yml_file=self.config,
                                       png_file="./cache/dsl_parser_graph.png",
                                       level=3)

    def test_make_ast_from_yml(self):
        dsl_parser.make_ast_from_yml(yml_file=self.config,
                                     png_file="./cache/dsl_parser_ast.png")


if __name__ == '__main__':
    unittest.main()
