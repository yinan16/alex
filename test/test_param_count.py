# ----------------------------------------------------------------------
# Created: lör maj 15 20:16:25 2021 (+0200)
# Last-Updated:
# Filename: test_annotator.py
# Author: Yinan Yu
# Description:
# ----------------------------------------------------------------------
import unittest
import os
from pprint import pprint
from alex.alex import core, const
from alex.annotators import param_count


class TestParamCount(unittest.TestCase):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.config_path = "examples/configs/small2.yml"

    def test_annotate_trainbale_params(self):
        params_counter = param_count.ParamCount(self.config_path)
        annotated = params_counter.annotate_tree()
        core.draw(annotated,
                  graph_path=os.path.join(const.CACHE_BASE_PATH,
                                          'trainable_param_count.png'),
                  label_field="count",
                  excluded_types=["hyperparam"])
        core.draw(annotated,
                  graph_path=os.path.join(const.CACHE_BASE_PATH,
                                          'shape.png'),
                  label_field="shape",
                  excluded_types=["hyperparam"])


if __name__ == '__main__':
    unittest.main()
