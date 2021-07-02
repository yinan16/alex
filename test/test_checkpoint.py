# ----------------------------------------------------------------------
# Created: mån maj 31 18:55:54 2021 (+0200)
# Last-Updated:
# Filename: test_checkpoint.py
# Author: Yinan
# Description:
# ----------------------------------------------------------------------

import unittest
from alex.alex import checkpoint
from pprint import pprint


class TestCheckpoint(unittest.TestCase):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def setUp(self):
        pass

    # def test_checkpoint(self):
    #     ckpt = checkpoint.Checkpoint("examples/configs/small1.yml",
    #                                  ["checkpoints",
    #                                   "config_1622420349826577.json"],
    #                                  ["checkpoints", None])
    #     pprint(ckpt.matched)


if __name__ == '__main__':
    unittest.main()
