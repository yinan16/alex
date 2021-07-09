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
        print("Testing checkpoint")

    def setUp(self):
        pass

    def test_checkpoint(self):
        ckpt = checkpoint.Checkpoint("examples/configs/small2.yml",
                                     ["checkpoints",
                                      "config_1625826546926544.json"],
                                     ["checkpoints", None])
        # ckpt.save()
        print("Matched components")
        pprint(ckpt.matched)


if __name__ == '__main__':
    unittest.main()
