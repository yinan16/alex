# ----------------------------------------------------------------------
# Created: fre jul 23 01:42:20 2021 (+0200)
# Last-Updated:
# Filename: test_cli.py
# Author: Yinan Yu
# Description:
# ----------------------------------------------------------------------

import unittest
import subprocess


def run(command):
    print("Running", " ".join(command))
    subprocess.run(command, check=True)


class TestCli(unittest.TestCase):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def setUp(self):
        pass

    def test_diff(self):
        command = ["alex-nn",
                   "diff",
                   "examples/configs/small1.yml",
                   "examples/configs/small2.yml",
                   "--mode", "diff",
                   "--to_png", "cache/cli_diff"]
        run(command)
        print(" ".join(command))

    def test_dist(self):
        command = ["alex-nn",
                   "diff",
                   "examples/configs/small1.yml",
                   "examples/configs/small2.yml",
                   "--mode", "dist",
                   "--to_png", "cache/cli_dist"]
        run(command)


if __name__ == '__main__':
    unittest.main()
