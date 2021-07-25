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

    def test_codegen(self):
        command = ["alex-nn",
                   "codegen",
                   "examples/configs/small1.yml",
                   "--engine", "pytorch",
                   "--out_dir", "cache",
                   "--filename", "cli_codegen_pytorch.py"]
        run(command)
        print(" ".join(command))

        command = ["alex-nn",
                   "codegen",
                   "examples/configs/small1.yml",
                   "--engine", "tf",
                   "--out_dir", "cache",
                   "--filename", "cli_codegen_tf.py"]
        run(command)
        print(" ".join(command))

    def test_codegen_with_ckpt(self):
        command = ["alex-nn",
                   "codegen",
                   "examples/configs/small1.yml",
                   "--engine", "pytorch",
                   "--out_dir", "cache",
                   "--filename", "cli_codegen_ckpt_pytorch.py",
                   "--ckpt_from", "checkpoints/test.json"]
        run(command)
        print(" ".join(command))

        command = ["alex-nn",
                   "codegen",
                   "examples/configs/small1.yml",
                   "--engine", "tf",
                   "--out_dir", "cache",
                   "--filename", "cli_codegen_ckpt_tf.py",
                   "--ckpt_from", "checkpoints/test.json"]
        run(command)
        print(" ".join(command))

    def test_diff(self):
        command = ["alex-nn",
                   "diff",
                   "examples/configs/small1.yml",
                   "examples/configs/small3.yml",
                   "--mode", "diff",
                   "--dpi", "1600",
                   "--to_png", "cache/cli_diff"]
        run(command)
        print(" ".join(command))

    def test_dist(self):
        command = ["alex-nn",
                   "diff",
                   "examples/configs/small1.yml",
                   "examples/configs/small3.yml",
                   "--mode", "dist",
                   "--dpi", "1600",
                   "--to_png", "cache/cli_dist"]
        run(command)


if __name__ == '__main__':
    unittest.main()
