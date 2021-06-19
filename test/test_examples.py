# ----------------------------------------------------------------------
# Created: lör maj 15 17:31:04 2021 (+0200)
# Last-Updated:
# Filename: test_examples.py
# Author: Yinan Yu
# Description:
# ----------------------------------------------------------------------
from alex.alex import const
import unittest
import subprocess, os


class TestExamples(unittest.TestCase):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.example_dir = const.EXAMPLES_PATH

    def setUp(self):
        pass

    def test_compare(self):
        subprocess.run(["python3", os.path.join(self.example_dir, "compare.py")], check=True)

    def test_generate_python(self):
        subprocess.run(["python3", os.path.join(self.example_dir, "generate_python.py")], check=True)


if __name__ == '__main__':
    unittest.main()
