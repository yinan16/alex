# ----------------------------------------------------------------------
# Created: tor mar 25 10:03:41 2021 (+0100)
# Last-Updated:
# Filename: test_example.py
# Author: Yinan Yu
# Description:
# ----------------------------------------------------------------------

import unittest


class TestExample(unittest.TestCase):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def setUp(self):
        pass

    def test_function(self):
        pass


if __name__ == '__main__':
    unittest.main()
