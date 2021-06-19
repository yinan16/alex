# ----------------------------------------------------------------------
# Created: tor mar 25 10:03:41 2021 (+0100)
# Last-Updated:
# Filename: test_util.py
# Author: Yinan Yu
# Description:
# ----------------------------------------------------------------------

from alex.alex import util
import unittest
from collections import OrderedDict, Iterable


class TestUtil(unittest.TestCase):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        pass

    def setUp(self):
        pass

    def test_fill_dict(self):
        dict1 = {"fn1": {}, "fn2": {}}
        dict1["fn1"]["var1"] = "def1"
        dict1["fn1"]["var2"] = "def2"
        dict1["fn2"]["var1"] = "def1"
        dict2 = {"fn2": {"var2": "def3"}}
        util.fill_dict(dict1, dict2)
        self.assertDictEqual(dict2, {"fn1": {"var1": "def1",
                                             "var2": "def2"},
                                     "fn2": {"var1": "def1",
                                             "var2": "def3"}})

    def test_updateordered_dict(self):
        def _get_dict1():
            dict1 = {"fn1": OrderedDict(),
                     "fn2": OrderedDict()}
            dict1["fn1"]["var1"] = "def1"
            dict1["fn1"]["var2"] = "def2"
            dict1["fn2"]["var1"] = "def1"
            dict1["fn2"]["var2"] = "def2"
            return dict1

        dict1 = _get_dict1()
        dict2 = {"fn2": {"var2": "def3"}}
        util.update_ordereddict(dict2, dict1)
        self.assertDictEqual(dict1, OrderedDict({"fn1": {"var1": "def1",
                                                         "var2": "def2"},
                                                 "fn2": {"var1": "def1",
                                                         "var2": "def3"}}))
        dict1 = _get_dict1()
        dict2 = {"fn3": {"var2": "def3"}}
        util.update_ordereddict(dict2, dict1)
        self.assertDictEqual(dict1, OrderedDict({"fn1": {"var1": "def1",
                                                         "var2": "def2"},
                                                 "fn2": {"var1": "def1",
                                                         "var2": "def2"},
                                                 "fn3": {"var2": "def3"}}))

        dict1 = _get_dict1()
        dict2 = {"fn2": 0}
        util.update_ordereddict(dict2, dict1)
        self.assertDictEqual(dict1, OrderedDict({"fn1": {"var1": "def1",
                                                         "var2": "def2"},
                                                 "fn2": 0}))

        dict1 = _get_dict1()
        dict1["fn3"] = {"var1": [1, 2, 3]}
        dict2 = {"fn3": {"var1": [2, 3, 4]}}
        util.update_ordereddict(dict2, dict1)
        self.assertDictEqual(dict1, OrderedDict({"fn1": {"var1": "def1",
                                                         "var2": "def2"},
                                                 "fn2": {"var1": "def1",
                                                         "var2": "def2"},
                                                 "fn3": {"var1": [1, 2, 3, 4]}}))


        dict1 = _get_dict1()
        dict1["fn3"] = {"var1": [1, 2, 3], "var2": "def2"}
        dict2 = {"fn3": {"var1": [2, 3, 4], "var2": "def5"}}
        util.update_ordereddict(dict2, dict1)
        self.assertDictEqual(dict1, OrderedDict({"fn1": {"var1": "def1",
                                                         "var2": "def2"},
                                                 "fn2": {"var1": "def1",
                                                         "var2": "def2"},
                                                 "fn3": {"var1": [1, 2, 3, 4],
                                                         "var2": "def5"}}))

    def test_get_value(self):

        dict1 = {"a": 1,
                 "b": {"c": {"d": 4},
                       "e": {"f": {"g": 2}, "h": 5}},
                 "i": {"j": 100}}
        res, found = util.get_value(dict1, "g", found=False)
        self.assertEqual(res, 2)
        self.assertEqual(found, True)
        res, found = util.get_value(dict1, "i", found=False)
        self.assertDictEqual(res, {"j": 100})
        self.assertEqual(found, True)

    def test_replace_key(self):
        dict1 = {"a": 1,
                 "b": {"c": {"d": 4},
                       "e": {"f": {"g": 2}, "h": 5}},
                 "i": {"j": 100}}
        dict2 = {"a": 1,
                 "b": {"c": {"d": 4},
                       "e": {"f": 99, "h": 5}},
                 "i": {"j": 100}}
        util.replace_key(dict1, "f", 99)
        self.assertDictEqual(dict1, dict2)
        dict1 = {"a": 1,
                 "b": {"c": {"d": 4},
                       "e": {"f": {"g": 2}, "h": 5}},
                 "i": {"j": 100}}
        dict3 = {"a": 1,
                 "b": {"c": {"d": 4},
                       "e": {"f": {"k": {"l": 1}}, "h": 5}},
                 "i": {"j": 100}}
        util.replace_key(dict1, "f", {"k": {"l": 1}})
        self.assertDictEqual(dict1, dict3)

    def test_flatten_dict_keys(self):
        dict1 = {"a": 1,
                 "b": {"c": {"d": 4},
                       "e": {"f": {"g": 2}, "h": 5}}}
        ancestors, key_lst = util.flatten_dict_keys(dict1)
        self.assertSetEqual(set(key_lst),
                            {"a", "b",
                             "b/c/d", "b/c",
                             "b/e", "b/e/f", "b/e/f/g", "b/e/h"})

    def test_flatten_subdict_keys(self):
        dict1 = {"a": 1,
                 "b": {"c": {"d": 4},
                       "e": {"f": {"g": 2}, "h": 5}}}
        ancestors, key_lst = util.flatten_subdict_keys(dict1)
        self.assertSetEqual(set(key_lst),
                            {"a", "b",
                             "c/d", "b/c",
                             "b/e", "e/f", "f/g", "e/h"})

    def test_keys_unique(self):
        dict1 = {"a": 1,
                 "b": {"c": {"d": 4},
                       "e": {"f": {"g": 2}, "h": 5}}}
        self.assertTrue(util.keys_unique(dict1))
        dict2 = {"a": 1,
                 "b": {"c": {"d": 4},
                       "b": {"f": {"g": 2}, "h": 5}}}

        self.assertTrue(not util.keys_unique(dict2))


if __name__ == '__main__':
    unittest.main()
