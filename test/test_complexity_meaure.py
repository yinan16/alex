# ----------------------------------------------------------------------
# Created: ons maj 19 01:39:56 2021 (+0200)
# Last-Updated:
# Filename: test_complexity_meaure.py
# Author: Yinan Yu
# Description:
# ----------------------------------------------------------------------
import os
from alex.annotators import complexity_measure
from alex.alex import core, const
import unittest


class TestComplexityMeasure(unittest.TestCase):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.config_path = "examples/configs/small2.yml"

    def test_compute_halstead(self):
        complexity = complexity_measure.halstead(12, 7, 27, 15)
        self.assertAlmostEqual(complexity["vocabulary"], 19, places=2)
        self.assertAlmostEqual(complexity["length"], 42, places=2)
        self.assertAlmostEqual(complexity["volume"], 178.41, places=2)
        self.assertAlmostEqual(complexity["time"], 127.46, places=2)
        self.assertAlmostEqual(complexity["difficulty"], 12.86, places=2)
        self.assertAlmostEqual(complexity["bugs"], 0.06, places=2)
        self.assertAlmostEqual(complexity["effort"], 2294.35, places=2)

    def test_compute_cyclomatic(self):
        self.assertEqual(complexity_measure.cyclomatic(8, 8, 8), 16)

    def test_annotate_complexity(self):
        complexity = complexity_measure.ComplexityMeasure(self.config_path)
        annotated = complexity.annotate_tree()
        core.draw(annotated,
                  graph_path=os.path.join(const.CACHE_BASE_PATH,
                                          'complexity.png'),
                  label_field="complexity",
                  excluded_types=["hyperparam"])


if __name__ == '__main__':
    unittest.main()
