# ----------------------------------------------------------------------
# Created: lör maj 22 10:39:51 2021 (+0200)
# Last-Updated:
# Filename: test_code_gen.py
# Author: Yinan Yu
# Description:
# ----------------------------------------------------------------------
import unittest
import os
from pprint import pprint
from alex.alex import core, const, dsl_parser, util
from alex.annotators import code_gen


class TestCodeGen(unittest.TestCase):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.config_path = "examples/configs/small1.yml"
        self.config_path_alt = "examples/configs/small2.yml"
        self.engines = ["tf", "pytorch"]

    def test_code_genenration(self):

        for engine in self.engines:
            filename = "generated_%s.py" % engine
            self.boiler_plate = "alex/engine/example_boilerplate_%s.py" % engine
            code_path = os.path.join(const.CACHE_BASE_PATH,
                                     filename)
            # code_generator = code_gen.CodeGen(filename,
            #                                   self.config_path,
            #                                   engine,
            #                                   dirname=const.CACHE_BASE_PATH,
            #                                   load_ckpt=["cache",
            #                                              None])
            code_generator = code_gen.CodeGen(filename,
                                              self.config_path,
                                              engine,
                                              dirname=const.CACHE_BASE_PATH,
                                              load_ckpt=["checkpoints",
                                                         "config_1626993992750915.json"]) # "config_1622411488965296.json"
            code_generator.generate_python()
            util.concatenate_files([code_path,
                                    self.boiler_plate],
                                   code_path)
            print("Saved to:", code_path)

    # def test_mismatched(self):
    #     code_path = os.path.join(const.CACHE_BASE_PATH,
    #                              "new_mismatched_generated.py")
    #     code_generator = code_gen.CodeGen("new_mismatched_generated.py",
    #                                       self.config_path_alt,
    #                                       "pytorch",
    #                                       dirname=const.CACHE_BASE_PATH,
    #                                       load_ckpt=["cache",
    #                                                  "config_1622420349826577.json"]) # "config_1622411488965296.json"
    #     annotated = code_generator.generate_python()
    #     util.concatenate_files([code_path,
    #                             "alex/engine/example_boilerplate_pytorch.py"],
    #                            code_path)
    #     print("Saved to:", code_path)


if __name__ == '__main__':
    unittest.main()
