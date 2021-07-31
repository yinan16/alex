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
from alex.alex import core, const, dsl_parser, util, checkpoint
from alex.annotators import code_gen


class TestCodeGen(unittest.TestCase):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.config_path = "examples/configs/small1.yml"
        self.config_path_alt = "examples/configs/small2.yml"
        self.engines = ["pytorch", "tf"]

    def test_code_genenration(self):

        for engine in self.engines:
            filename = "generated_%s.py" % engine
            boiler_plate = "alex/engine/example_data_%s.py" % engine
            code_path = os.path.join(const.CACHE_BASE_PATH,
                                     filename)

            code_gen.generate_python(code_path,
                                     self.config_path,
                                     engine,
                                     dirname=const.CACHE_BASE_PATH,
                                     def_only=False)
            util.concatenate_files([code_path,
                                    boiler_plate],
                                   code_path)
            print("Saved to:", code_path)

    def test_mismatched(self):
        code_path = os.path.join(const.CACHE_BASE_PATH,
                                 "new_mismatched_generated.py")
        code_path_orig = os.path.join(const.CACHE_BASE_PATH,
                                      "small1_orig.py")
        ckpt_name = "test_code_gen_ckpt_trained.json"

        code_gen.generate_python(code_path_orig,
                                 "examples/configs/small1_orig.yml",
                                 "pytorch",
                                 dirname=const.CACHE_BASE_PATH,
                                 save_ckpt=["checkpoints",
                                            ckpt_name],
                                 def_only=False)
        util.concatenate_files([code_path_orig,
                                "alex/engine/example_data_pytorch.py"],
                               code_path_orig)

        # ckpt = checkpoint.Checkpoint("examples/configs/small1_orig.yml",
        #                              ["checkpoints",
        #                               None],
        #                              ["checkpoints", ckpt_name])
        # ckpt.save()
        code_gen.generate_python(code_path,
                                 "examples/configs/small1_linear.yml",
                                 "pytorch",
                                 dirname=const.CACHE_BASE_PATH,
                                 load_ckpt=["checkpoints",
                                            ckpt_name],
                                 def_only=False)
        util.concatenate_files([code_path,
                                "alex/engine/example_data_pytorch.py"],
                               code_path)
        print("Saved to:", code_path)


if __name__ == '__main__':
    unittest.main()
