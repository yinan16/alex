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
import ast


class TestCodeGen(unittest.TestCase):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.config_path = "examples/configs/small1.yml"
        self.config_path_alt = "examples/configs/small3.yml"
        self.engines = ["tf", "pytorch"]

    def test_get_symbols_from_func_def_literal(self):
        x = "def add():  \n\t a=b(c) \n\t loss_block_regularizer = 0.002*sum(list(map(lambda x: torch.norm(input=trainable_params[x]), ['model_block/conv_6gw/filters', 'model_block/conv_14oi/filters', 'model_block/conv_16qy/filters', 'model_block/dense_20ue/weights'])))"

        print(code_gen.get_symbols_from_func_def_literal(x))

    def test_code_genenration(self):
        for engine in self.engines:
            filename = "test_code_gen_%s_small3.py" % engine
            boiler_plate = "alex/engine/example_data_%s.py" % engine
            code_path = os.path.join(const.CACHE_BASE_PATH,
                                     filename)

            code_gen.generate_python(code_path,
                                     self.config_path_alt,
                                     engine,
                                     dirname=const.CACHE_BASE_PATH,
                                     def_only=False)
            util.concatenate_files([code_path,
                                    boiler_plate],
                                   code_path)
            print("Saved to:", code_path)

    def test_mismatched(self):

        for engine in self.engines:
            code_path = os.path.join(const.CACHE_BASE_PATH,
                                     "test_code_gen_linear_%s.py" % engine)
            code_path_orig = os.path.join(const.CACHE_BASE_PATH,
                                          "test_code_gen_small1_orig.py")
            ckpt_name = "test_code_gen_ckpt_trained.json"
            code_gen.generate_python(code_path_orig,
                                     "examples/configs/small1_orig.yml",
                                     engine,
                                     dirname=const.CACHE_BASE_PATH,
                                     save_ckpt=["checkpoints",
                                                ckpt_name],
                                     def_only=False)
            util.concatenate_files([code_path_orig,
                                    "alex/engine/example_data_%s.py" % engine],
                                   code_path_orig)

            # ckpt = checkpoint.Checkpoint("examples/configs/small1_orig.yml",
            #                              engine,
            #                              ["checkpoints",
            #                               None],
            #                              ["checkpoints", ckpt_name])
            # ckpt.save()
            code_gen.generate_python(code_path,
                                     "examples/configs/small1_linear.yml",
                                     engine,
                                     dirname=const.CACHE_BASE_PATH,
                                     load_ckpt=["checkpoints",
                                                ckpt_name],
                                     def_only=False)
            util.concatenate_files([code_path,
                                    "alex/engine/example_data_%s.py" % engine],
                                   code_path)
            print("Saved to:", code_path)


if __name__ == '__main__':
    unittest.main()
