# ----------------------------------------------------------------------
# Created: tor maj 13 22:54:11 2021 (+0200)
# Last-Updated:
# Filename: generate_python.py
# Author: Yinan Yu
# Description:
# ----------------------------------------------------------------------

import os
from alex.annotators import code_gen
from alex.alex import const, util
from alex.engine import ns_alex


def main(network_config,
         filename="example_generated.py",
         code_dir=const.CACHE_BASE_PATH,
         engine="pytorch"):
    code_generator = code_gen.CodeGen(filename,
                                      network_config,
                                      engine=engine,
                                      dirname=code_dir)
    code_generator.generate_python()

    boilerplate_file = os.path.join(const.ENGINE_PATH,
                                    "example_boilerplate_%s.py" % engine)
    filepath = os.path.join(code_dir, filename)
    util.concatenate_files([filepath,
                            boilerplate_file],
                           filepath)


if __name__=="__main__":
    main("./examples/configs/small1.yml")
    # main("./examples/configs/cifar_test_params.yml")
