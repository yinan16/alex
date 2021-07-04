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
         filename = "generated_code",
         code_dir=const.CACHE_BASE_PATH,
         engines=["tf"]):
    for engine in engines:
        _filename = "%s_%s.py" % (filename, engine)
        code_generator = code_gen.CodeGen(_filename,
                                          network_config,
                                          engine=engine,
                                          dirname=code_dir)
        code_generator.generate_python()
        filepath = os.path.join(code_dir, _filename)
        boilerplate_file = os.path.join(const.ENGINE_PATH,
                                        "example_boilerplate_%s.py" % engine)
        util.concatenate_files([filepath,
                                boilerplate_file],
                               filepath)
        print("Generated code for %s written to %s" % (engine, filepath))


if __name__=="__main__":
    main("./examples/configs/small1.yml", engines=["pytorch", "tf"])
    # main("./examples/configs/cifar_test_params.yml")
