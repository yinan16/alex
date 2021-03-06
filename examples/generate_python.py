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


def main(network_config,
         filename="generated_code",
         code_dir="./cache/",
         engines=["tf"]):
    os.makedirs(code_dir, exist_ok=True)
    for engine in engines:
        _filename = "%s_%s.py" % (filename, engine)
        filepath = os.path.join(code_dir, _filename)
        code_gen.generate_python(filepath,
                                 network_config,
                                 engine=engine,
                                 dirname=code_dir,
                                 def_only=False)
        boilerplate_file = os.path.join(const.ENGINE_PATH,
                                        "example_data_%s.py" % engine)
        util.concatenate_files([filepath,
                                boilerplate_file],
                               filepath)
        print("Generated code for %s written to %s" % (engine, filepath))


if __name__=="__main__":
    main("./examples/configs/small1.yml", engines=["pytorch", "tf"])
    # main("./examples/configs/cifar_test_params.yml")
