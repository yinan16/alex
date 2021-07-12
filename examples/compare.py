# ----------------------------------------------------------------------
# Created: tor maj 13 23:13:42 2021 (+0200)
# Last-Updated:
# Filename: compare.py
# Author: Yinan Yu
# Description:
# ----------------------------------------------------------------------
import os
from alex.alex import compare


def main(code_dir):
    config_path_1 = "examples/configs/small1.yml"
    config_path_2 = "examples/configs/small3.yml"

    os.makedirs(code_dir, exist_ok=True)
    # config_path_1 = "examples/configs/cifar_test_params.yml"
    # config_path_2 = "examples/configs/cifar_test_params_modify.yml"
    render_dist_ingredient_path = os.path.join(code_dir, "network_ingredient_diff.png")
    print("Compare %s and %s" % (config_path_1, config_path_2))
    compare.dist(config_path_1, config_path_2, exclude_types=["hyperparam"], render_to=render_dist_ingredient_path)

    render_diff_path = os.path.join("./", "network_diff.png")
    compare.diff(config_path_1, config_path_2, render_to=render_diff_path)

    render_dist_path = os.path.join("./", "network_dist.png")
    compare.dist(config_path_1, config_path_2, render_to=render_dist_path)


if __name__=="__main__":
    main("./cache/")
