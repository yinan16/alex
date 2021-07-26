# ----------------------------------------------------------------------
# Created: mån jul 26 03:29:00 2021 (+0200)
# Last-Updated:
# Filename: run_code.py
# Author: Yinan Yu
# Description:
# ----------------------------------------------------------------------
import subprocess


def test_codegen_and_run():
    for engine in ["tf"]:
        command = ["alex-nn",
                   "codegen",
                   "examples/configs/small1.yml",
                   "--engine", engine,
                   "--out_dir", "cache",
                   "--run",
                   "--append", "alex/engine/example_boilerplate_%s.py" % engine,
                   "--filename", "cli_codegen_%s.py" % engine]
        subprocess.run(" ".join(command), shell=True, check=True)
        print(command)


if __name__=="__main__":
    test_codegen_and_run()
