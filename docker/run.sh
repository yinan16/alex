# ----------------------------------------------------------------------
# Created: mån jul 26 01:36:49 2021 (+0200)
# Last-Updated:
# Filename: run.sh
# Author: Yinan Yu
# Description:
# ----------------------------------------------------------------------

docker run \
       --shm-size=2g \
       --gpus all \
       -v $PWD/examples:/ws/examples/ \
       -w /ws/ \
       --rm pytorch
