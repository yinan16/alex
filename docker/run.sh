# ----------------------------------------------------------------------
# Created: mån jul 26 01:36:49 2021 (+0200)
# Last-Updated:
# Filename: run.sh
# Author: Yinan Yu
# Description:
# ----------------------------------------------------------------------

if [ $# -eq 0 ]
then
    engine="pytorch"
else
    engine=$1
fi

docker run \
       --shm-size=2g \
       --gpus all \
       -it \
       -v $PWD/examples:/ws/examples/ \
       -w /ws/ \
       --rm $engine
       /bin/bash
