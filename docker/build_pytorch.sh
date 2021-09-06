# ----------------------------------------------------------------------
# Created: mån jul 26 03:22:15 2021 (+0200)
# Last-Updated:
# Filename: build_pytorch.sh
# Author: Yinan Yu
# Description:
# ----------------------------------------------------------------------

docker build -t pytorch -f ./docker/Dockerfile.pytorch ./
