# ----------------------------------------------------------------------
# Created: mån jul 26 01:16:34 2021 (+0200)
# Last-Updated:
# Filename: install.sh
# Author: Yinan Yu
# Description:
# ----------------------------------------------------------------------

sudo apt update && sudo apt remove docker docker.io
sudo apt install docker.io
sudo systemctl start docker && sudo systemctl enable docker
docker --version

sudo usermod -a -G docker $USER
newgrp docker

distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list

sudo apt update && sudo apt install -y nvidia-container-toolkit
sudo systemctl restart docker
