# ----------------------------------------------------------------------
# Created: mån aug 30 09:37:24 2021 (+0200)
# Last-Updated:
# Filename: venv.sh
# Author: Yinan Yu
# Description:
# ----------------------------------------------------------------------
python3 -m pip install --upgrade pip
pip install virtualenv

virtualenv alex

source alex/bin/activate

python3 -m pip install --upgrade pip
python3 -m pip install --upgrade build

./scripts/local-install.sh
