# ----------------------------------------------------------------------
# Created: sön jun 20 00:50:19 2021 (+0200)
# Last-Updated:
# Filename: local-install.sh
# Author: Yinan Yu
# Description:
# ----------------------------------------------------------------------

pip uninstall -y alex-nn
rm -r dist
python3 -m build
pip install dist/*.whl
