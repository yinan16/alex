# ----------------------------------------------------------------------
# Created: tis mar 16 15:23:47 2021 (+0100)
# Last-Updated:
# Filename: engine_user.py
# Author: Yinan Yu
# Description:
# ----------------------------------------------------------------------
from alex.engine import ns_alex


def lr_scheduler(epoch, epochs, lrs):
    if len(lrs) != len(epochs)+1:
        raise Exception("Length mismatch!")
    prev_e = 0
    if epoch >= epochs[-1]:
        return lrs[-1]
    for i, l in enumerate(lrs):
        if prev_e <= epoch < epochs[i]:
            lr = l
            break
        prev_e = epochs[i]

    return lr
