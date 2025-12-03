# module geni.py
#
# Copyright (c) 2018 Rafael Reis
#
"""
geni module - Auxiliary functions to the GENI method.
"""
__version__="1.0"

import numpy as np

from pctsp.model.pctsp import *
from pctsp.model import solution


def genius(pctsp):
    s = solution.random(pctsp, size=3)
    s = geni(pstsp, s)
    s = us(pctsp, s)
    return s

def geni(pctsp, s):
    return

def us(pctsp, s):
    return