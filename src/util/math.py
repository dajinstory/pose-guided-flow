import os
import subprocess as sp
from time import time

import torch
import torch.nn as nn

import numpy as np
from math import log, sqrt, pi, exp, cos, sin

def floor(x):
    '''
    https://github.com/kitayama1234/Pytorch-BPDA
    '''
    forward_value = torch.floor(x)
    out = x.clone()
    out.data = forward_value.data
    return out

def round(x):
    forward_value = torch.round(x)
    out = x.clone()
    out.data = forward_value.data
    return out
    