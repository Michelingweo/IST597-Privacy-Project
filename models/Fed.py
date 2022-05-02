#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import copy
import torch
import os

from torch.autograd import Variable
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms
import PIL.Image as Image



def FedAvg(w):
    w_avg = copy.deepcopy(w[0])
    for k in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[k] += gaussian(w[i][k])
        w_avg[k] = torch.div(w_avg[k], len(w))
    return w_avg


def gaussian(ins, mean = 0, stddev = 0.01):
    noise = Variable(ins.data.new(ins.size()).normal_(mean, stddev))
    return ins + noise


def gaussian_img(img, mean = 0, stddev = 0.01):
    img = np.array(img, dtype=float)
    img = img/255
    noise = np.random.normal(mean, stddev**0.5, img.shape)
    out = img + noise
    return out


