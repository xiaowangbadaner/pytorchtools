'''
make one hot for mask
may have lots of version

    by:W.H
'''
import torch
import numpy as np
def make_one_hot(mask,num):
    shape = np.array(mask.shape)
    shape[1] = num
    shape = tuple(shape)
    res = torch.zeros(shape)
    res = res.scatter_(1,torch.LongTensor(mask),1)
    return res
