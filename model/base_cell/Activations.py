'''
Function:
    build activation
    for A variety of activation
Author:
    W.H
'''

import torch
import torch.nn as nn

def selectActivation(activation_type):
    supported_activations = {
        'relu': nn.ReLU,
        'gelu': nn.GELU,
        'relu6': nn.ReLU6,
        'sigmoid': nn.Sigmoid,
        'leakyrelu': nn.LeakyReLU,
    }
    assert activation_type in supported_activations, 'unsupport activation type %s...' % activation_type
    return supported_activations[activation_type]
