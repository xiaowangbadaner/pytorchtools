'''
Function:
    build attentionlayers
    for A variety of attentionlayers
Author:
    W.H
'''
import torch
import torch.nn as nn
from Attentionlayers import *
from Activations import *
from loss_function import *
from Normalizations import *

def selectAtten(activation_type):
    supported_activations = {
        'SEblock': SElayer
    }
    assert activation_type in supported_activations, 'unsupport activation type %s...' % activation_type
    return supported_activations[activation_type]

def selectActivation(activation_type):
    '''
    I was think that whether add the nn's activation on there
    '''
    supported_activations = {
        'relu': nn.ReLU,
        'gelu': nn.GELU,
        'relu6': nn.ReLU6,
        'sigmoid': nn.Sigmoid,
        'leakyrelu': nn.LeakyReLU,
    }
    assert activation_type in supported_activations, 'unsupport activation type %s...' % activation_type
    return supported_activations[activation_type]

# x = torch.ones((5,12,12,12))
# net = selectAtten('SEblock')(12,6,'linear')
# y = net(x)
# print(y)