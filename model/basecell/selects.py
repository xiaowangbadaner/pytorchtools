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

def selectAtten(attention_type):
    supported_attentions = {
        'SElayer': SElayer
    }
    assert attention_type in supported_attentions, 'unsupport activation type %s...' % attention_type
    return supported_attentions[attention_type]

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

def selectLoss(loss_type):
    supported_losses = {
        'DiceLoss':DiceLoss,
    }
    assert loss_type in supported_losses, 'unsupport loss type %s...' % loss_type
    return supported_losses[loss_type]

x = selectLoss('DiceLoss')()
print(x)