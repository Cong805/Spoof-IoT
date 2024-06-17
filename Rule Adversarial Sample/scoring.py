import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.autograd import Variable
import argparse
import math

import numpy as np

import utils
import sys
# Input: 
# model: the torch model
# input: the input at current stage 
#        Torch tensor with size (Batchsize,length)
# Output: score with size (batchsize, length)

#用于对语句中的词进行重要性得分的计算

def calculate_conf_batch(model, inputs, pred, vocab_length):
    #置信度方法
    # print(len(inputs[0]))
    input_lens = inputs[1].clone()

    losses = torch.zeros(inputs[0].size()[0], inputs[0].size()[1])
    for i in range(inputs[0].size()[1]):
        tempinputs = inputs[0].clone()
        tempinputs[:, i] = vocab_length
        Temp = (tempinputs, input_lens)
        with torch.no_grad():
            tempoutput = model(Temp)
        losses[:, i] = F.nll_loss(tempoutput, pred, reduction='none')

    return losses


def grad_batch(model, inputs, pred, vocab_length):
    #梯度方法

    # losses1 = torch.zeros(inputs.size()[0], inputs.size()[1])
    # dloss = torch.zeros(inputs.size()[0], inputs.size()[1])
    if isinstance(model, torch.nn.DataParallel):
        model = model.module
    # model.train()
    embd, output = model(inputs, returnembd=True)
    # embd.retain_grad()

    loss = F.nll_loss(output, pred)
    # loss = F.cross_entropy(output, pred)
    loss.backward()
    score = (inputs[0].squeeze(dim=0) >= vocab_length).float()
    # print(score)
    score = -score
    score = embd.grad.norm(2, dim=2) + score * 1e9

    return score

def random(model, inputs, pred, classes):
    #忽略，随机择词，一开始做对照的后期可能会删
    losses = torch.rand(inputs.size()[0],inputs.size()[1])
    return losses
    # Output a random list

#下面三个函数是deepwordbug中提出的方法
def temporal(model, inputs, pred):

    losses1 = torch.zeros(inputs[0].size()[0], inputs[0].size()[1])
    dloss = torch.zeros(inputs[0].size()[0], inputs[0].size()[1])
    input_lens = inputs[1].clone()
    # print(inputs)
    for i in range(inputs[0].size()[1]):
        tempinputs = inputs[0].clone()
        tempinputs[:,i+1:] = 50000

        Temp = (tempinputs, input_lens)
        with torch.no_grad():
            tempoutput = model(Temp)
        losses1[:, i] = tempoutput.gather(1, pred.view(-1, 1)).view(-1)

    dloss[:, 0] = losses1[:, 0] - 1.0 / 6
    for i in range(1, inputs[0].size()[1]):
        dloss[:, i] = losses1[:, i] - losses1[:, i - 1]
    return dloss


def temporaltail(model, inputs, pred):
    losses1 = torch.zeros(inputs[0].size()[0], inputs[0].size()[1])
    dloss = torch.zeros(inputs[0].size()[0], inputs[0].size()[1])
    input_lens = inputs[1].clone()
    for i in range(inputs[0].size()[1]):
        tempinputs = inputs[0].clone()
        tempinputs[:, :i] = 50000
        Temp = (tempinputs, input_lens)
        with torch.no_grad():
            tempoutput = model(Temp)
        losses1[:, i] = tempoutput.gather(1, pred.view(-1, 1)).view(-1)

    dloss[:, -1] = losses1[:, -1] - 1.0 / 6
    for i in range(inputs[0].size()[1] - 1):
        dloss[:, i] = losses1[:, i] - losses1[:, i + 1]
    return dloss


def combined(model, inputs, pred):
    temp = temporal(model, inputs, pred)
    temptail = temporaltail(model, inputs, pred)
    return (temp + temptail) / 2




    

