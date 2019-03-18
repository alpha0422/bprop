#!/usr/bin/env python

import torch

class SigmoidFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        y = 1 / (1 + torch.exp(-x))
        ctx.save_for_backward(y)
        return y

    @staticmethod
    def backward(ctx, grad_y):
        y, = ctx.saved_tensors
        grad_x = torch.mul(torch.mul(y, 1-y), grad_y)
        return grad_x

