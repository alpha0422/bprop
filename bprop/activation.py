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

class TanhFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        y = torch.div(torch.exp(x)-torch.exp(-x), torch.exp(x)+torch.exp(-x))
        ctx.save_for_backward(y)
        return y

    @staticmethod
    def backward(ctx, grad_y):
        y, = ctx.saved_tensors
        grad_x = torch.mul(1 - torch.pow(y, 2), grad_y)
        return grad_x

class ReluFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        mask = (x < 0)
        y = torch.mul(x, mask.type_as(x))
        ctx.save_for_backward(mask)
        return y

    @staticmethod
    def backward(ctx, grad_y):
        mask, = ctx.saved_tensors
        grad_x = torch.mul(grad_y, mask.type_as(grad_y))
        return grad_x

