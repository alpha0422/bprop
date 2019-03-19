#!/usr/bin/env python

import torch
import itertools

class FullyConnectedFunctionImpl1(torch.autograd.Function):
    @staticmethod
    def forward(ctx, weight, bias, inputs):
        N, I = inputs.size()
        H = weight.size(0)
        outputs = inputs.new_zeros((N, H))

        for i, j in itertools.product(range(N), range(H)):
            for k in range(I):
                outputs[i, j] += weight[j, k] * inputs[i, k]
            outputs[i, j] += bias[j, 0]

        ctx.save_for_backward(inputs, weight)
        return outputs

    @staticmethod
    def backward(ctx, grad_outputs):
        inputs, weight = ctx.saved_tensors
        N, I = inputs.size()
        H = weight.size(0)

        grad_inputs = torch.zeros_like(inputs)
        grad_weight = torch.zeros_like(weight)
        grad_bias = weight.new_zeros((H, 1))

        for i, j in itertools.product(range(N), range(H)):
            for k in range(I):
                grad_inputs[i, k] += weight[j, k] * grad_outputs[i, j]
                grad_weight[j, k] += grad_outputs[i, j] * inputs[i, k]
            grad_bias[j, 0] += grad_outputs[i, j]

        return grad_weight, grad_bias, grad_inputs

