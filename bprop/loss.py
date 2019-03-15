#!/usr/bin/env python

import torch

class SoftmaxCrossEntropyWithLogitsFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, logits, onehot_labels):
        exp_logits = torch.exp(logits)
        prob = exp_logits / torch.sum(exp_logits, dim=1, keepdim=True)
        loss = - torch.sum(onehot_labels * torch.log(prob))
        ctx.save_for_backward(prob, onehot_labels)

        return loss

    @staticmethod
    def backward(ctx, grad_output):
        prob, onehot_labels = ctx.saved_tensors

        grad_logits = torch.mul(prob - onehot_labels, grad_output)
        return grad_logits, None

