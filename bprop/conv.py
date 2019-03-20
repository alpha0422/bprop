#!/usr/bin/env python

import torch
import itertools

class Convolution2DFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inputs, filters, bias, strides=(1,1), padding=(0,0)):
        assert inputs.size(1) == filters.size(1), "Input channel mismatch!"
        assert filters.size(0) == bias.size(0), "Output channel mismatch!"
        
        N, C, H, W = inputs.shape
        K, C, R, S = filters.shape
        U, V = strides
        PH, PW = padding

        P = (H + 2 * PH - R) // U + 1
        Q = (W + 2 * PW - S) // V + 1

        pinputs = inputs.new_zeros((N, C, H + 2 * PH, W + 2 * PW))
        outputs = inputs.new_zeros((N, K, P, Q))

        pinputs[:,:,PH:H+PH,PW:W+PW] = inputs  # PH:-PH is wrong if PH is 0

        for n,k,p,q in itertools.product(range(N),range(K),range(P),range(Q)):
            outputs[n,k,p,q] = torch.sum(pinputs[n,:,p*U:p*U+R,q*V:q*V+S] * filters[k,...]) + bias[k]

        ctx.save_for_backward(inputs, filters, bias, torch.IntTensor(strides),
            torch.IntTensor(padding))
        return outputs

    @staticmethod
    def backward(ctx, gO):
        inputs, filters, bias, strides, padding = ctx.saved_tensors

        N, C, H, W = inputs.shape
        K, C, R, S = filters.shape
        U, V = strides
        PH, PW = padding

        P = (H + 2 * PH - R) // U + 1
        Q = (W + 2 * PW - S) // V + 1

        pinputs = inputs.new_zeros((N, C, H + 2 * PH, W + 2 * PW))
        gI = torch.zeros_like(pinputs)
        gW = torch.zeros_like(filters)
        gb = torch.zeros_like(bias)

        pinputs[:,:,PH:H+PH,PW:W+PW] = inputs

        for n,k,p,q in itertools.product(range(N),range(K),range(P),range(Q)):
            gI[n,:,p*U:p*U+R,q*V:q*V+S] += filters[k,...] * gO[n,k,p,q]
            gW[k,...] += gO[n,k,p,q] * pinputs[n,:,p*U:p*U+R,q*V:q*V+S]
            gb[k] += gO[n,k,p,q]

        return gI[:,:,PH:H+PH,PW:W+PW], gW, gb, None, None

