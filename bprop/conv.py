#!/usr/bin/env python

import torch
import itertools

class Convolution2DFunctionImpl1(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inputs, filters, bias, strides=(1,1), padding=(0,0), dilation=(1,1)):
        def check(tup, lb):
            if len(tup) == 2 and isinstance(tup[0], int) and isinstance(tup[1], int) \
                and tup[0] >= lb and tup[1] >= lb:
                return True
            else:
                return False

        assert inputs.size(1) == filters.size(1), "Input channel mismatch!"
        assert filters.size(0) == bias.size(0), "Output channel mismatch!"
        assert check(strides, 1), "Invalid strides!"
        assert check(padding, 0), "Invalid padding!"
        assert check(dilation, 1), "Invalid dilation!"
        
        N, C, H, W = inputs.shape
        K, C, R, S = filters.shape
        U, V = strides
        PH, PW = padding
        DH, DW = dilation
        DR = DH*(R-1) + 1
        DS = DW*(S-1) + 1

        P = (H + 2*PH - DH*(R-1) - 1) // U + 1
        Q = (W + 2*PW - DW*(S-1) - 1) // V + 1

        pinputs = inputs.new_zeros((N, C, H + 2 * PH, W + 2 * PW))
        outputs = inputs.new_zeros((N, K, P, Q))

        pinputs[:,:,PH:H+PH,PW:W+PW] = inputs  # PH:-PH is wrong if PH is 0

        for n,k,p,q in itertools.product(range(N),range(K),range(P),range(Q)):
            outputs[n,k,p,q] = torch.sum(pinputs[n,:,p*U:p*U+DR:DH,q*V:q*V+DS:DW] * filters[k,...]) + bias[k]

        ctx.save_for_backward(inputs, filters, bias, torch.IntTensor(strides),
            torch.IntTensor(padding), torch.IntTensor(dilation))
        return outputs

    @staticmethod
    def backward(ctx, gO):
        inputs, filters, bias, strides, padding, dilation = ctx.saved_tensors

        N, C, H, W = inputs.shape
        K, C, R, S = filters.shape
        U, V = strides
        PH, PW = padding
        DH, DW = dilation
        DR = DH*(R-1) + 1
        DS = DW*(S-1) + 1

        P = (H + 2*PH - DH*(R-1) - 1) // U + 1
        Q = (W + 2*PW - DW*(S-1) - 1) // V + 1

        pinputs = inputs.new_zeros((N, C, H + 2 * PH, W + 2 * PW))
        gI = torch.zeros_like(pinputs)
        gW = torch.zeros_like(filters)
        gb = torch.zeros_like(bias)

        pinputs[:,:,PH:H+PH,PW:W+PW] = inputs

        for n,k,p,q in itertools.product(range(N),range(K),range(P),range(Q)):
            gI[n,:,p*U:p*U+DR:DH,q*V:q*V+DS:DW] += filters[k,...] * gO[n,k,p,q]
            gW[k,...] += gO[n,k,p,q] * pinputs[n,:,p*U:p*U+DR:DH,q*V:q*V+DS:DW]
            gb[k] += gO[n,k,p,q]

        return gI[:,:,PH:H+PH,PW:W+PW], gW, gb, None, None, None

