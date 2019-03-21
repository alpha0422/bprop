#!/usr/bin/env python

import os
import random
import torch
import unittest

from bprop import *

class Convolution2DFunctionTest(unittest.TestCase):
    def setUp(self):
        torch.cuda.manual_seed(1234)

    def test_Convolution2DFunctionImpl1_fprop(self):
        N, C, H, W = 3, 2, 7, 5
        K, R, S = 2, 3, 2
        U, V = 1, 3
        PH, PW = 2, 3
        DH, DW = 2, 1

        conv = Convolution2DFunctionImpl1.apply

        x = torch.randn([N,C,H,W], dtype=torch.float64, requires_grad=True)
        filters = torch.randn([K,C,R,S], dtype=torch.float64, requires_grad=True)
        bias = torch.randn([K], dtype=torch.float64, requires_grad=True)
   
        ref_y = torch.nn.functional.conv2d(x, filters, bias, (U,V), (PH,PW), (DH,DW))
        tst_y = conv(x, filters, bias, (U,V), (PH,PW), (DH,DW))

        self.assertTrue(torch.allclose(ref_y, tst_y))

    def test_Convolution2DFunctionImpl1_bprop(self):
        N, C, H, W = 4, 3, 5, 6
        K, R, S = 3, 2, 3
        U, V = 2, 1
        PH, PW = 2, 0
        DH, DW = 1, 2

        conv = Convolution2DFunctionImpl1.apply

        x = torch.randn([N,C,H,W], dtype=torch.float64, requires_grad=True)
        filters = torch.randn([K,C,R,S], dtype=torch.float64, requires_grad=True)
        bias = torch.randn([K], dtype=torch.float64, requires_grad=True)
   
        self.assertTrue(torch.autograd.gradcheck(conv, (x, filters, bias, (U,V), (PH,PW), (DH,DW))))

if __name__ == '__main__':
    script_path = os.path.dirname(os.path.realpath(__file__))
    unittest.main()

