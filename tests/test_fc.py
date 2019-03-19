#!/usr/bin/env python

import os
import random
import torch
import unittest

from bprop import *

class FullyConnectedFunctionTest(unittest.TestCase):
    def setUp(self):
        torch.cuda.manual_seed(1234)

    def test_FullyConnectedFunctionImpl1(self):
        batch_size, hidden_size, input_size = 3, 4, 5
        fc = FullyConnectedFunctionImpl1.apply

        x = torch.randn([batch_size, input_size], dtype=torch.float64, requires_grad=True)
        weight = torch.randn([hidden_size, input_size], dtype=torch.float64, requires_grad=True)
        bias = torch.randn([hidden_size, 1], dtype=torch.float64, requires_grad=True)

        self.assertTrue(torch.autograd.gradcheck(fc, (weight, bias, x)))

if __name__ == '__main__':
    script_path = os.path.dirname(os.path.realpath(__file__))
    unittest.main()

