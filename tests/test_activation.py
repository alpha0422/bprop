#!/usr/bin/env python

import os
import random
import torch
import unittest

from bprop import *

class ActivationFunctionTest(unittest.TestCase):
    def setUp(self):
        torch.cuda.manual_seed(1234)

    def test_SigmoidFunction(self):
        batch_size, hidden_size = 16, 27
        sigmoid = SigmoidFunction.apply
        x = torch.randn([batch_size, hidden_size], device='cuda:0', dtype=torch.float64, requires_grad=True)
    
        self.assertTrue(torch.autograd.gradcheck(sigmoid, (x,)))

    def test_TanhFunction(self):
        batch_size, hidden_size = 16, 27
        tanh = TanhFunction.apply
        x = torch.randn([batch_size, hidden_size], device='cuda:0', dtype=torch.float64, requires_grad=True)
   
        self.assertTrue(torch.autograd.gradcheck(tanh, (x,)))

    def test_ReluFunction(self):
        batch_size, hidden_size = 16, 27
        relu = ReluFunction.apply
        x = torch.randn([batch_size, hidden_size], device='cuda:0', dtype=torch.float64, requires_grad=True)
   
        self.assertTrue(torch.autograd.gradcheck(relu, (x,)))

if __name__ == '__main__':
    script_path = os.path.dirname(os.path.realpath(__file__))
    unittest.main()

