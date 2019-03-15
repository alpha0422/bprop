#!/usr/bin/env python

import os
import random
import torch
import unittest

from bprop import *

class LossFunctionTest(unittest.TestCase):
    def setUp(self):
        torch.cuda.manual_seed(1234)

    def test_SoftmaxCrossEntropyWithLogitsFunction(self):
        batch_size, num_classes = 16, 32
        loss = SoftmaxCrossEntropyWithLogitsFunction.apply
        x = torch.randn([batch_size, num_classes], device='cuda:0', dtype=torch.float64, requires_grad=True)
        target = torch.randint(0, num_classes, [batch_size], device='cuda:0', requires_grad=False)
        if hasattr(torch.nn.functional, 'one_hot'):
            onehot_labels = torch.nn.functional.one_hot(target, num_classes=num_classes)
        else:
            onehot_labels = torch.zeros_like(x, requires_grad=False)
            onehot_labels.scatter_(1, target.view(-1, 1), 1)
    
        self.assertTrue(torch.autograd.gradcheck(loss, (x, onehot_labels)))

if __name__ == '__main__':
    script_path = os.path.dirname(os.path.realpath(__file__))
    unittest.main()

