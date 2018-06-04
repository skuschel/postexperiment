#!/usr/bin/env python

'''
Stephan Kuschel, 2018
'''

import unittest
import numpy as np
import ped

class TestShot(unittest.TestCase):

    def setUp(self):
        self.a = dict(id=0, a=1, b=2, c=3)
        self.sa = ped.Shot(self.a)
        self.b = dict(id=2, a=1, b=2, c=3)
        self.c = dict(id=3, a='1', b='2', c='3')

    def test_init(self):
        s = ped.Shot(self.a)
        self.assertEqual(len(s), 4)
        # unknwon content should be ignored
        s['d'] = 'None'
        self.assertEqual(len(s), 4)

    def test_unknown(self):
        self.sa.update(d='unknown')
        self.assertEqual(len(self.sa), 4)











if __name__ == '__main__':
    unittest.main()
