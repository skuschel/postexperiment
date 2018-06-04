#!/usr/bin/env python

'''
Stephan Kuschel, 2018
'''

import unittest
import numpy as np
import ped
import pickle

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

    def test_init_unknown(self):
        d = dict(id=0, a=1, b=2, c=3, d='unknown')
        s = ped.Shot(**d)
        #print(dict(s))
        self.assertEqual(len(s), 4)
        s = ped.Shot(d)
        #print(dict(s))
        self.assertEqual(len(s), 4)

    def test_unknown(self):
        self.sa.update(d='unknown')
        self.assertEqual(len(self.sa), 4)

    def test_pickle(self):
        f_string = pickle.dumps(self.sa)
        sa = pickle.loads(f_string)
        self.assertTrue(isinstance(sa, ped.Shot))
        self.assertEqual(dict(sa), self.a)


    def test_LazyAccess(self):
        self.sa.update(x=ped.LazyAccessDummy(42))
        arr = self.sa['x']
        self.assertTrue(isinstance(arr, np.ndarray))
        la = self.sa._mapping['x']
        self.assertTrue(isinstance(la, ped.LazyAccess))
        print(la)

    def test_pickle_LazyAccess(self):
        self.sa.update(x=ped.LazyAccessDummy(42, exceptonaccess=True))
        # The following methods must not use Shot.__getitem__
        print('x' in self.sa)
        # try pickling
        # pickle uses the objects __slots__ or __dict__ to access the data.
        # threfore Shot.__getitem__ is automatically bypassed and only the LazyAccess
        # reference objects are saved to disk, just as intended.
        f_string = pickle.dumps(self.sa)
        sa = pickle.loads(f_string)
        # However, `dict(self.sa)` would expand all LazyAccess!
        self.assertTrue(isinstance(sa, ped.Shot))
        self.assertEqual(len(sa), 5)
        self.assertTrue(isinstance(sa._mapping['x'], ped.LazyAccess))

    def test_double_init(self):
        self.sa.update(x=ped.LazyAccessDummy(42, exceptonaccess=True))
        newshot = ped.Shot(self.sa)
        self.assertTrue(isinstance(newshot, ped.Shot))
        self.assertTrue(isinstance(self.sa._mapping['x'], ped.LazyAccess))
        self.assertTrue(newshot is self.sa)





if __name__ == '__main__':
    unittest.main()
