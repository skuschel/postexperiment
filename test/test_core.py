#!/usr/bin/env python

'''
Stephan Kuschel, 2018
'''

import unittest
import numpy as np
import postexperiment as pe
import pickle


class TestShot(unittest.TestCase):

    def setUp(self):
        self.a = dict(id=0, a=1, b=2, c=3)
        self.sa = pe.Shot(self.a)

    def test_init(self):
        s = pe.Shot(self.a)
        self.assertEqual(len(s), 4)
        # unknwon content should be ignored
        s['d'] = 'None'
        self.assertEqual(len(s), 4)

    def test_init_unknown(self):
        d = dict(id=0, a=1, b=2, c=3, d='unknown')
        s = pe.Shot(**d)
        # print(dict(s))
        self.assertEqual(len(s), 4)
        s = pe.Shot(d)
        # print(dict(s))
        self.assertEqual(len(s), 4)

    def test_unknown(self):
        self.sa.update(d='unknown')
        self.assertEqual(len(self.sa), 4)

    def test_pickle(self):
        f_string = pickle.dumps(self.sa)
        sa = pickle.loads(f_string)
        self.assertTrue(isinstance(sa, pe.Shot))
        self.assertEqual(dict(sa), self.a)

    def test_LazyAccess(self):
        self.sa.update(x=pe.LazyAccessDummy(42))
        arr = self.sa['x']
        self.assertTrue(isinstance(arr, np.ndarray))
        la = self.sa._mapping['x']
        self.assertTrue(isinstance(la, pe.LazyAccess))
        print(la)
    def test_call(self):
        # make sure non-used LazyAccess doesnt get expanded
        self.sa.update(x=pe.LazyAccessDummy(42, exceptonaccess=True))
        self.sa.update(y=pe.LazyAccessDummy(42))
        self.assertEqual(self.sa('np.all(y/y)'), 1)
        self.assertEqual(self.sa['a'], self.sa('a'))
        self.assertEqual(self.sa['a'] + self.sa['c'], self.sa('a + c'))
        pe.Shot.register_diagnostic(stupiddiag)
        self.assertEqual(self.sa.stupiddiag(), self.sa('stupiddiag()'))
        def test():
            self.sa('a + x')
        self.assertRaises(pe.datasources.lazyaccess._LazyAccessException, test)

    def test_alias(self):
        self.sa.updatealias({'test': 'ab', 'ab':'a'})
        self.assertEqual(self.sa('test'), 1)
        self.assertEqual(self.sa('(test, b+c)'), (1, 5))

    def test_pickle_LazyAccess(self):
        self.sa.update(x=pe.LazyAccessDummy(42, exceptonaccess=True))
        # The following methods must not use Shot.__getitem__
        print('x' in self.sa)
        # try pickling
        # pickle uses the objects __slots__ or __dict__ to access the data.
        # threfore Shot.__getitem__ is automatically bypassed and only the LazyAccess
        # reference objects are saved to disk, just as intended.
        f_string = pickle.dumps(self.sa)
        sa = pickle.loads(f_string)
        # However, `dict(self.sa)` would expand all LazyAccess!
        self.assertTrue(isinstance(sa, pe.Shot))
        self.assertEqual(len(sa), 5)
        self.assertTrue(isinstance(sa._mapping['x'], pe.LazyAccess))

    def test_double_init(self):
        self.sa.update(x=pe.LazyAccessDummy(42, exceptonaccess=True))
        newshot = pe.Shot(self.sa)
        self.assertTrue(isinstance(newshot, pe.Shot))
        self.assertTrue(isinstance(self.sa._mapping['x'], pe.LazyAccess))
        self.assertTrue(newshot is self.sa)

    def test_unknwoncontent(self):
        # empty Shot
        shot = pe.Shot({})
        shot['test'] = 'actual data'
        self.assertEqual(len(shot), 1)
        # data that should be ignored
        for unknown in pe.Shot.unknowncontent:
            shot['test'] = unknown
        # double checking
        shot['test'] = []
        shot['test'] = ()
        shot['test'] = 'unknown'
        # numpy array cases require special care
        for unknown in pe.Shot.unknowncontent:
            shot['test'] = np.array(unknown)
            shot['test'] = np.array([unknown])
        #shot['test'] = np.array([])
        #shot['test'] = np.array(np.nan)

    def test_diagnostic(self):
        pe.Shot.register_diagnostic(stupiddiag)
        self.assertEqual(self.sa.stupiddiag(), 4)
        # diagnostic must be pickable for parallel computing
        import pickle
        s = pickle.dumps(pe.Shot.diagnostics)


def stupiddiag(shot):
    return shot['a'] + shot['c']


class TestDiagnostic(unittest.TestCase):

    def setUp(self):
        self.a = dict(id=0, a=1, b=2, c=3)
        self.sa = pe.Shot(self.a)

    def test_double_init(self):
        pe.Shot.register_diagnostic(stupiddiag)
        self.assertEqual(self.sa.stupiddiag(), 4)
        d = pe.Diagnostic(stupiddiag)
        d2 = pe.Diagnostic(d)
        self.assertTrue(d is d2)
        pe.Shot.register_diagnostic(d)
        self.assertEqual(self.sa.stupiddiag(), 4)


class TestShotSeries(unittest.TestCase):

    def setUp(self):
        # create a dummy Shotlist
        self.shotlist = [self.createshot(i) for i in range(100)]
        ss = pe.ShotSeries(('id', int))
        ss.merge(self.shotlist)
        self.shotseries = ss

    def createshot(self, i):
        return pe.Shot(id=i, a=i + 1, b=i - 1)

    def test_init(self):
        self.assertEqual(len(self.shotseries), 100)

    def test_merge(self):
        ss = self.shotseries
        ss.merge([self.createshot(5)])
        self.assertEqual(len(self.shotseries), 100)
        ss.merge([self.createshot(500)])
        self.assertEqual(len(self.shotseries), 101)
        s = pe.Shot(id=50, a=50)

        def test():
            ss.merge([s])
        self.assertRaises(ValueError, test)

    def test_doublemerge(self):
        n = len(self.shotseries)
        self.shotseries.merge(self.shotseries)
        self.assertEqual(n, len(self.shotseries))

    def test_pickle(self):
        import pickle
        #print(list(self.shotseries._shots.values()))
        ds = pickle.dumps(self.shotseries)
        shots = pickle.loads(ds)
        self.assertEqual(shots[5], self.shotseries[5])

    def test_call(self):
        data = list(self.shotseries('id'))
        self.assertEqual(data, list(range(100)))

    def test_call_missing(self):
        nn = [33, 14, 92, 68, 38, 75, 74, 58, 41, 99]
        for n in nn:
            shot = self.shotseries[n]
            shot['sometimes_there'] = 86
        data = list(self.shotseries('sometimes_there + 5'))
        self.assertEqual(data, [91]*10)

    def test_filter(self):
        shots = self.shotseries.filter('id > 50')
        self.assertEqual(len(shots), 49)

if __name__ == '__main__':
    unittest.main()
