

import numpy as np
import abc
from future.utils import with_metaclass

__all__ = ['LazyAccess', 'LazyAccessDummy', 'LazyAccessH5']


class LazyAccess(with_metaclass(abc.ABCMeta, object)):

    @abc.abstractmethod
    def access(self, shot, key):
        '''
        The LazyAccess interface requires on the access method, which is called
        with the arguments `shot` and `key`. This allows the same LazyAccess
        object to be reused on different shots and keys. This way, the object
        can also be used for data evalutation.
        '''
        pass


class _LazyAccessException(Exception):
    '''
    Used for testing only. This is raised if a LazyAccess happens, which
    should not be happening.
    '''
    pass


class LazyAccessDummy(LazyAccess):
    '''
    used for testing purposes only. Returns random data with specified seed.
    '''

    def __init__(self, seed, exceptonaccess=False):
        self.seed = seed
        self.exceptonaccess = exceptonaccess

    def access(self, shot, key):
        if self.exceptonaccess:
            raise _LazyAccessException('Access denied.')
        print('Accessing LazyAccessDummy(seed={}) at {}'.format(self.seed, key))
        # set seed for reproducibility
        np.random.seed(self.seed)
        return np.random.rand(1000, 1700)

    def __str__(self):
        s = '<LazyAccessDummy(seed={})>'
        return s.format(self.seed)


class LazyAccessH5(LazyAccess):
    '''
    This object only stores a reference to an hdf5 file including key and index.
    The data can be accessed by calling the access method.

    Stephan Kuschel, 2018
    '''

    def __init__(self, filename, key=None, index=None):
        self.filename = filename
        self.key = key  # if given, this one has priority
        self.index = index

    def access(self, shot=None, key=None):
        '''
        The key provided here will only be used, if no key was
        already given at object initialization.
        '''
        #print('accessing')
        import h5py
        k = key if self.key is None else self.key
        h5group = h5py.File(self.filename)[k]
        return h5group if self.index is None else h5group[self.index]

    def __str__(self):
        key = 'key' if self.key is None else "'{}'".format(self.key)
        s = "<LazyAccessH5@{file}[{key}][{idx}]>"
        return s.format(file=self.filename, key=key, idx=self.index)

    __repr__ = __str__
