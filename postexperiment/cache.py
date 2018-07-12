'''
decorators for function caching.

Stephan Kuschel, 2018
'''

import os
import sys
import functools
import time
import pickle

__all__ = ['permanentcachedecorator']


class permanentcachedecorator():
    '''
    The permanent cache for a function.

    Stephan Kuschel, 2018
    '''

    def __init__(self, file, ShotId):
        '''
        returns a decorater.

        args
        ----
          file: str
            The filename-prefix for the cachefile.
          ShotId: callable
            A callable mapping from a `Shot` to a hasable object to identify
            identical shots, even between python sessions!
        '''
        self.file = file
        self.ShotId = ShotId

    def __call__(self, function):
        ret = _PermanentCache(self.file, self.ShotId, function)
        return ret


class _PermanentCache():
    '''
    A permanent cache for a function.

    Stephan Kuschel, 2018
    '''
    _filelock = dict()

    @classmethod
    def __del__(cls):
        print('autosaving postexperiment.permanentcachedecorator...')
        for _, c in cls._filelock:
            print('autosaving: {}'.format(c))
            c.save()

    def __new__(cls, file, ShotId, function, **kwargs):
        absfile = os.path.abspath('{}_{}.cache'.format(file, str(function.__name__)))
        if absfile in cls._filelock:
            s = '''
                replacing an already cached function by "{}".
                Remember to clear the cache in case its definition has changed.
                '''
            print(s.format(function))
            return cls._filelock[absfile]
        ret = super().__new__(cls)
        cls._filelock[absfile] = ret
        return ret

    def __init__(self, file, ShotId, function, maxsize=250, load=True):
        functools.update_wrapper(self, function)
        self.file = os.path.abspath('{}_{}.cache'.format(file, str(function.__name__)))
        self._maxsize = maxsize
        self.ShotId = ShotId
        self.function = function
        # load data
        if load and os.path.isfile(self.file):
            self.load()
        else:
            self.clearcache()

    def __call__(self, shot, **kwargs):
        shotid = self.ShotId(shot)
        idx = (shotid, tuple(sorted(kwargs)))
        try:
            ret = self.cache[idx]
            self.hits += 1
        except(KeyError):
            t0 = time.time()
            ret = self.function(shot, **kwargs)
            self.exectime = time.time() - t0
            if self._maxsize is None or sys.getsizeof(ret) <= self._maxsize:
                self.cache[idx] = ret
        return ret

    def clearcache(self):
        self.cache = dict()
        self.hits = 0
        self.exectime = 0

    def save(self):
        with open(self.file, 'wb') as f:
            pickle.dump((self.exectime, self.cache), f)
        print('"{}" ({} entries) saved.'.format(self.file, len(self)))

    def load(self):
        s = 'loading {:.1f} MB from {}'
        size = os.path.getsize(self.file) / 1e6
        print(s.format(size, self.file))
        with open(self.file, 'rb') as f:
            self.exectime, self.cache = pickle.load(f)
        self.hits = 0

    def __len__(self):
        return len(self.cache)

    def __str__(self):
        s = '<Cache of "{}" ({} entries, {} hits = {:.1f}s saved)>'
        return s.format(self.__name__, len(self), self.hits, self.hits*self.exectime)

    __repr__ = __str__
