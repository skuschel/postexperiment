'''
decorators for function caching.

Stephan Kuschel, 2018
'''

import os
import sys
import functools
import time
import pickle
import glob

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

    def saveall(self):
        _PermanentCache.saveall()

    def __str__(self):
        caches = [str(c) for _, c in _PermanentCache._filelock.items()]
        return os.linesep.join(caches)


class _PermanentCache():
    '''
    A permanent cache for a function.

    Stephan Kuschel, 2018
    '''
    _filelock = dict()

    @classmethod
    def saveall(cls, suffix=None):
        print('autosaving postexperiment.permanentcachedecorator...')
        for _, c in cls._filelock.items():
            print('autosaving: {}'.format(c))
            c.save(suffix=suffix)

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

    def save(self, suffix=None):
        '''
        `suffix=None` is an optional suffix such that the data will
        be written to another file.
        '''
        file = self.file if suffix is None else self.file + '-' + suffix
        with open(file, 'wb') as f:
            pickle.dump((self.exectime, self.cache), f)
        print('"{}" ({} entries) saved.'.format(file, len(self)))

    @staticmethod
    def _loaddata(file):
        s = 'loading {:.1f} MB from {}'
        size = os.path.getsize(file) / 1e6
        print(s.format(size, file))
        with open(file, 'rb') as f:
            exectime, cache = pickle.load(f)
        return exectime, cache

    def load(self):
        files = []
        if os.path.isfile(self.file):
            files += self.file
        files += glob.glob(self.file + '-*')
        cache = {}
        for file in files:
            exectime, c = self._loaddata()
            cache.update(c)
        self.exectime = exectime
        self.cache = cache
        self.hits = 0

    def __len__(self):
        return len(self.cache)

    def __str__(self):
        s = '<Cache of "{}" ({} entries, {} hits = {:.1f}s saved)>'
        return s.format(self.__name__, len(self), self.hits, self.hits*self.exectime)

    __repr__ = __str__
