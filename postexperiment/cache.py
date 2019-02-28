#
# This file is part of postexperiment.
#
# postexperiment is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# postexperiment is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with postexperiment. If not, see <http://www.gnu.org/licenses/>.
'''
decorators for function caching.

Stephan Kuschel, 2018-2019
'''

import os
import sys
import functools
import time
import pickle
import socket
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

    def gcall(self):
        '''
        Garbage Collect all cache files.
        '''
        _PermanentCache.gcall()

    def reloadall(self):
        _PermanentCache.reloadall()

    def collectcachenew(self):
        '''
        This function is meant to be used in parallel mode to sync all new updates over the
        network with the parent process.
        returns a dictionary containing all updates of all caches.
        This dictionary can be readded by using `self.mergecachenew(updates)`.
        '''
        ret = {c.function.__name__: c.cachenew for _, c in _PermanentCache._filelock.items()}
        return ret

    def mergecachenew(self, updates):
        '''
        This function is meant to be used in parallel mode to sync all new updates over the
        network with the parent process.
        '''
        namedict = self.collectcachenew()
        for name, data in updates.items():
            # this can be used for writing, since `self.collectcachenew()` only returns a pointer
            # to the current dictionary `self.cachenew`
            namedict[name].update(data)

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
    def saveall(cls):
        print('autosaving postexperiment.permanentcachedecorator...')
        for _, c in cls._filelock.items():
            print('autosaving: {}'.format(c))
            c.save()

    @classmethod
    def gcall(cls):
        print('Garbage collecting postexperiment.permanentcachedecorator...')
        for _, c in cls._filelock.items():
            print('gc: {}'.format(c))
            c.gc()

    @classmethod
    def reloadall(cls):
        for _, c in cls._filelock.items():
            c.save()
            c.load()

    @staticmethod
    def _absfile(name, functionname):
        '''
        creates absolute file name from a given cache-filename
        '''
        template = '{name}_{functionname}.cache-{host}-{pid}'
        hostname = socket.gethostname()
        pid = os.getpid()
        file = template.format(name=name, functionname=functionname, host=hostname, pid=pid)
        fileglob = template.format(name=name, functionname=functionname, host='*', pid='*')
        return os.path.abspath(file), os.path.abspath(fileglob)

    def __new__(cls, file, ShotId, function, **kwargs):
        absfile, _ = cls._absfile(file, function.__name__)
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
        self.file, self.globfile = self._absfile(file, function.__name__)
        self._maxsize = maxsize
        self.ShotId = ShotId
        self.function = function
        self.clearcache()
        # load data
        if load:
            self.load()

    def __del__(self):
        self.save()

    def __getitem__(self, key):
        '''
        the cache access.
        '''
        if key in self.cache:
            ret = self.cache[key]
        else:
            ret = self.cachenew[key]
        self.hits += 1
        return ret

    def __setitem__(self, key, val):
        self.cachenew[key] = val

    def __call__(self, shot, **kwargs):
        shotid = self.ShotId(shot)
        idx = (shotid, tuple(sorted(kwargs.items())))  # this contains keys and values of kwargs
        idxold = (shotid, tuple(sorted(kwargs)))  # this one contains only the keys of kwargs
        try:
            try:
                ret = self[idx]
            except(KeyError):
                ret = self[idxold]
        except(KeyError):
            t0 = time.time()
            ret = self.function(shot, **kwargs)
            self.exectime = time.time() - t0
            if self._maxsize is None or sys.getsizeof(ret) <= self._maxsize:
                self[idx] = ret
        return ret

    @property
    def exectime(self):
        '''
        contains the average execution time per call.
        '''
        return self._exectime

    @exectime.setter
    def exectime(self, val):
        # running average
        self._exectime = (self.exectime * self.n_exec + val) / (self.n_exec + 1)
        self.n_exec += 1

    def clearcache(self):
        # cache will be populated by data read from disc
        self.cache = dict()
        # cachenew will be populated by new function executions
        # during runtime
        self.cachenew = dict()
        self.hits = 0
        self._exectime = 0
        self.n_exec = 0

    def save(self):
        '''
        the function returns the filename, which has actually been used for saving.
        '''
        if len(self.cachenew) == 0:
            # there is no new data, which would require saving.
            return None
        for i in range(100):
            nextfile = '{}-{}'.format(self.file, i)
            if not os.path.isfile(nextfile):
                break
        else:
            print('autorun Garbage Collection...')
            # gc starts this routine again after deleting files.
            return self.gc()
        with open(nextfile, 'wb') as f:
            pickle.dump((self.exectime, self.cachenew), f)
        print('"{}" ({} entries) saved.'.format(nextfile, len(self.cachenew)))
        self.cache.update(self.cachenew)
        self.cachenew = {}
        return nextfile

    @staticmethod
    def _loaddata(file):
        s = 'loading {:.1f} MB from {}'
        size = os.path.getsize(file) / 1e6
        print(s.format(size, file))
        with open(file, 'rb') as f:
            exectime, cache = pickle.load(f)
        return exectime, cache

    def _loadalldata(self):
        files = glob.glob(self.globfile)
        cache = {}
        exectime = 0
        for file in files:
            exectime, c = self._loaddata(file)
            cache.update(c)
        return exectime, cache, files

    def load(self):
        self.exectime, self.cache, _ = self._loadalldata()
        self.hits = 0

    def gc(self, delete=True):
        '''
        Merge existing data files and save current data.
        '''
        _, cache, files = self._loadalldata()
        nextfile = self.file + '-gc'
        with open(nextfile, 'wb') as f:
            pickle.dump((self.exectime, self.cache), f)
        for file in files:
            os.remove(file)
        return self.save()

    def __len__(self):
        return len(self.cache) + len(self.cachenew)

    def __str__(self):
        if len(self.cachenew) == 0:
            s = '<Cache of "{}" ({} entries, {} hits = {:.1f}s saved)>'
            ret = s.format(self.__name__, len(self), self.hits, self.hits*self.exectime)
        else:
            s = '<Cache of "{}" ({} entries ({} new(!)), {} hits = {:.1f}s saved)>'
            ret = s.format(self.__name__, len(self), len(self.cachenew),
                           self.hits, self.hits*self.exectime)
        return ret

    __repr__ = __str__
