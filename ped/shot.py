'''
Copyright:
Alexander Blinne, 2018
Stephan Kuschel, 2018
'''


import copy
import collections
import itertools
import os
import os.path as osp
import re
import concurrent.futures as cf

import numpy as np

from . import common
from . import labbook

class Shot(dict):
    '''
    The Shot class representing a single shot or event on the experiment.

    Copyright:
    Alexander Blinne, 2018
    Stephan Kuschel, 2018
    '''
    diagnostics = dict()
    unknowncontent = [None, '', ' ', 'None', 'unknown', '?', 'NA']

    def __getattr__(self, key):
        if key.startswith('_'):
            raise AttributeError
        def call(*args, context=None, **kwargs):
            if context is None:
                context = common.DefaultContext()
            context['shot'] = self
            return self.diagnostics[key](self, *args, context=context, **kwargs)
        return call

    def __setitem__(self, key, val):
        if key in self:
            if val in self.unknowncontent:
                # do not change anythining if new info is not actually real info
                return
            if self[key] not in self.unknowncontent and str(self[key]) != str(val):
                s = '''
                    Once assigned, shots cannot be changed. If you have
                    multiple data sources, their information must match.
                    You are attempting to reassign the key "{}"
                    from "{}" to "{}"
                    on Shot "{}".
                    '''
                raise ValueError(s.format(key, self[key], val, self))
        # assign value if all sanity checks passed
        super().__setitem__(key, val)

    def update(self, *args, **kwargs):
        # build in update function does not call __setitem__
        if len(args) > 1:
            raise TypeError("update expected at most 1 arguments, got %d" % len(args))
        other = dict(*args, **kwargs)
        for key in other:
            self[key] = other[key]

    def __hash__(self):
        return id(self)


def make_shotid(*shot_id_fields):
    shot_id_fields = collections.OrderedDict(shot_id_fields)

    PlainShotId = collections.namedtuple('ShotId', shot_id_fields.keys())

    class ShotId(PlainShotId):
        def __new__(cls, shot):
            plain_shot_id = PlainShotId(**{k: v for k, v in shot.items() if k in shot_id_fields.keys()})
            vals = [conv(val) for conv, val in zip(shot_id_fields.values(), plain_shot_id)]
            return super().__new__(cls, *vals)

        @classmethod
        def literal(cls, *vals):
            return super().__new__(cls, *vals)

    return ShotId


class ShotSeries(object):
    def __init__(self, *shot_id_fields):
        '''
        Data must be a list of dictionaries or None.
        '''
        self.ShotId = make_shotid(*shot_id_fields)
        self._shots = collections.OrderedDict()
        self.sources = dict()

    def __copy__(self):
        newone = type(self)()
        newone.__dict__.update(self.__dict__)
        newone._shots = copy.copy(self.shots)
        return newone

    @classmethod
    def empty_like(cls, other):
        newone = type(other)()
        newone.__dict__.update(other.__dict__)
        newone._shots = collections.OrderedDict()
        return newone

    def load(self):
        """
        Loads shots from all attached sources
        """
        for source in self.sources.values():
            self.merge(source())

        return self

    def merge(self, shotlist):
        '''
        merges a shotlist into the current ShotSeries `self` and
        and combines the provided information. Shots are considered identical if
        ALL shot_id_fields given by `shot_id_fields` are equal. Both ShotSeries
        MUST have all `shot_id_fields` present.
        '''
        for datadict in shotlist:
            shotid = self.ShotId(datadict)
            if shotid in self._shots:
                self._shots[shotid].update(datadict)
            else:
                # add entirely new the data and enusure data is a Shot object
                shot = datadict if isinstance(datadict, Shot) else Shot(datadict)
                self._shots[shotid] = shot

        self._shots = collections.OrderedDict(sorted(self._shots.items(), key=lambda item: item[0]))

        return self

    def __iter__(self):
        return iter(self._shots.values())

    def __reversed__(self):
        return reversed(self._shots.values())

    def __getitem__(self, key):
        if isinstance(key, int):
            if key < 0:
                key = len(self)+key
            return next(itertools.islice(self._shots.values(), key, None))

        elif isinstance(key, slice):
            shots = self._shots.values()

            start, stop, step = key.start, key.stop, key.step

            if start and start < 0:
                start = len(self)+start

            if stop and stop < 0:
                stop = len(self)+stop

            if step and step < 0:
                shots = reversed(shots)
                step = -step
                if start:
                    start = len(self)-1-start
                if stop:
                    stop = len(self)-1-stop

            return list(itertools.islice(shots, start, stop, step))

        else:
            return self._shots[key]

    def __contains__(self, key):
        return key in self._shots

    def __len__(self):
        return len(self._shots)

    def groupby(self, *keys):
        keyfun = lambda shot: tuple(shot[key] for key in keys)
        for k, g in itertools.groupby(sorted(self, key=keyfun), key=keyfun):
            if isinstance(k, tuple) and len(k)==1:
                k = k[0]
            yield k, ShotSeries.empty_like(self).merge(g)

    def filter(self, fun):
        return ShotSeries.empty_like(self).merge(filter(fun, self))

    def filterby(self, **key_val_dict):
        fun = lambda shot: all(shot[key] == val for key, val in key_val_dict.items())
        return self.filter(fun)

    def mean(self, attr, *args, parallel=False, **kwargs):
        caller = _ShotAttributeCaller(attr, *args, **kwargs)

        if parallel:
            pool = cf.ProcessPoolExecutor()
            data = list(pool.map(caller, self))
            pool.shutdown()
        else:
            data = list(map(caller, self))

        namedtupletype = None
        if isinstance(data[0], tuple) and type(data[0]) is not tuple:
            # will get here for namedtuples (and maybe some other things but I don't care)
            namedtupletype = type(data[0])

        dd = np.stack(np.array(d) for d in data)
        dm = np.mean(dd, axis=0)

        if namedtupletype:
            return namedtupletype(*dm)

        return dm

    def grouped_mean(self, attr, keys, *args, **kwargs):
        group_id = []
        results = []
        for value, shots in self.groupby(*keys):
            group_id.append(value)
            results.append(shots.mean(attr, *args, **kwargs))
        return group_id, results


class _ShotAttributeCaller:
    def __init__(self, attr, *args, **kwargs):
        self.attr = attr
        self.args = args
        self.kwargs = kwargs

    def __call__(self, shot):
        return getattr(shot, self.attr)(*self.args, **self.kwargs)
