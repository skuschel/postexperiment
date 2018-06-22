'''
The core classes for data evaluation:
`Shot`
  is representing a single event (Laser shot, accelerator shot,..)

`ShotSeries`
  represents a collections of shots.

Copyright:
Alexander Blinne, 2018
Stephan Kuschel, 2018
'''


import copy
import collections
import collections.abc
import itertools
import os
import os.path as osp
import re
import abc
from future.utils import with_metaclass
import concurrent.futures as cf

import numpy as np

from . import common
from .datasources import LazyAccess

__all__ = ['Shot', 'ShotSeries']


class Shot(collections.abc.MutableMapping):
    '''
    The Shot class representing a single shot or event on the experiment.
    The most convenient way is to think of Shots as python dictionaries. They
    have many key value pairs containing all information that is known about
    this particular Shot/Event.

    Values may be assigend a `LazyAccess` object to retrieve the data from disk
    or network on demand. They are automatically accessed using `Shot.__getitem__`.
    The data is not accessed on pickling.
    Be careful, as `dict(shot)` will access all data! Us `shot._mapping` to directly
    access the data while preventing the LazyAccess to load the full data.

    Copyright:
    Alexander Blinne, 2018
    Stephan Kuschel, 2018
    '''
    diagnostics = dict()
    unknowncontent = [None, '', ' ', 'None', 'unknown', '?', 'NA', np.nan]
    __slots__ = ['_mapping']

    def __new__(cls, *args, **kwargs):
        # ensure: `Shot(shot) is shot`. see also: test_double_init
        if len(args) == 1 and isinstance(args[0], Shot):
            return args[0]  # kwargs are handled in __init__
        else:
            return super(Shot, cls).__new__(cls)

    def __init__(self, *args, **kwargs):
        '''
        Initiate a Shot like a dictionary.

        Up to one dictionary or another Shot can be supplied. Additional
        values can be given as `kwargs`.

        kwargs
        ------
          skipcheck: bool, False: this is a special kwarg, which -- if set to true --
            changes the __init__ process of the Shot.
            If `skipcheck is True`, then the supplied dictionary
            is directly attached to the Shot as its internal
            dictionary, skipping some sanity checks, which `__setitem__` would
            perform.
            In case this Shot is initialized with another Shot, sanity
            checks are always bypassed (`Shot(shot) is Shot` is always `True`).

          all other kwargs are added to the shots dictionary.
        '''
        if len(args) > 1:
            s = 'Shot.__init__ expected at most 1 arguments, got {}'
            raise TypeError(s.format(len(args)))
        if len(args) == 1 and isinstance(args[0], Shot):
            # self._mapping = args[0]._mapping already set in __new__
            # therefore `self.update(*args)` is not neccessary.
            self.update(**kwargs)
            return
        # if here, args[0] must be a dict not a shot
        if kwargs.pop('skipcheck', False):
            # skip the use of __setitem__ for every dict item
            # (40ms vs 3sec for 7500 Shots with 70 items each)
            self._mapping = args[0] if len(args) == 1 else dict()
            self.update(**kwargs)
        else:
            self._mapping = dict()
            # self.update calls __setitem__ internally
            self.update(*args, **kwargs)

    def __getitem__(self, key):
        # print('accessing {}'.format(key))
        ret = self._mapping[key]
        # Handle LazyAccess. Lazy access object only hold references to
        # the data and retrieve them when needed.
        if isinstance(ret, LazyAccess):
            # it depends on the LazyAccess object whether or not,
            # the "key" information is beeing used.
            ret = ret.access(self, key)
        return ret

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
        if not isinstance(val, np.ndarray) and val in self.unknowncontent:
            # `np.array([1,2]) in [1,2]` yields a TypeError
            # ignore request as new info is not actually real info
            return
        if key in self and self._mapping[key] != val:
            s = '''
                Once assigned, shots cannot be changed. If you have
                multiple data sources, their information must match.
                You are attempting to reassign the key "{}"
                from "{}" to "{}"
                on Shot "{}".
                '''
            raise ValueError(s.format(key, self[key], val, repr(self)))
        # assign value if all sanity checks passed
        self._mapping[key] = val

    def update(self, *args, **kwargs):
        if len(args) > 1:
            s = 'update expected at most 1 arguments, got {}'
            raise TypeError(s.format(len(args)))
        elif len(args) == 1:
            arg = args[0]
            updatedict = arg._mapping if isinstance(arg, Shot) else arg
            super().update(updatedict, **kwargs)
        else:
            super().update(**kwargs)

    def __iter__(self):
        # iterating over the keys.
        return iter(self._mapping)

    def __eq__(self, other):
        if isinstance(other, Shot):
            return self._mapping == other._mapping
        else:
            return self._mapping == other

    def __len__(self):
        return len(self._mapping)

    def __contains__(self, key):
        # this is a big timesaver, as the default implementation just tries to
        # access the key, discards the result and returns true on success.
        # Therefore this can trigger a LazyAccess.
        return key in self._mapping

    def __delitem__(self, key):
        raise NotImplemented

    def __str__(self):
        s = '<Shot with {} items>'
        return s.format(len(self))

    def __repr__(self):
        s = '<Shot ({} items): {}>'
        return s.format(len(self), self._mapping)


def make_shotid(*shot_id_fields):
    shot_id_fields = collections.OrderedDict(shot_id_fields)

    PlainShotId = collections.namedtuple('ShotId', shot_id_fields.keys())

    class ShotId(PlainShotId):
        def __new__(cls, shot):
            plain_shot_id = PlainShotId(
                **{k: shot[k] for k in shot.keys() if k in shot_id_fields.keys()})
            vals = [conv(val) for conv, val in zip(
                shot_id_fields.values(), plain_shot_id)]
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
        self._shot_id_fields = shot_id_fields
        self.ShotId = make_shotid(*shot_id_fields)
        self._shots = collections.OrderedDict()
        self.sources = dict()

    def __copy__(self):
        newone = type(self)()
        newone.__dict__.update(self.__dict__)
        newone._shots = copy.copy(self.shots)
        return newone

    # the pickle protocol

    def __getstate__(self):
        import copy
        selfdict = copy.copy(self.__dict__)
        # ShotId cannot be pickled
        del selfdict['ShotId']
        del selfdict['_shots']
        shotlistdata = list(self._shots.values())
        return selfdict, shotlistdata

    def __setstate__(self, state):
        selfdict, shotlistdata = state
        self.__init__(*selfdict['_shot_id_fields'])
        self.__dict__.update(selfdict)
        self.merge(shotlistdata)
        return

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
                # Shot(shot) is shot, see Shot.__new__
                self._shots[shotid] = Shot(datadict)
        return self

    def sorted(self, **kwargs):
        sortedlist = sorted(self, **kwargs)
        return ShotSeries.empty_like(self).merge(sortedlist)

    def __iter__(self):
        return iter(self._shots.values())

    def __reversed__(self):
        return reversed(self._shots.values())

    def __getitem__(self, key):
        if isinstance(key, int):
            if key < 0:
                key = len(self) + key
            return next(itertools.islice(self._shots.values(), key, None))

        elif isinstance(key, slice):
            shots = self._shots.values()

            start, stop, step = key.start, key.stop, key.step

            if start and start < 0:
                start = len(self) + start

            if stop and stop < 0:
                stop = len(self) + stop

            if step and step < 0:
                shots = reversed(shots)
                step = -step
                if start:
                    start = len(self) - 1 - start
                if stop:
                    stop = len(self) - 1 - stop

            return list(itertools.islice(shots, start, stop, step))

        else:
            return self._shots[key]

    def __contains__(self, key):
        return key in self._shots

    def __len__(self):
        return len(self._shots)

    def __str__(self):
        s = '<ShotSeries({}): {} entries>'
        sid = 'ShotId{}'.format(self.ShotId._fields)
        return s.format(sid, len(self))

    __repr__ = __str__

    def groupby(self, *keys):
        def keyfun(shot):
            return tuple(shot[key] for key in keys)
        for k, g in itertools.groupby(sorted(self, key=keyfun), key=keyfun):
            if isinstance(k, tuple) and len(k) == 1:
                k = k[0]
            yield k, ShotSeries.empty_like(self).merge(g)

    def filter(self, fun):
        return ShotSeries.empty_like(self).merge(filter(fun, self))

    def filterby(self, **key_val_dict):
        def fun(shot):
            return all(
                shot[key] == val for key, val in key_val_dict.items())
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
