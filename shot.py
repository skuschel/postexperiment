
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
    diagnostics = dict()
    unknowncontent = [None, '', ' ', 'None', 'unknown', '?']

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

class ShotSeriesSource(object):
    pass

class LabBookSource(ShotSeriesSource):
    '''
    Creates a list of `Shot`s from given csv data downloadable from google docs.
    '''
    def __init__(self, link, continued_int_id_field, **kwargs):
        self.link = link
        self.continued_int_id_field = continued_int_id_field
        self.kwargs = kwargs

    def __call__(self):
        full_shotlist = labbook.create_full_shotlist_from_googledocs(self.link,
                                        self.continued_int_id_field, **self.kwargs)
        return full_shotlist


class FileSource(ShotSeriesSource):
    """
    Produces a list of `Shot`s, given a filename `pattern`, a directory `dirname` and a,
    description of `fields` that shall be extracted from the file names.

    The filepath will be stored under the key given by `filekey`. `filekey` can be the,
    literal key or an integer. In the latter case the integer will be used to identify
    a regex group number and the filekey will be taken from the match.

    The `fields` argument should be a mapping from regex group numbers to tuples stating the
    name of the field and an optional transformation function for the field (e. g. `int`).
    If the transformation fails, the matched string is stored untransformed.
    """
    def __init__(self, dirname, pattern, filekey, fields, skiptemp=True):
        self.dirname = dirname
        self.pattern = re.compile(pattern)
        self.filekey = filekey
        self.fields = fields
        self.skiptemp = skiptemp

    def __call__(self):
        shots = []

        for root, dirs, files in os.walk(self.dirname):
            for name in files:
                if self.skiptemp and name.endswith('temp'):
                    continue

                path = osp.join(root, name)

                match = self.pattern.match(name)
                if not match:
                    continue

                shot = Shot()
                shots.append(shot)

                shot[match.group(self.filekey)] = path
                for i, (n, t) in self.fields.items():
                    try:
                        if t:
                            shot[n] = t(match.group(i))
                        else:
                            shot[n] = match.group(i)
                    except ValueError:
                        shot[n] = match.group(i)

        return shots


class ShotSeries(object):
    def __init__(self, *shot_id_fields):
        '''
        Data must be a list of dictionaries or None.
        '''
        self._shot_id_fields = shot_id_fields
        self.ShotId = collections.namedtuple('ShotId', self._shot_id_fields)
        self._shots = collections.OrderedDict()
        self.sources = dict()

    def __copy__(self):
        new = ShotSeries(*self._shot_id_fields)
        new.merge(iter(self))
        return new

    @classmethod
    def empty_like(cls, other):
        return cls(*other._shot_id_fields)

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
        for shot in shotlist:
            shotid = self.ShotId(**{k: v for k, v in shot.items() if k in self._shot_id_fields})
            if shotid in self._shots:
                self._shots[shotid].update(shot)
            else:
                # add entirely new the data
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

            if step and step < 0:
                shots = reversed(shots)
                step -= step
                start, stop = stop, start

            if start and start < 0:
                start = len(self)+start

            if stop and stop < 0:
                stop = len(self)+stop

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
