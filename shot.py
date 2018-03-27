
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
    def __getattr__(self, key):
        if key.startswith('_'):
            raise AttributeError
        def call(*args, context=None, **kwargs):
            if context is None:
                context = common.DefaultContext()
            context['shot'] = self
            return self.diagnostics[key](self, *args, context=context, **kwargs)
        return call

    def __hash__(self):
        return id(self)

class ShotSeries(list):

    @classmethod
    def from_googledocs(cls, link, continued_int_id_field, **kwargs):
        '''
        Creates a list of `Shot`s from given csv data downloadable from google docs.
        '''
        full_shotlist = labbook.create_full_shotlist_from_googledocs(link,
                                                            continued_int_id_field, **kwargs)
        return ShotSeries(full_shotlist)

    @classmethod
    def from_files(cls, dirname, pattern, filekey, fields, shot_id_fields=None, skiptemp=True):
        """
        Produces a list of `Shot`s, given a filename `pattern`, a directory `dirname` and a,
        description of `fields` that shall be extracted from the file names.

        The filepath will be stored under the key given by `filekey`. `filekey` can be the,
        literal key or an integer. In the latter case the integer will be used to identify
        a regex group number and the filekey will be taken from the match.

        The `fields` argument should be a mapping from regex group numbers to tuples stating the
        name of the field and an optional transformation function for the field (e. g. `int`).
        If the transformation fails, the matched string is stored untransformed.

        If `shot_id_fields` is given, then all files which are identical in all the fields
        mentioned in `shot_id_fields` will be merged into a single shot. In this case using an
        integer `filekey` argument is most useful to identify the different files belonging to
        a shot.
        """
        pattern = re.compile(pattern)
        if shot_id_fields is not None:
            ShotId = collections.namedtuple('ShotId', shot_id_fields)
        shots = dict()

        for root, dirs, files in os.walk(dirname):
            for name in files:
                if skiptemp and name.endswith('temp'):
                    continue

                path = osp.join(root, name)

                match = pattern.match(name)
                if not match:
                    continue

                shotinfo = dict()
                for i, (n, t) in fields.items():
                    try:
                        if t:
                            shotinfo[n] = t(match.group(i))
                        else:
                            shotinfo[n] = match.group(i)
                    except ValueError:
                        shotinfo[n] = match.group(i)

                if shot_id_fields is None:
                    shot = shots[path] = Shot(**shotinfo)
                    shot[filekey] = path
                else:
                    shot_id = ShotId(**{k: v for k, v in shotinfo.items() if k in shot_id_fields})
                    shot = shots.setdefault(shot_id, Shot(**shotinfo))
                    shot[match.group(filekey)] = path

        return ShotSeries(shots.values())

    def sortby(self, *keys):
        keyfun = lambda shot: tuple(shot[key] for key in keys)
        return ShotSeries(sorted(self, key=keyfun))

    def groupby(self, *keys):
        keyfun = lambda shot: tuple(shot[key] for key in keys)
        for k, g in itertools.groupby(sorted(self, key=keyfun), key=keyfun):
            if isinstance(k, tuple) and len(k)==1:
                k = k[0]
            yield k, ShotSeries(g)

    def filter(self, fun):
        return ShotSeries(filter(fun, self))

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
