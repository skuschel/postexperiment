
import collections
import itertools
import os
import os.path as osp
import re

import numpy as np

from .common import Context

class Shot(dict):
    diagnostics = dict()

    def __getattr__(self, key):
        def call(*args, context=None, **kwargs):
            if context is None:
                context = Context()
            context['shot'] = self
            return self.diagnostics[key](self, *args, context=context, **kwargs)
        return call

    def __hash__(self):
        return id(self)

class ShotSeries(list):
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

    def groupby(self, key):
        keyfun = lambda shot: shot[key]
        for k, g in itertools.groupby(sorted(self, key=keyfun), key=keyfun):
            yield k, ShotSeries(g)

    def filter(self, fun):
        return ShotSeries(filter(fun, self))

    def mean(self, attr, *args, **kwargs):
        return np.mean([getattr(shot, attr)(*args, **kwargs) for shot in self])

    def grouped_mean(self, key, attr, *args, **kwargs):
        res = []
        for value, shots in self.groupby(key):
            res.append((value, shots.mean(attr, *args, **kwargs)))
        res = np.array(res)
        return res.T
