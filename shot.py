
import itertools
import os
import os.path as osp
import re

import numpy as np

class Shot(dict):
    diagnostics = dict()

    def __getattr__(self, key):
        def call(*args, context=None, **kwargs):
            if context is None:
                context = dict()
            context['shot'] = self
            return self.diagnostics[key](self, *args, context=context, **kwargs)
        return call

class ShotSeries(list):
    @classmethod
    def from_files(cls, dirname, pattern, filekey, fields):
        """
        Produces a list of `Shot`s, given a filename `pattern`, a directory `dirname` and a description of `fields`
        that shall be extracted from the file names. The filepath will be stored under the key given by `filekey`.
        """
        pattern = re.compile(pattern)
        names = os.listdir(dirname)
        shots = cls()
        for name in names:
            path = osp.join(dirname, name)
            if not osp.isfile(path):
                continue
            match = pattern.match(name)
            if not match:
                continue

            shotinfo = dict()
            shotinfo['name'] = osp.splitext(name)[0]
            shotinfo[filekey] = path
            for i, (n, t) in fields.items():
                try:
                    shotinfo[n] = t(match.group(i))
                except ValueError:
                    shotinfo[n] = match.group(i)

            shots.append(Shot(**shotinfo))

        return shots

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
