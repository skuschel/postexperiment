
import itertools

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
