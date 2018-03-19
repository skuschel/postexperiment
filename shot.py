
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

    def grouped_mean(self, key, attr):
        res = []
        for value, shots in self.groupby(key):
            res.append((value, np.mean([getattr(shot, attr)() for shot in shots])))
        res = np.array(res)
        return res.T
