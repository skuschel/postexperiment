
import collections
import functools

def FilterFactory(f):
    def wrapper(*args, **kwargs):
        @functools.wraps(f)
        def call(field, context=None):
            return f(field, *args, context=context, **kwargs)
        return call
    return wrapper


class Context(dict):
    """
    All Contexts are to be considered equal w.r.t. hashing, such that they are
    ignored by LRU caching. Apart from this property, they behave just like regular dicts.
    """
    def __hash__(self):
        return 0


def FilterLRU(fil, maxsize=None):
    if maxsize is None:
        return functools.lru_cache()(fil)
    return functools.lru_cache(maxsize=maxsize)(fil)


GaussianParams1D = collections.namedtuple("GaussianParams1D", "amplitude center sigma const_bg")

class GaussianParams2D(collections.namedtuple("GaussianParams2D", "amplitude center_x center_y varx vary covar const_bg")):
    @property
    def covmatrix(self):
        return np.array([[self.varx, self.covar],[self.covar, self.vary]])
