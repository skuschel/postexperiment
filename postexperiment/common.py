'''
Copyright:
Alexander Blinne, 2018
'''


import collections
import functools

import numpy as np
import numpy.linalg as nplin

def FilterFactory(f):
    def wrapper(*args, **kwargs_default):
        @functools.wraps(f)
        def call(field, **kwargs_call):
            kwargs = dict()
            kwargs.update(kwargs_default)
            kwargs.update(kwargs_call)
            return f(field, *args, **kwargs)
        return call
    return wrapper


class DefaultContext(dict):
    """
    All implicitly created Contexts are to be considered equal w.r.t. hashing, such that they are
    ignored by LRU caching. Apart from this property, they behave just like regular dicts.
    """
    def __hash__(self):
        return 0

    def __eq__(self, other):
        return True


class Context(dict):
    """
    All explicitly created Contexts are to be considered unequal w.r.t. hashing, such that the LRU
    cache is effectively bypassed. Apart from this property, they behave just like regular dicts.
    """
    def __hash__(self):
        return id(self)


def FilterLRU(fil, maxsize=None):
    if maxsize is None:
        return functools.lru_cache()(fil)
    return functools.lru_cache(maxsize=maxsize)(fil)