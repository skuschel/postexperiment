#
# This file is part of postexperiment.
#
# postexperiment is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# postexperiment is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with postexperiment. If not, see <http://www.gnu.org/licenses/>.
'''
Copyright:
Alexander Blinne, 2018
'''


import collections
import functools

import numpy as np


def FilterFactory(f):
    '''
    Sets a variable number of default positional arguments,
    which CAN NOT be overridden on the call.
    And sets default kwargs,
    which CAN be overridden on the call.
    '''
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
