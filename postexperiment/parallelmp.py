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
This module provides helper function used for parallel execution.
The functions here are quite general and may be useful without postexperiment as well.

All routines here use the multiprocessing package.

Copyright:
Stephan Kuschel, 2019
'''

__all__ = ['limitedbuffer_imap']


def limitedbuffer_imap(func, iterable, pool=None):
    '''
    Same as `pool.imap` with `pool = multiprocessing.Pool()`, but
    only `multiprocessing.cpu_count()` many results are calculated ahead of time.
    `pool.imap` would calculate all, even if no results is yielded, which is
    somehow against the idea of an iterator.

    Stephan Kuschel, 2019
    '''
    if pool is None:
        # serial verison
        for obj in iterable:
            yield func(obj)
        return
    # parallel version
    import multiprocessing as mp
    import queue
    q = queue.Queue()
    iterator = iter(iterable)
    # fill buffer up to the length of `mp.cpu_count()`
    for obj, _ in zip(iterator, range(mp.cpu_count())):
        q.put(pool.apply_async(func, (obj,)))

    while not q.empty():
        # Results from `apply_async` will be yielded.
        # In order to access the result data, a second `.get()` call is required.
        # Exceptions will be reraised by the second `.get()`
        yield q.get()
        try:
            # after yielding, add another to be executed in the background:
            q.put(pool.apply_async(func, (next(iterator),)))
        except(StopIteration):
            pass


def limitedbuffer_imap_reversed(funcs, arg, pool=None):
    '''
    Same as `limitedbuffer_imap`, but takes a list of functions and the argument
    is fixed.

    Stephan Kuschel, 2019
    '''
    if pool is None:
        # serial verison
        for f in funcs:
            yield f(arg)
        return
    # parallel version
    import multiprocessing as mp
    import queue
    q = queue.Queue()
    funciter = iter(funcs)
    # fill buffer up to the length of `mp.cpu_count()`
    for func, _ in zip(funciter, range(mp.cpu_count())):
        q.put(pool.apply_async(func, (arg,)))

    while not q.empty():
        # Results from `apply_async` will be yielded.
        # In order to access the result data, a second `.get()` call is required.
        # Exceptions will be reraised by the second `.get()`
        yield q.get()
        try:
            # after yielding, add another to be executed in the background:
            q.put(pool.apply_async(next(funciter), (arg,)))
        except(StopIteration):
            pass
