'''
The core classes for data evaluation:
`Shot`
  is representing a single event (Laser shot, accelerator shot,..)

`ShotSeries`
  represents a collections of shots.

Copyright:
Alexander Blinne, 2018
Stephan Kuschel, 2018
'''


import copy
import collections
import collections.abc
import itertools
import os
import sys
import os.path as osp
import re
import abc
import functools
from future.utils import with_metaclass
import concurrent.futures as cf

import numpy as np

from . import common
from .datasources import LazyAccess

__all__ = ['Diagnostics', 'Shot', 'ShotSeries']


class Diagnostics():
    '''
    represents a diagnostics.
    This class wraps the callable.
    '''
    def __new__(cls, func, **kwargs):
        # ensure: `Diagnostics(diagnostics) is diagnostics`. see also: test_double_init
        if isinstance(func, Diagnostics):
            return func  # kwargs are handled in __init__
        else:
            return super(Diagnostics, cls).__new__(cls)

    def __init__(self, func):
        if self is func:
            # `Diagnostics(diagnostics)`, do not wrap twice.
            return
        if not callable(func):
            s = '{} must be a callable'.format(func)
            raise TypeError(s)
        self.function = func

    @property
    def __name__(self):
        return self.function.__name__

    def __call__(self, shot, **kwargs):
        return self._execute(shot, **kwargs)

    def _execute(self, shot, **kwargs):
        try:
            ret = self.function(shot, **kwargs)
        except(TypeError):
            kwargs.pop('context')
            ret = self.function(shot, **kwargs)
        return ret

    def __repr__(self):
        return '<Diagnostics({})>'.format(self.function)

    __str__ = __repr__


class Shot(collections.abc.MutableMapping):
    '''
    The Shot class representing a single shot or event on the experiment.
    The most convenient way is to think of Shots as python dictionaries. They
    have many key value pairs containing all information that is known about
    this particular Shot/Event.

    Values may be assigend a `LazyAccess` object to retrieve the data from disk
    or network on demand. They are automatically accessed using `Shot.__getitem__`.
    The data is not accessed on pickling.
    Be careful, as `dict(shot)` will access all data! Us `shot._mapping` to directly
    access the data while preventing the LazyAccess to load the full data.

    Copyright:
    Alexander Blinne, 2018
    Stephan Kuschel, 2018
    '''
    diagnostics = dict()
    alias = dict()
    unknowncontent = [None, '', ' ', 'None', 'none', 'unknown',
                      '?', 'NA', 'N/A', 'n/a', [], ()]
    __slots__ = ['_mapping']
    import numpy as np  # to be used within `__call__`. See also: `__getitem__`.

    @classmethod
    def _register_diagnostic_fromdict(cls, diags):
        '''
        register diagnostics from the provided dictionary.
        The key is used as the functions name. The contents of the
        dictionary should be:

        {'diganostics_name': callable}
        '''
        # make sure to convert all callables to
        # diagnostics
        d = {key: Diagnostics(val) for key, val in diags.items()}
        cls.diagnostics.update(d)

    @classmethod
    def register_diagnostic(cls, arg):
        '''
        This function should be used to register multiple diagnostics.

        A diagnostic is a function, which takes a single `Shot` object and returns
        a result of any kind. Examples of such functions can be found in the
        `postexperiment.diagnostics` submodule.

        arg
        ---
          * Either a function: and the name will be taken
            from the `args.__name__` attribute.
          * Or a dictionary mapping from names to the actual function.
        '''
        if isinstance(arg, collections.Mapping):
            diags = arg
        elif callable(arg):
            diags = {arg.__name__: arg}
        else:
            diags = {f.__name__: f for f in arg}
        cls._register_diagnostic_fromdict(diags)

    def __new__(cls, *args, **kwargs):
        # ensure: `Shot(shot) is shot`. see also: test_double_init
        if len(args) == 1 and isinstance(args[0], Shot):
            return args[0]  # kwargs are handled in __init__
        else:
            return super(Shot, cls).__new__(cls)

    def __init__(self, *args, **kwargs):
        '''
        Initiate a Shot like a dictionary.

        Up to one dictionary or another Shot can be supplied. Additional
        values can be given as `kwargs`.

        kwargs
        ------
          skipcheck: bool, False: this is a special kwarg, which -- if set to true --
            changes the __init__ process of the Shot.
            If `skipcheck is True`, then the supplied dictionary
            is directly attached to the Shot as its internal
            dictionary, skipping some sanity checks, which `__setitem__` would
            perform.
            In case this Shot is initialized with another Shot, sanity
            checks are always bypassed (`Shot(shot) is Shot` is always `True`).

          all other kwargs are added to the shots dictionary.
        '''
        if len(args) > 1:
            s = 'Shot.__init__ expected at most 1 arguments, got {}'
            raise TypeError(s.format(len(args)))
        if len(args) == 1 and isinstance(args[0], Shot):
            # self._mapping = args[0]._mapping already set in __new__
            # therefore `self.update(*args)` is not neccessary.
            self.update(**kwargs)
            return
        # if here, args[0] must be a dict not a shot
        if kwargs.pop('skipcheck', False):
            # skip the use of __setitem__ for every dict item
            # (40ms vs 3sec for 7500 Shots with 70 items each)
            self._mapping = args[0] if len(args) == 1 else dict()
            self.update(**kwargs)
        else:
            self._mapping = dict()
            # self.update calls __setitem__ internally
            self.update(*args, **kwargs)

    def __getitem__(self, key):
        # print('accessing {}'.format(key))
        if key in self.alias:
            return self[self.alias[key]]
        if key in self:
            ret = self._mapping[key]
        else:
            # with this the call interface can use self as the local mapping
            # to gain access to attached diagnostics.
            # this line also gives access to the numpy import on class level.
            ret = getattr(self, key)
        # Handle LazyAccess. Lazy access object only hold references to
        # the data and retrieve them when needed.
        if isinstance(ret, LazyAccess):
            # it depends on the LazyAccess object whether or not,
            # the "key" information is beeing used.
            ret = ret.access(self, key)
        return ret

    def updatealias(self, *args, **kwargs):
        '''
        adds an alias to the mapping of aliases `self.alias`
        same as `self.alias.update(*args, **kwargs)`.
        '''
        self.alias.update(*args, **kwargs)

    def __getattr__(self, key):
        '''
        this function implements the access to all diagnostics.
        '''
        if key.startswith('_'):
            raise AttributeError

        # this raises a NameError if key cannot be found.
        diagnostics = self.diagnostics[key]

        def call(*args, context=None, **kwargs):
            if context is None:
                context = common.DefaultContext()
            context['shot'] = self
            ret = diagnostics(self, *args, context=context, **kwargs)
            return ret
        return call

    def __call__(self, expr):
        '''
        a unified interface to access shot data. Just use dictionary or
        diagnostics names.
        numpy is also available in the namespace as `np`.

        Example:
          * `shot('x + y + examplediagnostics()')`
          * `shot('np.sum(image)')`
        '''
        # globals must be a real dict.
        # locals can be any mapping, therefore just use `self`.
        return eval(expr, {}, self)

    @staticmethod
    def _isvaliddata(val):
        if isinstance(val, np.ndarray):
            s = val.size
            if s == 0:
                # empty array: np.array([])
                # nested empty: np.array([[]])
                return False
            elif s > 1:
                # Arrays with more than one element are always considered data.
                # Todo: `np.array([np.nan, np.nan, None])` is still
                # considered data, but should not.
                return True
            else:
                # either `np.array(['data']`) (`shape=(1,)`)
                # or `np.array('data')` (`shape=()`)
                pass
        if val in Shot.unknowncontent:
            return False
        try:
            if np.isnan(val):
                return False
        except(TypeError):
            pass
        return True

    def __setitem__(self, key, val):
        if not self._isvaliddata(val):
            # ignore invalid data
            # print('ignored: {}'.format(val))
            return
        if key in self and self._mapping[key] != val:
            s = '''
                Once assigned, shots cannot be changed. If you have
                multiple data sources, their information must match.
                You are attempting to reassign the key "{}"
                from "{}" to "{}"
                on Shot "{}".
                '''
            raise ValueError(s.format(key, repr(self[key]), repr(val), repr(self)))
        # assign value if all sanity checks passed
        self._mapping[key] = val

    def update(self, *args, **kwargs):
        if len(args) > 1:
            s = 'update expected at most 1 arguments, got {}'
            raise TypeError(s.format(len(args)))
        elif len(args) == 1:
            arg = args[0]
            updatedict = arg._mapping if isinstance(arg, Shot) else arg
            super().update(updatedict, **kwargs)
        else:
            super().update(**kwargs)

    def __iter__(self):
        # iterating over the keys.
        return iter(self._mapping)

    def __eq__(self, other):
        if isinstance(other, Shot):
            return self._mapping == other._mapping
        else:
            return self._mapping == other

    def __len__(self):
        return len(self._mapping)

    def __contains__(self, key):
        # this is a big timesaver, as the default implementation just tries to
        # access the key, discards the result and returns true on success.
        # Therefore this can trigger a LazyAccess.
        return key in self._mapping

    def __delitem__(self, key):
        raise NotImplemented

    def __str__(self):
        s = '<Shot with {} items>'
        return s.format(len(self))

    def __repr__(self):
        s = '<Shot ({} items): {}>'
        return s.format(len(self), self._mapping)

    def __hash__(self):
        return id(self)


class make_shotid():

    def __init__(self, *shot_id_fields):
        '''
        *shot_id_fields must be tuples containg
          the key name and its type.
          example: `make_shotid(('time', int), ('accurate_time', int))`
        '''
        self._shot_id_fields = sorted(shot_id_fields)

    def __call__(self, shot):
        '''
        returns a hashable tuple which can be used for indexing the shot
        '''
        idx = tuple((k, f(shot[k])) for k, f in self._shot_id_fields)
        return idx

    def __str__(self):
        s = 'ShotId("{}")'
        return s.format(self._shot_id_fields)

    __repr__ = __str__


class ShotSeries(object):

    def __init__(self, *shot_id_fields):
        '''
        Data must be a list of dictionaries or None.
        '''
        self._shot_id_fields = shot_id_fields
        self.ShotId = make_shotid(*shot_id_fields)
        self._shots = collections.OrderedDict()
        self.sources = dict()

    def __copy__(self):
        newone = type(self)()
        newone.__dict__.update(self.__dict__)
        newone._shots = copy.copy(self.shots)
        return newone

    @classmethod
    def empty_like(cls, other):
        newone = type(other)()
        newone.__dict__.update(other.__dict__)
        newone._shots = collections.OrderedDict()
        return newone

    def load(self, nmax=None):
        """
        Loads shots from all attached sources.

        kwargs
        ------
          nmax=None:
            if an int is given only this many shots will be loaded from each source.
        """
        for source in self.sources.values():
            self.merge(source(), nmax=nmax)

        return self

    def merge(self, shotlist, nmax=None):
        '''
        merges a shotlist into the current ShotSeries `self` and
        and combines the provided information. Shots are considered identical if
        ALL shot_id_fields given by `shot_id_fields` are equal. Both ShotSeries
        MUST have all `shot_id_fields` present.

        kwargs
        ------
          nmax=None:
            if an int is given only this many shots merged.
        '''
        nmax = sys.maxsize if nmax is None else nmax
        for datadict, _ in zip(shotlist, range(nmax)):
            shotid = self.ShotId(datadict)
            if shotid in self._shots:
                self._shots[shotid].update(datadict)
            else:
                # add entirely new the data and enusure data is a Shot object
                # Shot(shot) is shot, see Shot.__new__
                self._shots[shotid] = Shot(datadict)
        return self

    def sorted(self, **kwargs):
        sortedlist = sorted(self, **kwargs)
        return ShotSeries.empty_like(self).merge(sortedlist)

    def __call__(self, expr):
        '''
        access shot data via the call interface. Calls will be forwarted
        to all shots contained in this shot series and the results will be yielded.

        Data is only returned for shots containting all required information. All other shots
        are simply left out.
        '''
        # compile the expr once
        # Example: 'a+b+x(2)'
        # compile time: 7.8 us
        # eval time of compiled expr: < 500 ns
        exprc = compile(expr, '<string>', 'eval')
        for shot in self:
            try:
                # yield the result. It may be a single int or a huge image.
                yield shot(exprc)
            except(KeyError, NameError):
                pass

    def __iter__(self):
        return iter(self._shots.values())

    def __reversed__(self):
        return reversed(self._shots.values())

    def __getitem__(self, key):
        if isinstance(key, int):
            if key < 0:
                key = len(self) + key
            return next(itertools.islice(self._shots.values(), key, None))

        elif isinstance(key, slice):
            shots = self._shots.values()

            start, stop, step = key.start, key.stop, key.step

            if start and start < 0:
                start = len(self) + start

            if stop and stop < 0:
                stop = len(self) + stop

            if step and step < 0:
                shots = reversed(shots)
                step = -step
                if start:
                    start = len(self) - 1 - start
                if stop:
                    stop = len(self) - 1 - stop

            return list(itertools.islice(shots, start, stop, step))

        else:
            return self._shots[key]

    def __contains__(self, key):
        return key in self._shots

    def __len__(self):
        return len(self._shots)

    def __str__(self):
        s = '<ShotSeries({}): {} entries>'
        sid = '{}'.format(self.ShotId)
        return s.format(sid, len(self))

    __repr__ = __str__

    def groupby(self, *keys):
        def keyfun(shot):
            return tuple(shot[key] for key in keys)
        for k, g in itertools.groupby(sorted(self, key=keyfun), key=keyfun):
            if isinstance(k, tuple) and len(k) == 1:
                k = k[0]
            yield k, ShotSeries.empty_like(self).merge(g)

    def _filter_fun(self, fun):
        return ShotSeries.empty_like(self).merge(filter(fun, self))

    def _filter_string(self, expr):
        exprc = compile(expr, '<string>', 'eval')
        shotlist = []
        for shot in self:
            try:
                if shot(exprc):
                    shotlist.append(shot)
            except(KeyError, NameError):
                pass
        return ShotSeries.empty_like(self).merge(shotlist)

    def filter(self, f):
        '''
        returns a new ShotSeries, filtered by f.
        f can be:
          * A function where `f(shot)` evaluates to True or False
          * A string such that `shot(f)` evaluates to True or False
        '''
        if callable(f):
            return self._filter_fun(f)
        else:
            return self._filter_string(f)

    def filterby(self, **key_val_dict):
        def fun(shot):
            return all(
                shot[key] == val for key, val in key_val_dict.items())
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
