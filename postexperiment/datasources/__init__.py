'''
this submodule contains some example datasouces as well as lazyaccess
implementations. Naturally both are somehow tailored to the underlying
data on disk and should be written with some general aspects in mind.

A datasource is just a callable returning a list of dicts.

The `LazyAccess` interface can be found in lazyaccess.

In case you have very specific datasources for your facility, you
can add another python file here named after that facility to grow
a collection of possible routines.

Stephan Kuschel, 2018
'''


from . import lazyaccess
from . import datasources
from .filereaders import *
from .lazyaccess import *
from .datasources import *


__all__ = []
__all__ += filereaders.__all__
__all__ += lazyaccess.__all__
__all__ += datasources.__all__
