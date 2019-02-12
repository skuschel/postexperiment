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
