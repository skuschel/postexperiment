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

import numpy as np
import postpic as pp

from .. import common

__all__ = ['ImageReader', 'RawReader']


def ImageReader(filename):
    return pp.Field.importfrom(filename)


@common.FilterFactory
def RawReader(fname, name, width, height, bands=1, bands_axis=2, dtype=np.uint16, **kwargs):
    d = np.fromfile(fname, dtype=dtype)

    shape = [height, width]
    if bands > 1:
        shape.insert(bands_axis, bands)

    d = d.reshape(shape)

    if bands > 1:
        # switch to pixel-inverleaved mode, default for matplotlib
        d = np.moveaxis(d, bands_index, 2)

    # switch to PostPic.Field compatible axes
    d = np.swapaxes(d, 0, 1)[:, ::-1, ...]

    # convert to float
    d = np.asfarray(d)

    axes = []
    axes.append(pp.Axis(name='x', unit='px',
                        grid=np.linspace(0, width - 1, width)))
    axes.append(pp.Axis(name='y', unit='px',
                        grid=np.linspace(0, height - 1, height)))

    if bands > 1:
        axes.append(pp.Axis(name='band', unit='',
                            grid=np.linspace(0, bands - 1, bands)))

    return pp.Field(d, name=name, unit='counts', axes=axes)
