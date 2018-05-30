'''
Datasources are functions, which -- when called without arguments -- return a list of Shot-objects (a list of dictionaries is also sufficient).
Later a list of shots will be converted to a ShotSeries object and
further lit of shots can be merged into the ShotSeries object, which will then
provide combined information. During this conversion/merge santy check will be run
to ensure that the idendical entries have not been assigend different values from
different sources,....

Stephan Kuschel, 2018
'''

import os
import os.path as osp
import re

from . import common
from . import labbook


class ShotSeriesSource(object):
    pass


class LabBookSource(ShotSeriesSource):
    '''
    Creates a list of `Shot`s from given csv data downloadable from google docs.

    Stephan Kuschel, 2018
    '''
    def __init__(self, link, continued_int_id_field, **kwargs):
        self.link = link
        self.continued_int_id_field = continued_int_id_field
        self.kwargs = kwargs

    def __call__(self):
        full_shotlist = labbook.create_full_shotlist_from_googledocs(self.link,
                                        self.continued_int_id_field, **self.kwargs)
        return full_shotlist


class FileSource(ShotSeriesSource):
    """
    Produces a list of `Shot`s, given a filename `pattern`, a directory `dirname` and a,
    description of `fields` that shall be extracted from the file names.

    The filepath will be stored under the key given by `filekey`. `filekey` can be the,
    literal key or an integer. In the latter case the integer will be used to identify
    a regex group number and the filekey will be taken from the match.

    The `fields` argument should be a mapping from regex group numbers to tuples stating the
    name of the field and an optional transformation function for the field (e. g. `int`).
    If the transformation fails, the matched string is stored untransformed.

    Alexander Blinne, 2018
    """
    def __init__(self, dirname, pattern, filekey, fields, skiptemp=True):
        self.dirname = dirname
        self.pattern = re.compile(pattern)
        self.filekey = filekey
        self.fields = fields
        self.skiptemp = skiptemp

    def __call__(self):
        shots = []

        for root, dirs, files in os.walk(self.dirname):
            for name in files:
                if self.skiptemp and name.endswith('temp'):
                    continue

                path = osp.join(root, name)

                match = self.pattern.match(name)
                if not match:
                    continue

                shot = Shot()
                shots.append(shot)

                if isinstance(self.filekey, int):
                    shot[match.group(self.filekey)] = path
                else:
                    shot[self.filekey] = path

                for i, (n, t) in self.fields.items():
                    try:
                        if t:
                            shot[n] = t(match.group(i))
                        else:
                            shot[n] = match.group(i)
                    except ValueError:
                        shot[n] = match.group(i)

        return shots
