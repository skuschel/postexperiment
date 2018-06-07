'''
Datasources are functions, which -- when called without arguments -- return a list of
Shot-objects (a list of dictionaries is also sufficient).
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

from .labbook import LabBookSource
from .lazyaccess import LazyAccessH5


__all__ = ['LabBookSource', 'FileSource', 'H5ArraySource']


class FileSource():
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

                shot = dict()
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


class H5ArraySource():
    '''
    This source describes a hdf5 data source, in which various keys contain
    an array fo values, one for each Shot.

    Stephan Kuschel, 2018
    '''

    def __init__(self, filename, validkey):
        '''
        args
        ----
        filename: str
          the filename

        validkey: str
          key to one valid dataset
        '''
        self._filename = filename
        self._validkey = validkey
        self._validkeys = None

    @property
    def filename(self):
        return self._filename

    @property
    def validkey(self):
        return self._validkey

    @property
    def validkeys(self):
        if self._validkeys is None:
            self._validkeys = self._genkeylist(self.validkey)
        return self._validkeys

    def __len__(self):
        import h5py
        h5 = h5py.File(self.filename, 'r')
        return h5[self.validkey].shape[0]

    def _genkeylist(self, validkey):
        '''
        generate the keylist.

        validkey points to an exisiting valid dataset.
        This function will return the keys to all datasets with equal length.
        They will be didvided into two groups: small and large.
        "small" data contains only single values
        "large" data contains (large) arrays.
        For large data only a reference is saved in the dictionary.
        '''
        import h5py
        length = len(self)

        def isvaliddata(item):
            if not isinstance(item, h5py.Dataset):
                return False
            try:
                if item.shape[0] == length:
                    return True
            except(TypeError, IndexError):
                return False

        retsmall = []
        retlarge = []

        def visitf(key, item):
            if not isvaliddata(item):
                return
            if item.shape[1:] is () or item.shape[1:] == (1,):
                retsmall.append(key)
            else:
                retlarge.append(key)
        h5 = h5py.File(self.filename, 'r')
        h5.visititems(visitf)
        return retsmall, retlarge

    def __call__(self):
        '''
        returns a list of dictionaries
        '''
        return self.gendatadict()

    def gendatadict(self, n=None):
        '''
        creates the datadict for the nth event.
        Creates a list of events if n is not given
        '''
        import h5py
        smallkeys, largekeys = self.validkeys
        # this is compuationally surprisingly cheap
        h5 = h5py.File(self.filename, 'r')
        # creating the h5datasets only once cuts execution time in half.
        dsets = {key: h5[key] for key in smallkeys}

        def gendict(i):
            d = {key: dsets[key][i] for key in smallkeys}
            # key not given, index is fixed
            la = LazyAccessH5(self.filename, index=i)
            d.update({key: la for key in largekeys})
            return d
        if n is None:
            return [gendict(i) for i in range(len(self))]
        else:
            return gendict(n)
