
import collections
import functools

import numpy as np
import numpy.linalg as nplin

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

    def __eq__(self, other):
        return True


def FilterLRU(fil, maxsize=None):
    if maxsize is None:
        return functools.lru_cache()(fil)
    return functools.lru_cache(maxsize=maxsize)(fil)


GaussianParams1D = collections.namedtuple("GaussianParams1D", "amplitude center sigma const_bg")

class GaussianParams2D(collections.namedtuple("GaussianParams2D", "amplitude center_x center_y varx vary covar const_bg")):
    @property
    def covmatrix(self):
        return np.array([[self.varx, self.covar],[self.covar, self.vary]])

    @property
    def covmat_ellipse(self):
        '''
        converts the covariance matrix to width, height and angle.

        Parameters
        ----------
        covmatrix: 2x2 numpy.ndarray
            the covariance matrix

        Returns
        -------
        width: float
            the width of the divergence ellipse (radius)
        height: float
            the height of the divergence ellipse (radius)
        angle: float
            the angle of the divernce ellipse in rad
        area: float
            the area of the ellipse
            area = np.pi * width * height

        Author: Stephan Kuschel, 2016, Alexander Blinne, 2018
        '''
        (eigval, eigvec) = nplin.eig(self.covmatrix)
        eigval = np.abs(eigval)
        width = np.sqrt(eigval[0])
        height= np.sqrt(eigval[1])
        angle = np.arctan2(eigvec[1,0], eigvec[0,0])
        area = np.pi * width * height
        return (width, height, angle, area)

