
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

PolyExponentialParams1D = collections.namedtuple('PolyExponentialParams1D', "a b")

GaussianParams1D = collections.namedtuple("GaussianParams1D", "amplitude center sigma const_bg")

class GaussianParams2D(collections.namedtuple("GaussianParams2D", "amplitude center_x center_y varx vary covar const_bg")):
    @property
    def covmatrix(self):
        return np.array([[self.varx, self.covar],[self.covar, self.vary]])

    @property
    def covmat_ellipse(self):
        '''
        converts the covariance matrix to width, height and angle.

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

    def covmat_ellipse_line(self, n=100, s=1.0):
        '''
        returns the covariance ellipse as a sequence of points

        Parameter
        ---------
        n: int
            number of points
        s: float
            scale


        Returns
        -------
        x: ndarray
        y: ndarray
            the sequence of points used to plot the ellipse

        Author: Alexander Blinne, 2018
        '''
        width, height, angle, area = self.covmat_ellipse

        theta = np.linspace(0, 2*np.pi, n)

        ex = s * width * np.cos(theta)
        ey = s * height * np.sin(theta)

        x = ex * np.cos(-angle) + ey * np.sin(-angle)
        y = -ex * np.sin(-angle) + ey * np.cos(-angle)

        return self.center_x + x, self.center_y + y

    def covmat_ellipse_artist(self, **kwargs):
        import matplotlib.patches
        ell = self.covmat_ellipse
        center = self.center_x, self.center_y
        if 'fill' not in kwargs:
            kwargs['fill'] = None
        return matplotlib.patches.Ellipse(center, 2*ell[0], 2*ell[1], ell[2]/np.pi*180, **kwargs)

