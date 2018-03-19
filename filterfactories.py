
import collections

import numpy as np

import postpic as pp

from . import common
from . import algorithms


@common.FilterFactory
def Chain(line, *args, context=None, **kwargs):
    """
    Applies a chain of filters
    """
    for f in args:
        line = f(line, context=context)
    return line


GaussianParams1D = collections.namedtuple("GaussianParams1D", "center sigma height background")

@common.FilterFactory
def GaussianInitialGuess1D(line, cut_off=0.15, **kwargs):
    """
    Calculate initial guess for a 1D gaussian fit
    """
    background = np.min(line.matrix)
    line_reduced = line - background
    height = np.max(line_reduced.matrix)
    line_reduced = line.replace_data(np.where(line_reduced < cut_off * height, 0 , line_reduced))
    center = algorithms.momentum(line_reduced, 1)
    var = algorithms.momentum(line_reduced, 2, center=center)
    sigma = np.sqrt(var)

    return GaussianParams1D(center=center, sigma=sigma, background=background, height=height)

@common.FilterFactory
def SumAxis(field, axis, **kwargs):
    """
    Sums a field along one axis
    """
    return field.sum(axis=axis)

@common.FilterFactory
def LoadImage(shot, img_key, **kwargs):
    return pp.Field.importfrom(shot[img_key])

@common.FilterFactory
def GetAttr(obj, attrname, **kwargs):
    return getattr(obj, attrname)
