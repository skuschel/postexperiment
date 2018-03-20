
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

@common.FilterFactory
def GaussianInitialGuess1D(line, cutoff=0.15, **kwargs):
    """
    Calculate initial guess for a 1D gaussian fit
    """
    const_bg = np.min(line.matrix)
    line_reduced = line - const_bg
    amplitude = np.percentile(line_reduced,99.995)

    line_reduced = line.replace_data(np.where(line_reduced < amplitude * cutoff, 0 , line_reduced))
    center = algorithms.momentum1d(line_reduced, 1)
    var = algorithms.momentum1d(line_reduced, 2, center=center)
    sigma = np.sqrt(var)

    return common.GaussianParams1D(center=center, sigma=sigma, const_bg=const_bg, amplitude=amplitude)


@common.FilterFactory
def GaussianInitialGuess2D(field, cutoff=None, **kwargs):
    '''
    Calculates the covariance matrix from a given 2d histogram.
    This function produces bullshit because its way too sensitive
    to noise.
    Copied from auswertungsscripte.git rev 1f85936

    Args:
        data (np.array): the 2D probability density

    kwargs:
        center ((float, float)): The center postion (default: (0,0))

    Returns:
        numpy.array: the covmatrix

    Author: Stephan Kuschel, 2016
    '''
    const_bg = np.min(field.matrix)
    field_reduced = field - const_bg
    amplitude = np.max(field_reduced)

    if cutoff is None:
        cutoff = np.percentile(field_reduced,99.995) * 1/np.sqrt(np.e)

    field_reduced = field_reduced.replace_data(np.where(field_reduced > cutoff, field_reduced, 0))

    center_x = algorithms.momentum1d(field_reduced.sum(axis=0), 1)
    center_y = algorithms.momentum1d(field_reduced.sum(axis=1), 1)

    varx = algorithms.momentum1d(field_reduced.sum(axis=0), 2, center=center_x)
    vary = algorithms.momentum1d(field_reduced.sum(axis=1), 2, center=center_y)
    covar = algorithms.momentum2d(field_reduced, 1, center=[center_x, center_y])

    return common.GaussianParams2D(amplitude=amplitude, center_x=center_x, center_y=center_y, varx=varx, vary=vary, covar=covar, const_bg=const_bg)


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
