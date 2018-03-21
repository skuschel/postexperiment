
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
def GaussianInitialGuess1D(line, cutoff=0.15,  **kwargs):
    """
    Calculate initial guess for a 1D gaussian fit
    """
    const_bg = np.percentile(line, 0.005)
    amplitude = np.percentile(line, 99.995) - const_bg
    line_reduced = line - const_bg

    line_reduced = line.replace_data(np.where(line_reduced < amplitude * cutoff, 0 , line_reduced))
    center = algorithms.momentum1d(line_reduced, 1)
    var = algorithms.momentum1d(line_reduced, 2, center=center)
    sigma = np.sqrt(var)

    return common.GaussianParams1D(center=center, sigma=sigma, const_bg=const_bg, amplitude=amplitude)


@common.FilterFactory
def GaussianFit1D(line, cutoff=0.15, context=None, **kwargs):
    """
    Calculate a 1D gaussian fit
    """
    p0 = GaussianInitialGuess1D(cutoff=cutoff, **kwargs)(line)
    p, pcov = algorithms.fit_gaussian_1d(line, p0)
    p = common.GaussianParams1D(*p)

    if context:
        context['GaussianFit1D_p0'] = p0
        context['GaussianFit1D_p'] = p
        context['GaussianFit1D_pcov'] = pcov

    return p


@common.FilterFactory
def GaussianFit2D(field, cutoff=0.15, context=None, **kwargs):
    """
    Calculate a 2D gaussian fit
    """
    p0 = GaussianInitialGuess2D(cutoff=cutoff, **kwargs)(field)
    p, pcov = algorithms.fit_gaussian_2d(field, p0)
    p = common.GaussianParams2D(*p)

    if context:
        context['GaussianFit2D_p0'] = p0
        context['GaussianFit2D_p'] = p
        context['GaussianFit2D_pcov'] = pcov

    return p


@common.FilterFactory
def GaussianInitialGuess2D(field, cutoff=0.15, **kwargs):
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
    const_bg = np.percentile(field, 0.005)
    amplitude = np.percentile(field, 99.995) - const_bg
    field_reduced = field - const_bg

    field_reduced = field_reduced.replace_data(np.where(field_reduced > amplitude * cutoff, field_reduced, 0))

    center_x = algorithms.momentum1d(field_reduced.sum(axis=1), 1)
    center_y = algorithms.momentum1d(field_reduced.sum(axis=0), 1)

    varx = algorithms.momentum1d(field_reduced.sum(axis=1), 2, center=center_x)
    vary = algorithms.momentum1d(field_reduced.sum(axis=0), 2, center=center_y)
    covar = algorithms.momentum2d(field_reduced, 1, center=[center_x, center_y])

    return common.GaussianParams2D(amplitude=amplitude, center_x=center_x, center_y=center_y, varx=varx, vary=vary, covar=covar, const_bg=const_bg)

@common.FilterFactory
def SubtractOffset(field, offset, **kwargs):
    return field - offset


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
def LoadGasPressure(shot, filekey, **kwargs):
    data = np.fromfile(shot[filekey])
    tAx = pp.Axis(name='t', unit='s', grid=data[:10000])
    fields = [pp.Field(data[(i+1)*10000:(i+2)*10000],
                       name='Channel {}'.format(i+1),
                       unit='V',
                       axes=[tAx])
              for i in range(4)
             ]
    return fields


@common.FilterFactory
def GetAttr(obj, attrname, **kwargs):
    return getattr(obj, attrname)

@common.FilterFactory
def GetItem(obj, item, **kwargs):
    return obj[item]
