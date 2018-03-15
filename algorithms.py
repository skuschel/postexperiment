
import numpy as np
import numpy.linalg as la
import scipy.ndimage
from scipy import optimize

def momentum(field, r, center=0):
    '''
    Calculates the r-th momentum of the 1D distribution assuming that
    center is the central position.
    Copied from auswertungsscripte.git rev 1f85936

    Args:
        field (pp.Field): the 1D probability density

    kwargs:
        center (float): The center postion (default: 0)

    Returns:
        float: the r-th momentum of the data

    Author: Stephan Kuschel, 2016
    '''
    field = field.squeeze()
    if field.dimensions != 1:
        raise ValueError("This function is only for 1D Fields.")

    x = field.grid

    ret = ((x-center)**r * field.matrix).sum()
    norm = (field.matrix).sum()

    return ret/norm

def gaussian_1d(meshgrid, params, **kwargs):
    x, = meshgrid
    return params.background + params.height*np.exp(-(x-params.center)**2/(2*params.sigma**2))

def field_evaluate(field, fun, *args, **kwargs):
    return field.replace_data(fun(field.meshgrid(), *args, **kwargs))
