
import numpy as np
import numpy.linalg as la
import scipy.ndimage
from scipy import optimize

from . import common

def momentum1d(field, r, center=0):
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

    Author: Stephan Kuschel, 2016, Alexander Blinne, 2018
    '''
    field = field.squeeze()
    if field.dimensions != 1:
        raise ValueError("This function is only for 1D Fields.")

    x = field.grid

    ret = ((x-center)**r * field.matrix).sum()
    norm = (field.matrix).sum()

    return ret/norm


def momentum2d(field, r, center=[0,0]):
    '''
    Calculates the r-th momentum of the 2D distribution assuming that
    center is the central position.
    Copied from auswertungsscripte.git rev 1f85936

    Args:
        data (np.array): the 2D probability density

    kwargs:
        center ((float, float)): The center postion (default: (0,0))

    Returns:
        float: the r-th momentum of the data

    Author: Stephan Kuschel, 2016, Alexander Blinne, 2018
    '''
    norm = (field.matrix).sum()
    #ret = 0
    #for x in xrange(0, len(data)):
    #    for y in xrange(0,len(data[0])):
    #        ret += ((x-center[0])*(y-center[1]))**r * data[x,y]
    # this is the same, but factor 100 faster:
    X, Y = field.meshgrid()
    ret = (X-center[0])*(Y-center[1])**r * field.matrix
    return ret.sum()/norm


def gaussian_1d(params):
    '''
    A Gaussion 1D distribution with the properties provided.
    Can be used for fitting.

    Args:
        params: GaussianParams1D() namedtuple

    Returns:
        function: A function, that is dependend on x
            that can be used to sample the distribution.

    Author: Alexander Blinne, 2018
    '''
    return lambda x: params.const_bg + params.amplitude*np.exp(-(x-params.center)**2/(2*params.sigma**2))


def gaussian_2d(params):
    '''
    A Gaussion 2D distribution with the properties provided.
    Can be used for fitting.
    Copied from auswertungsscripte.git rev 1f85936
    See: http://en.wikipedia.org/wiki/Multivariate_normal_distribution

    Args:
        params: GaussianParams2D() namedtuple

    Returns:
        function: A function, that is dependend on x and y
            that can be used to sample the distribution.

    Author: Stephan Kuschel, 2016, Alexander Blinne, 2018
    '''
    amplitude, center_x, center_y, varx, vary, covar, const_bg = params
    center_x = float(center_x)
    center_y = float(center_y)
    varx = float(varx)
    vary = float(vary)
    covar = float(covar)
    rho = covar/np.sqrt(varx*vary)
    sigmax = np.sqrt(varx)
    sigmay = np.sqrt(vary)
    # http://en.wikipedia.org/wiki/Multivariate_normal_distribution
    return lambda x,y: const_bg + amplitude * np.exp(-(
        1./(2*(1.-rho**2)) *
        ((x-center_x)**2/sigmax**2 + (y-center_y)**2/sigmay**2 -
         (2.*rho*(x-center_x)*(y-center_y)/(sigmax*sigmay)))
        ))


def fit_gaussian_1d(line, params):
    x = line.grid

    p0 = np.array(params)

    def errfunc(p):
        p = common.GaussianParams1D(*p)
        model = gaussian_1d(p)
        return line.matrix - model(x)

    return optimize.leastsq(errfunc, p0)


def fit_gaussian_2d(field, params):
    x, y = field.meshgrid()

    p0 = np.array(params)

    def errfunc(p):
        p = common.GaussianParams2D(*p)
        model = gaussian_2d(p)
        return (field.matrix - model(x, y)).reshape(-1)

    return optimize.leastsq(errfunc, p0)


def field_evaluate(field, fun, *args, **kwargs):
    return field.replace_data(fun(field.meshgrid(), *args, **kwargs))


def projective_transform(p, i, j):
    """
    generic 2d transformation given by 4 mapped points
    """
    M = p[:4].reshape(2,2)
    r0 = p[4:6]
    nl = p[6:]

    x = r0[0] + M[0,0] * i + M[0,1] * j + nl[0] * i * j
    y = r0[1] + M[1,0] * i + M[1,1] * j + nl[1] * i * j

    return x, y

def calculate_projective_transform_parameters(points_ij, points_xy):
    def residue(p, points_ij, points_xy):
        x, y = projective_transform(p, points_ij[:,0], points_ij[:,1])
        return np.hstack((points_xy[:,0] - x, points_xy[:,1] - y))

    p, conv = optimize.leastsq(residue, np.array([-1., 0., 0., 1., 0., 0., 0., 0.]),
                                     args=(points_ij, points_xy))

    return p
