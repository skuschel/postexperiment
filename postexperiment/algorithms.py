'''
A loose collection of general purpose free functions.
Please make sure, that every function has:
* A marvelous docsting, ideally containing a 3 line example.
* Your name written in the docsting.
* takes **data** as an input -- not a `Shot`! If it takes
  a `Shot` as the input, it should go into the `diagnostics`
  submodule.

Copyright:
Alexander Blinne, 2018
Stephan Kuschel, 2018
'''


import numpy as np
import numpy.ma as npma
import numpy.linalg as la
import scipy.ndimage
from scipy import optimize


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


def field_evaluate(field, fun, *args, **kwargs):
    return field.replace_data(fun(*field.meshgrid(), *args, **kwargs))


def projective_transform(p, i, j):
    """
    generic 2d projective transformation given by 4 mapped points

    https://math.stackexchange.com/questions/296794/
    finding-the-transform-matrix-from-4-projected-points-with-javascript

    Author: Alexander Blinne, 2018
    """
    M = np.hstack((p, [1]))
    M = M.reshape(3,3)

    x = M[0,0] * i + M[0,1] * j + M[0,2]
    y = M[1,0] * i + M[1,1] * j + M[1,2]
    z = M[2,0] * i + M[2,1] * j + M[2,2]

    return x/z, y/z

def calculate_projective_transform_parameters(points_ij, points_xy):
    def residue(p, points_ij, points_xy):
        x, y = projective_transform(p, points_ij[:,0], points_ij[:,1])
        return np.hstack((points_xy[:,0] - x, points_xy[:,1] - y))

    p, conv = optimize.leastsq(residue, np.array([1., 0., 0., 0., 1., 0., 0., 0.]),
                                     args=(points_ij, points_xy))

    return p

def remove_linear_background_2d(array, mask):
    """
    array: array to remove the background from
    mask: array with dtype=bool that determines which pixels should be considered to contain signal.
    All pixels i,j with mask[i,j] == False will be used to make a linear fit for the background.
    Those should be distributed across the image, otherwise unexpected behaviour may occur.
    """
    i, j = np.indices(array.shape)

    def linear(p, i, j):
        mx, my, b = p
        return mx * i + my * j + b

    def residue(p, i, j, v):
        return linear(p, i, j) - v

    im = np.ma.masked_array(i, mask).compressed()
    jm = np.ma.masked_array(j, mask).compressed()
    vm = np.ma.masked_array(array, mask).compressed()

    p, conv = optimize.leastsq(residue, [0,0,0], args=(im, jm, vm))

    return array - linear(p, i, j)
