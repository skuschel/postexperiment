'''
A collection of fit models and functions to fit them to the data.


Copyright:
Alexander Blinne, 2018
'''


import collections

import numpy as np
from scipy import optimize
import numpy.linalg as nplin

from . import algorithms
from . import common


class FitModel(object):
    def do_fit(self, line, context=None, **kwargs):
        """
        Calculate a fit
        """
        p0 = self.initial_guess(line, **kwargs)

        mesh = line.meshgrid()

        def errfunc(p):
            p = self.params_array_to_tuple(p)
            model = self(p)
            res = (line.matrix - model(*mesh)).reshape(-1)
            return res[np.isfinite(res)]

        p, pconv = optimize.leastsq(errfunc, self.params_tuple_to_array(p0))

        p = self.params_array_to_tuple(p)

        if context:
            context['Fit_p0'] = p0
            context['Fit_p'] = p
            context['Fit_pconv'] = pconv

        return p

    def params_array_to_tuple(self, params):
        return self.ParamsType(*params)

    def params_tuple_to_array(self, params):
        return np.array(params)


Gaussian1DParams = collections.namedtuple(
    "GaussianParams1D", "amplitude center sigma const_bg")


class Gaussian1D(FitModel):
    ParamsType = Gaussian1DParams

    def initial_guess(self, line, cutoff=0.15, **kwargs):
        """
        Calculate initial guess for a 1D gaussian fit
        """
        const_bg = np.percentile(line, 0.005)
        amplitude = np.percentile(line, 99.995) - const_bg
        line_reduced = line - const_bg

        line_reduced = line.replace_data(
            np.where(line_reduced < amplitude * cutoff, 0, line_reduced))
        center = algorithms.momentum1d(line_reduced, 1)
        var = algorithms.momentum1d(line_reduced, 2, center=center)
        sigma = np.sqrt(var)

        return self.ParamsType(center=center, sigma=sigma, const_bg=const_bg, amplitude=amplitude)

    def __call__(self, params):
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
        return lambda x: params.const_bg + params.amplitude * np.exp(-(x - params.center)**2 / (2 * params.sigma**2))

    def params_array_to_tuple(self, params):
        amplitude, center, sigma, const_bg = params
        return self.ParamsType(amplitude, center, sigma, abs(const_bg))


gaussian_1d = Gaussian1D()


class GaussianParams2D(collections.namedtuple("GaussianParams2D", "amplitude center_x center_y varx vary covar const_bg")):
    @property
    def covmatrix(self):
        return np.array([[self.varx, self.covar], [self.covar, self.vary]])

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
        height = np.sqrt(eigval[1])
        angle = np.arctan2(eigvec[1, 0], eigvec[0, 0])
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

        theta = np.linspace(0, 2 * np.pi, n)

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
        return matplotlib.patches.Ellipse(center, 2 * ell[0], 2 * ell[1], ell[2] / np.pi * 180, **kwargs)


class Gaussian2D(FitModel):
    ParamsType = GaussianParams2D

    def initial_guess(self, field, cutoff=0.15, **kwargs):
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

        field_reduced = field_reduced.replace_data(
            np.where(field_reduced > amplitude * cutoff, field_reduced, 0))

        center_x = algorithms.momentum1d(field_reduced.sum(axis=1), 1)
        center_y = algorithms.momentum1d(field_reduced.sum(axis=0), 1)

        varx = algorithms.momentum1d(
            field_reduced.sum(axis=1), 2, center=center_x)
        vary = algorithms.momentum1d(
            field_reduced.sum(axis=0), 2, center=center_y)
        covar = algorithms.momentum2d(
            field_reduced, 1, center=[center_x, center_y])

        return self.ParamsType(amplitude=amplitude, center_x=center_x, center_y=center_y, varx=varx, vary=vary, covar=covar, const_bg=const_bg)

    def __call__(self, params):
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
        rho = covar / np.sqrt(varx * vary)
        sigmax = np.sqrt(varx)
        sigmay = np.sqrt(vary)
        # http://en.wikipedia.org/wiki/Multivariate_normal_distribution
        return lambda x, y: const_bg + amplitude * np.exp(-(
            1. / (2 * (1. - rho**2)) *
            ((x - center_x)**2 / sigmax**2 + (y - center_y)**2 / sigmay**2 -
             (2. * rho * (x - center_x) * (y - center_y) / (sigmax * sigmay)))
        ))

    def params_array_to_tuple(self, params):
        amplitude, center_x, center_y, varx, vary, covar, const_bg = params
        return self.ParamsType(amplitude, center_x, center_y, varx, vary, covar, abs(const_bg))


gaussian_2d = Gaussian2D()


PolyExponentialParams1D = collections.namedtuple(
    'PolyExponentialParams1D', "a b")


class PolyExponential1D(FitModel):
    ParamsType = PolyExponentialParams1D

    def do_fit(self, line, fit_roi=None, **kwargs):
        if fit_roi is not None:
            line = line[slice(*fit_roi)]

        return super().do_fit(line, **kwargs)

    def initial_guess(self, line, fit_roi=None, **kwargs):
        if fit_roi is not None:
            line = line[slice(*fit_roi)]

        m = np.max(line.matrix)
        i = float(np.argmax(line))
        b0 = 1.5 * i
        a0 = (b0 * 2. / 3. / np.exp(1))**(2. / 3.) * m

        return self.ParamsType(a=a0, b=b0)

    def __call__(self, params):
        return lambda x: params.a * x**(2. / 3.) * np.exp(-x / params.b)


polyexponential_1d = PolyExponential1D()
