
import functools

import numpy as np

from .common import FilterLRU
from .shot import Shot
from .filterfactories import *
from .filters import *

def SetupFocusDiagnostic(key, img_key=None):
    if not img_key:
        img_key = key + '_image'

    Shot.diagnostics[key] = LoadImage(img_key)
    Shot.diagnostics[key+'_horizontal_gaussian'] = FilterLRU(Chain(LoadImage(img_key),
                                                                 SumAxis(1),
                                                                 GaussianFit1D()),
                                                                 maxsize=1024)

    Shot.diagnostics[key+'_sigma_x'] = Chain(Shot.diagnostics[key+'_horizontal_gaussian'],
                                             GetAttr('sigma'))


    Shot.diagnostics[key+'_gaussian'] = FilterLRU(Chain(LoadImage(img_key),
                                                                 GaussianFit2D()),
                                                                 maxsize=1024)

    Shot.diagnostics[key+'_covmat'] = Chain(Shot.diagnostics[key+'_gaussian'],
                                             GetAttr('covmatrix'))

    Shot.diagnostics[key+'_covar'] = Chain(Shot.diagnostics[key+'_gaussian'],
                                             GetAttr('covar'))

    Shot.diagnostics[key+'_varx'] = Chain(Shot.diagnostics[key+'_gaussian'],
                                             GetAttr('varx'))

    Shot.diagnostics[key+'_vary'] = Chain(Shot.diagnostics[key+'_gaussian'],
                                             GetAttr('vary'))


    Shot.diagnostics[key+'_covmat_ellipse'] = Chain(Shot.diagnostics[key+'_gaussian'],
                                             GetAttr('covmat_ellipse'))
