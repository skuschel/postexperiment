
import functools

from .common import FilterLRU
from .shot import Shot
from .filterfactories import *
from .filters import *

def SetupFocusDiagnostic(key, img_key=None):
    if not img_key:
        img_key = key + '_image'

    Shot.diagnostics[key] = LoadImage(img_key)
    Shot.diagnostics[key+'_horizontal_direct_stats'] = FilterLRU(Chain(LoadImage(img_key),
                                                                 SumAxis(1),
                                                                 GaussianInitialGuess1D()),
                                                                 maxsize=1024)
    Shot.diagnostics[key+'_sigma_x'] = Chain(Shot.diagnostics[key+'_horizontal_direct_stats'],
                                             GetAttr('sigma'))

