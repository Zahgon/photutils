
import warnings

import numpy as np
from astropy.utils.exceptions import AstropyUserWarning

from photutils.isophote.fitter import (DEFAULT_CONVERGENCE, DEFAULT_FFLAG,
                                       DEFAULT_MAXGERR, DEFAULT_MAXIT,
                                       DEFAULT_MINIT, CentralEllipseFitter,
                                       EllipseFitter)
from photutils.isophote.geometry import EllipseGeometry
from photutils.isophote.integrator import BILINEAR
from photutils.isophote.isophote import Isophote, IsophoteList
from photutils.isophote.sample import CentralEllipseSample, EllipseSample
from photutils.utils._deprecation import (deprecated_positional_kwargs,
                                          deprecated_renamed_argument)

__all__ = ['Ellipse']


class Ellipse:

    @deprecated_positional_kwargs(since='3.0', until='4.0')
    def __init__(self, image, geometry=None, threshold=0.1):
        self.image = image

        if geometry is not None:
            self._geometry = geometry
        else:
            _x0 = image.shape[1] / 2
            _y0 = image.shape[0] / 2
            self._geometry = EllipseGeometry(_x0, _y0, 10.0, eps=0.2,
                                             pa=np.pi / 2)
        self.set_threshold(threshold)

    def set_threshold(self, threshold):
        """
        Modify the threshold value used by the centerer.

        Parameters
        ----------
        threshold : float
            The new threshold value to use.
        """
        self._geometry.centerer_threshold = threshold

    @deprecated_positional_kwargs(since='3.0', until='4.0')
    @deprecated_renamed_argument('nclip', 'n_clip', '3.0', until='4.0')
    def fit_image(self, sma0=None, minsma=0.0, maxsma=None, step=0.1,
                  conver=DEFAULT_CONVERGENCE, minit=DEFAULT_MINIT,
                  maxit=DEFAULT_MAXIT, fflag=DEFAULT_FFLAG,
                  maxgerr=DEFAULT_MAXGERR, sclip=3.0, n_clip=0,
                  integrmode=BILINEAR, linear=None, maxrit=None,
                  fix_center=False, fix_pa=False, fix_eps=False):
        pass

    @deprecated_positional_kwargs(since='3.0', until='4.0')
    @deprecated_renamed_argument('nclip', 'n_clip', '3.0', until='4.0')
    def fit_isophote(self, sma, step=0.1, conver=DEFAULT_CONVERGENCE,
                     minit=DEFAULT_MINIT, maxit=DEFAULT_MAXIT,
                     fflag=DEFAULT_FFLAG, maxgerr=DEFAULT_MAXGERR,
                     sclip=3.0, n_clip=0, integrmode=BILINEAR,
                     linear=False, maxrit=None, noniterate=False,
                     going_inwards=False, isophote_list=None):
        pass

    def _iterative(self, sma, step, linear, geometry, sclip, n_clip,
                   integrmode, conver, minit, maxit, fflag, maxgerr, *,
                   going_inwards=False):
        pass

    def _non_iterative(self, sma, step, linear, geometry, sclip, n_clip,
                       integrmode):
        pass

    @staticmethod
    def _fix_last_isophote(isophote_list, index):
        pass
