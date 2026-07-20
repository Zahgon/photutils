
import math

import numpy as np
from astropy import log

from photutils.isophote.harmonics import (first_and_second_harmonic_function,
                                          fit_first_and_second_harmonics)
from photutils.isophote.isophote import CentralPixel, Isophote
from photutils.isophote.sample import EllipseSample

__all__ = ['EllipseFitter']

__doctest_skip__ = ['EllipseFitter.fit']


PI2 = np.pi / 2
MAX_EPS = 0.95
MIN_EPS = 0.05

DEFAULT_CONVERGENCE = 0.05
DEFAULT_MINIT = 10
DEFAULT_MAXIT = 50
DEFAULT_FFLAG = 0.7
DEFAULT_MAXGERR = 0.5


class EllipseFitter:

    def __init__(self, sample):
        self._sample = sample

    def fit(self, *, conver=DEFAULT_CONVERGENCE, minit=DEFAULT_MINIT,
            maxit=DEFAULT_MAXIT, fflag=DEFAULT_FFLAG, maxgerr=DEFAULT_MAXGERR,
            going_inwards=False):
        pass

    @staticmethod
    def _check_conditions(sample, maxgerr, going_inwards, lexceed):
        pass


class _ParameterCorrector:
    def correct(self, sample, harmonic):
        raise NotImplementedError


class _PositionCorrector(_ParameterCorrector):
    @staticmethod
    def finalize_correction(dx, dy, sample):
        pass


class _PositionCorrector0(_PositionCorrector):
    def correct(self, sample, harmonic):
        pass


class _PositionCorrector1(_PositionCorrector):
    def correct(self, sample, harmonic):
        pass


class _AngleCorrector(_ParameterCorrector):
    def correct(self, sample, harmonic):
        pass


class _EllipticityCorrector(_ParameterCorrector):
    def correct(self, sample, harmonic):
        pass


_CORRECTORS = [_PositionCorrector0(), _PositionCorrector1(),
               _AngleCorrector(), _EllipticityCorrector()]


class CentralEllipseFitter(EllipseFitter):

    def fit(self, **kwargs):  # noqa: ARG002
        pass
