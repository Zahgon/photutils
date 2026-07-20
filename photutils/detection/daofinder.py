
import warnings

import astropy.units as u
import numpy as np
from astropy.utils import lazyproperty

from photutils.detection.core import (_DEPR_DEFAULT, StarFinderBase,
                                      StarFinderCatalogBase,
                                      _handle_deprecated_range,
                                      _StarFinderKernel, _validate_n_brightest)
from photutils.utils._convolution import _filter_data
from photutils.utils._deprecation import (deprecated_positional_kwargs,
                                          deprecated_renamed_argument)
from photutils.utils._quantity_helpers import check_units, isscalar
from photutils.utils._repr import make_repr
from photutils.utils.exceptions import NoDetectionsWarning

__all__ = ['DAOStarFinder']


class DAOStarFinder(StarFinderBase):

    @deprecated_positional_kwargs(since='3.0', until='4.0')
    @deprecated_renamed_argument('brightest', 'n_brightest', '3.0',
                                 until='4.0')
    @deprecated_renamed_argument('peakmax', 'peak_max', '3.0', until='4.0')
    def __init__(self, threshold, fwhm, ratio=1.0, theta=0.0,
                 sigma_radius=1.5, sharplo=_DEPR_DEFAULT,
                 sharphi=_DEPR_DEFAULT, roundlo=_DEPR_DEFAULT,
                 roundhi=_DEPR_DEFAULT, exclude_border=False,
                 n_brightest=None, peak_max=None, xycoords=None,
                 min_separation=None, scale_threshold=True, *,
                 sharpness_range=(0.2, 1.0),
                 roundness_range=(-1.0, 1.0)):

        inputs = (threshold, peak_max)
        names = ('threshold', 'peak_max')
        check_units(inputs, names)

        if not isscalar(fwhm):
            msg = 'fwhm must be a scalar value'
            raise TypeError(msg)

        sharpness_range = _handle_deprecated_range(
            sharplo, sharphi, sharpness_range,
            'sharp', 'sharpness_range', (0.2, 1.0))
        roundness_range = _handle_deprecated_range(
            roundlo, roundhi, roundness_range,
            'round', 'roundness_range', (-1.0, 1.0))

        if sharpness_range is not None:
            if np.ndim(sharpness_range) != 1 or np.size(sharpness_range) != 2:
                msg = ('sharpness_range must be a 2-element (lower, upper) '
                       'tuple or None')
                raise ValueError(msg)
            sharpness_range = tuple(sharpness_range)

        if roundness_range is not None:
            if np.ndim(roundness_range) != 1 or np.size(roundness_range) != 2:
                msg = ('roundness_range must be a 2-element (lower, upper) '
                       'tuple or None')
                raise ValueError(msg)
            roundness_range = tuple(roundness_range)

        self.threshold = threshold
        self.fwhm = fwhm
        self.ratio = ratio
        self.theta = theta % 360.0
        self.sigma_radius = sigma_radius
        self.sharpness_range = sharpness_range
        self.roundness_range = roundness_range
        self.exclude_border = exclude_border
        self.n_brightest = _validate_n_brightest(n_brightest)
        self.peak_max = peak_max

        if min_separation is not None:
            if min_separation < 0:
                msg = 'min_separation must be >= 0'
                raise ValueError(msg)
            self.min_separation = min_separation
        else:
            self.min_separation = 2.5 * self.fwhm

        if xycoords is not None:
            xycoords = np.asarray(xycoords)
            if xycoords.ndim != 2 or xycoords.shape[1] != 2:
                msg = 'xycoords must be shaped as an Nx2 array'
                raise ValueError(msg)
        self.xycoords = xycoords
        self.scale_threshold = scale_threshold

        self.kernel = _StarFinderKernel(self.fwhm,
                                        ratio=self.ratio,
                                        theta=self.theta,
                                        sigma_radius=self.sigma_radius)
        if self.scale_threshold:
            self.threshold_eff = self.threshold * self.kernel.rel_err
        else:
            self.threshold_eff = self.threshold

    def _repr_str_params(self):
        pass

    def __repr__(self):
        params, overrides = self._repr_str_params()
        return make_repr(self, params, overrides=overrides)

    def __str__(self):
        params, overrides = self._repr_str_params()
        return make_repr(self, params, overrides=overrides, long=True)

    def _get_raw_catalog(self, data, *, mask=None):
        pass

    @deprecated_positional_kwargs(since='3.0', until='4.0')
    def find_stars(self, data, mask=None):
        pass


class _DAOStarFinderCatalog(StarFinderCatalogBase):

    def __init__(self, data, convolved_data, xypos, threshold, kernel, *,
                 sharpness_range=(0.2, 1.0), roundness_range=(-1.0, 1.0),
                 n_brightest=None, peak_max=None, scale_threshold=True):

        inputs = (data, convolved_data, threshold, peak_max)
        names = ('data', 'convolved_data', 'threshold', 'peak_max')
        check_units(inputs, names)

        super().__init__(data, xypos, kernel,
                         n_brightest=n_brightest,
                         peak_max=peak_max)

        self.convolved_data = convolved_data
        self.threshold = threshold
        self.sharpness_range = sharpness_range
        self.roundness_range = roundness_range

        if scale_threshold:
            self.threshold_eff = threshold * kernel.rel_err
        else:
            self.threshold_eff = threshold
        self.cutout_center = tuple((size - 1) // 2 for size in kernel.shape)
        self.default_columns = ('id', 'x_centroid', 'y_centroid', 'sharpness',
                                'roundness1', 'roundness2', 'n_pixels',
                                'peak', 'flux', 'mag', 'daofind_mag')

    def _get_init_attributes(self):
        pass

    @lazyproperty
    def cutout_convdata(self):
        pass

    @lazyproperty
    def peak(self):
        pass

    @lazyproperty
    def convdata_peak(self):
        pass

    @lazyproperty
    def roundness1(self):
        pass

    @lazyproperty
    def sharpness(self):
        pass

    def _marginal_weights(self, axis):
        pass

    def _marginal_kernel_sums(self, wt, wts, axis, center, size):
        pass

    def _marginal_data_sums(self, wt, wts, axis, dxx, kern_sums):
        pass

    @staticmethod
    def _marginal_lstsq(kern_sums, data_sums, sigma, size):
        pass

    def daofind_marginal_fit(self, *, axis=0):
        pass

    @lazyproperty
    def dx_hx(self):
        pass

    @lazyproperty
    def dy_hy(self):
        pass

    @lazyproperty
    def dx(self):
        pass

    @lazyproperty
    def dy(self):
        pass

    @lazyproperty
    def hx(self):
        pass

    @lazyproperty
    def hy(self):
        pass

    @lazyproperty
    def x_centroid(self):
        pass

    @lazyproperty
    def y_centroid(self):
        pass

    @lazyproperty
    def roundness2(self):
        pass

    @lazyproperty
    def _threshold_eff_per_source(self):
        pass

    @lazyproperty
    def daofind_mag(self):
        pass

    @lazyproperty
    def n_pixels(self):
        pass

    def apply_filters(self):
        pass
