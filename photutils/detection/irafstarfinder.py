
import warnings

import numpy as np
from astropy.utils import lazyproperty
from astropy.utils.exceptions import AstropyDeprecationWarning

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

__all__ = ['IRAFStarFinder']


class IRAFStarFinder(StarFinderBase):

    @deprecated_positional_kwargs(since='3.0', until='4.0')
    @deprecated_renamed_argument('brightest', 'n_brightest', '3.0',
                                 until='4.0')
    @deprecated_renamed_argument('peakmax', 'peak_max', '3.0', until='4.0')
    def __init__(self, threshold, fwhm, sigma_radius=1.5,
                 minsep_fwhm=_DEPR_DEFAULT,
                 sharplo=_DEPR_DEFAULT, sharphi=_DEPR_DEFAULT,
                 roundlo=_DEPR_DEFAULT, roundhi=_DEPR_DEFAULT,
                 exclude_border=False, n_brightest=None, peak_max=None,
                 xycoords=None, min_separation=None, *,
                 sharpness_range=(0.5, 2.0),
                 roundness_range=(0.0, 0.2)):

        inputs = (threshold, peak_max)
        names = ('threshold', 'peak_max')
        check_units(inputs, names)

        if not isscalar(fwhm):
            msg = 'fwhm must be a scalar value'
            raise TypeError(msg)

        sharpness_range = _handle_deprecated_range(
            sharplo, sharphi, sharpness_range,
            'sharp', 'sharpness_range', (0.5, 2.0))
        roundness_range = _handle_deprecated_range(
            roundlo, roundhi, roundness_range,
            'round', 'roundness_range', (0.0, 0.2))

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

        if minsep_fwhm is not _DEPR_DEFAULT:
            msg = ("The 'minsep_fwhm' parameter is deprecated "
                   'and will be removed in a future version. Use '
                   "'min_separation' instead.")
            warnings.warn(msg, AstropyDeprecationWarning)
            if minsep_fwhm < 0:
                msg = 'minsep_fwhm must be >= 0'
                raise ValueError(msg)
            if min_separation is None:
                min_separation = max(2, int((fwhm * minsep_fwhm) + 0.5))

        self.threshold = threshold
        self.fwhm = fwhm
        self.sigma_radius = sigma_radius
        self.sharpness_range = sharpness_range
        self.roundness_range = roundness_range
        self.exclude_border = exclude_border
        self.n_brightest = _validate_n_brightest(n_brightest)
        self.peak_max = peak_max

        if xycoords is not None:
            xycoords = np.asarray(xycoords)
            if xycoords.ndim != 2 or xycoords.shape[1] != 2:
                msg = 'xycoords must be shaped as an Nx2 array'
                raise ValueError(msg)
        self.xycoords = xycoords

        self.kernel = _StarFinderKernel(self.fwhm, ratio=1.0, theta=0.0,
                                        sigma_radius=self.sigma_radius)

        if min_separation is not None:
            if min_separation < 0:
                msg = 'min_separation must be >= 0'
                raise ValueError(msg)
            self.min_separation = min_separation
        else:
            self.min_separation = 2.5 * self.fwhm

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


class _IRAFStarFinderCatalog(StarFinderCatalogBase):

    def __init__(self, data, convolved_data, xypos, kernel, *,
                 sharpness_range=(0.2, 1.0), roundness_range=(-1.0, 1.0),
                 n_brightest=None, peak_max=None):

        inputs = (data, convolved_data, peak_max)
        names = ('data', 'convolved_data', 'peak_max')
        check_units(inputs, names)

        super().__init__(data, xypos, kernel,
                         n_brightest=n_brightest,
                         peak_max=peak_max)

        self.convolved_data = convolved_data
        self.sharpness_range = sharpness_range
        self.roundness_range = roundness_range

        self.default_columns = ('id', 'x_centroid', 'y_centroid', 'fwhm',
                                'sharpness', 'roundness', 'orientation',
                                'n_pixels', 'peak', 'flux', 'mag')

    def _get_init_attributes(self):
        pass

    @lazyproperty
    def sky(self):
        pass

    @lazyproperty
    def cutout_data_nosub(self):
        pass

    @lazyproperty
    def cutout_data(self):
        pass

    @lazyproperty
    def n_pixels(self):
        pass

    @lazyproperty
    def cutout_xorigin(self):
        pass

    @lazyproperty
    def cutout_yorigin(self):
        pass

    @lazyproperty
    def x_centroid(self):
        pass

    @lazyproperty
    def y_centroid(self):
        pass

    @lazyproperty
    def sharpness(self):
        pass

    def apply_filters(self):
        pass
