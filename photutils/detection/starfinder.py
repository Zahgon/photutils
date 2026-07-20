
import warnings

import numpy as np
from astropy.utils import lazyproperty

from photutils.detection.core import (StarFinderBase, StarFinderCatalogBase,
                                      _validate_n_brightest)
from photutils.utils._convolution import _filter_data
from photutils.utils._deprecation import (deprecated_positional_kwargs,
                                          deprecated_renamed_argument)
from photutils.utils._quantity_helpers import check_units
from photutils.utils._repr import make_repr
from photutils.utils.exceptions import NoDetectionsWarning

__all__ = ['StarFinder']


class StarFinder(StarFinderBase):

    @deprecated_positional_kwargs(since='3.0', until='4.0')
    @deprecated_renamed_argument('brightest', 'n_brightest', '3.0',
                                 until='4.0')
    @deprecated_renamed_argument('peakmax', 'peak_max', '3.0', until='4.0')
    def __init__(self, threshold, kernel, min_separation=None,
                 exclude_border=False, n_brightest=None, peak_max=None):

        check_units((threshold, peak_max), ('threshold', 'peak_max'))

        self.threshold = threshold

        kernel = np.asarray(kernel)
        if kernel.ndim != 2:
            msg = 'kernel must be a 2D array'
            raise ValueError(msg)
        self.kernel = kernel

        if min_separation is not None:
            if min_separation < 0:
                msg = 'min_separation must be >= 0'
                raise ValueError(msg)
            self.min_separation = min_separation
        else:
            self.min_separation = 2.5 * (min(self.kernel.shape) // 2)
        self.exclude_border = exclude_border
        self.n_brightest = _validate_n_brightest(n_brightest)
        self.peak_max = peak_max

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


class _StarFinderCatalog(StarFinderCatalogBase):

    def __init__(self, data, xypos, kernel, *, n_brightest=None,
                 peak_max=None):
        super().__init__(data, xypos, kernel,
                         n_brightest=n_brightest,
                         peak_max=peak_max)
        self.default_columns = ('id', 'x_centroid', 'y_centroid', 'fwhm',
                                'roundness', 'orientation', 'max_value',
                                'flux', 'mag')

    def _get_init_attributes(self):
        pass

    @lazyproperty
    def cutout_data(self):
        pass

    @lazyproperty
    def max_value(self):
        pass

    @lazyproperty
    def x_centroid(self):
        pass

    @lazyproperty
    def y_centroid(self):
        pass

    def apply_filters(self):
        pass
