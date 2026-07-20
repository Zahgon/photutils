
import abc
import inspect
import math
import warnings

import astropy.units as u
import numpy as np
from astropy.stats import gaussian_fwhm_to_sigma
from astropy.utils import lazyproperty
from astropy.utils.exceptions import AstropyDeprecationWarning

from photutils.detection.peakfinder import find_peaks
from photutils.utils._deprecation import (create_empty_deprecated_qtable,
                                          deprecated_getattr,
                                          deprecated_positional_kwargs,
                                          deprecated_renamed_argument)
from photutils.utils._misc import _get_meta
from photutils.utils._quantity_helpers import check_units
from photutils.utils._repr import make_repr
from photutils.utils.cutouts import _make_cutouts
from photutils.utils.exceptions import NoDetectionsWarning

__all__ = ['StarFinderBase', 'StarFinderCatalogBase']

_DEPRECATED_ATTRIBUTES: dict = {
    'xcentroid': 'x_centroid',
    'ycentroid': 'y_centroid',
    'cutout_xcentroid': 'cutout_x_centroid',
    'cutout_ycentroid': 'cutout_y_centroid',
    'pa': 'orientation',
    'npix': 'n_pixels',
}


class StarFinderBase(metaclass=abc.ABCMeta):

    @deprecated_positional_kwargs(since='3.0', until='4.0')
    def __call__(self, data, mask=None):
        """
        Find stars in an astronomical image.

        Parameters
        ----------
        data : 2D array_like
            The 2D image array.

        mask : 2D bool array, optional
            A boolean mask with the same shape as ``data``, where a
            `True` value indicates the corresponding element of ``data``
            is masked. Masked pixels are ignored when searching for
            stars.

        Returns
        -------
        table : `~astropy.table.Table` or `None`
            A table of found stars. If no stars are found then `None` is
            returned.
        """
        return self.find_stars(data, mask=mask)

    @staticmethod
    def _find_stars(convolved_data, kernel, threshold, *, min_separation=0.0,
                    mask=None, exclude_border=False):
        pass

    @abc.abstractmethod
    def find_stars(self, data, *, mask=None):
        """
        Find stars in an astronomical image.

        Parameters
        ----------
        data : 2D array_like
            The 2D image array.

        mask : 2D bool array, optional
            A boolean mask with the same shape as ``data``, where a
            `True` value indicates the corresponding element of ``data``
            is masked. Masked pixels are ignored when searching for
            stars.

        Returns
        -------
        table : `~astropy.table.Table` or `None`
            A table of found stars. If no stars are found then `None` is
            returned.
        """


class StarFinderCatalogBase(metaclass=abc.ABCMeta):

    @deprecated_renamed_argument('brightest', 'n_brightest', '3.0',
                                 until='4.0')
    @deprecated_renamed_argument('peakmax', 'peak_max', '3.0', until='4.0')
    def __init__(self, data, xypos, kernel, *, n_brightest=None,
                 peak_max=None):
        check_units((data, peak_max), ('data', 'peak_max'))

        self.data = data
        unit = data.unit if isinstance(data, u.Quantity) else None
        self.unit = unit
        self.kernel = kernel
        self.cutout_shape = kernel.shape

        self.xypos = np.atleast_2d(xypos)
        self.n_brightest = n_brightest
        self.peak_max = peak_max
        self.default_columns = ()

        self.id = np.arange(len(self)) + 1

    def __repr__(self):
        params = ('nsources',)
        overrides = {'nsources': len(self)}
        return make_repr(self, params, brackets=True, overrides=overrides)

    def __str__(self):
        params = ('nsources',)
        overrides = {'nsources': len(self)}
        return make_repr(self, params, overrides=overrides, long=True)

    def __len__(self):
        return len(self.xypos)

    def __getitem__(self, index):
        """
        Index or slice the catalog.

        This method should be overridden in subclasses to handle
        class-specific attributes.
        """

        newcls = object.__new__(self.__class__)

        init_attr = self._get_init_attributes()
        for attr in init_attr:
            setattr(newcls, attr, getattr(self, attr))

        attr = 'xypos'
        value = getattr(self, attr)[index]
        setattr(newcls, attr, np.atleast_2d(value))

        keys = set(self.__dict__.keys()) & set(self._lazyproperties)
        keys.add('id')
        for key in keys:
            value = self.__dict__[key]

            if np.isscalar(value):
                continue

            value = np.atleast_1d(value[index])

            newcls.__dict__[key] = value

        return newcls

    def _get_init_attributes(self):
        pass

    @property
    def _lazyproperties(self):
        pass

    @lazyproperty
    def isscalar(self):
        """
        Whether the instance is scalar (e.g., a single source).
        """
        return self.xypos.shape == (1, 2)

    def reset_ids(self):
        pass

    def make_cutouts(self, data):
        pass

    @lazyproperty
    def cutout_data(self):
        pass

    @lazyproperty
    def moments(self):
        pass

    @lazyproperty
    def moments_central(self):
        pass

    @lazyproperty
    def cutout_centroid(self):
        pass

    @lazyproperty
    def cutout_x_centroid(self):
        pass

    @lazyproperty
    def cutout_y_centroid(self):
        pass

    @property
    @abc.abstractmethod
    def x_centroid(self):
        """
        Object centroid in the x direction.

        This property must be implemented in subclasses.
        """

    @property
    @abc.abstractmethod
    def y_centroid(self):
        """
        Object centroid in the y direction.

        This property must be implemented in subclasses.
        """

    def __getattr__(self, name):
        return deprecated_getattr(self, name, _DEPRECATED_ATTRIBUTES,
                                  since='3.0', until='4.0')

    @lazyproperty
    def mu_sum(self):
        pass

    @lazyproperty
    def mu_diff(self):
        pass

    @lazyproperty
    def fwhm(self):
        pass

    @lazyproperty
    def orientation(self):
        pass

    @lazyproperty
    def roundness(self):
        pass

    @lazyproperty
    def peak(self):
        pass

    @lazyproperty
    def flux(self):
        pass

    @lazyproperty
    def mag(self):
        pass

    def select_brightest(self):
        pass

    def _filter_finite(self, attrs, *, initial_mask=None,
                       skip_attrs=()):
        pass

    def _filter_bounds(self, bounds, *, initial_mask=None, peakattr='peak'):
        pass

    @abc.abstractmethod
    def apply_filters(self):
        """
        Filter the catalog.

        This method must be implemented in subclasses to apply
        algorithm-specific filtering criteria.
        """

    def apply_all_filters(self):
        pass

    def to_table(self, *, columns=None):
        pass


class _StarFinderKernel:

    def __init__(self, fwhm, *, ratio=1.0, theta=0.0, sigma_radius=1.5,
                 normalize_zerosum=True):

        if np.ndim(fwhm) != 0:
            msg = 'fwhm must be a scalar value'
            raise TypeError(msg)

        if fwhm <= 0:
            msg = 'fwhm must be positive'
            raise ValueError(msg)

        if ratio <= 0 or ratio > 1:
            msg = 'ratio must be > 0 and <= 1.0'
            raise ValueError(msg)

        if sigma_radius <= 0:
            msg = 'sigma_radius must be positive'
            raise ValueError(msg)

        self.fwhm = fwhm
        self.ratio = ratio
        self.theta = theta % 360.0
        self.sigma_radius = sigma_radius
        self.x_sigma = self.fwhm * gaussian_fwhm_to_sigma
        self.y_sigma = self.x_sigma * self.ratio

        theta_radians = np.deg2rad(self.theta)
        cost = np.cos(theta_radians)
        sint = np.sin(theta_radians)
        x_sigma2 = self.x_sigma**2
        y_sigma2 = self.y_sigma**2

        a = (cost**2 / (2.0 * x_sigma2)) + (sint**2 / (2.0 * y_sigma2))
        b = 0.5 * cost * sint * ((1.0 / x_sigma2) - (1.0 / y_sigma2))
        c = (sint**2 / (2.0 * x_sigma2)) + (cost**2 / (2.0 * y_sigma2))

        f = self.sigma_radius**2 / 2.0
        denom = (a * c) - b**2

        nx = 2 * int(max(2, math.sqrt(c * f / denom))) + 1
        ny = 2 * int(max(2, math.sqrt(a * f / denom))) + 1

        self.x_radius = nx // 2
        self.y_radius = ny // 2

        xc = self.x_radius
        yc = self.y_radius
        yy, xx = np.mgrid[0:ny, 0:nx]
        circular_radius = np.sqrt((xx - xc)**2 + (yy - yc)**2)
        elliptical_radius = (a * (xx - xc)**2
                             + 2.0 * b * (xx - xc) * (yy - yc)
                             + c * (yy - yc)**2)

        self.mask = np.where(
            (elliptical_radius <= f)
            | (circular_radius <= 2.0), 1, 0).astype(int)
        self.n_pixels = self.mask.sum()

        self.gaussian_kernel_unmasked = np.exp(-elliptical_radius)
        gaussian_kernel = self.gaussian_kernel_unmasked * self.mask

        denom = ((gaussian_kernel**2).sum()
                 - (gaussian_kernel.sum()**2 / self.n_pixels))
        self.rel_err = 1.0 / np.sqrt(denom)

        if normalize_zerosum:
            self.data = ((gaussian_kernel
                          - (gaussian_kernel.sum() / self.n_pixels))
                         / denom) * self.mask
        else:
            self.data = gaussian_kernel

        self.shape = self.data.shape

    def __repr__(self):
        params = ('fwhm', 'ratio', 'theta', 'sigma_radius')
        return make_repr(self, params)

    def __str__(self):
        params = ('fwhm', 'ratio', 'theta', 'sigma_radius')
        return make_repr(self, params, long=True)


def _validate_n_brightest(n_brightest):
    """
    Validate the ``n_brightest`` parameter.

    It must be >0 and an integer.

    Parameters
    ----------
    n_brightest : int, None, or bool
        The number of brightest sources to select. If `None`, all
        sources are selected. If a boolean is passed, a `TypeError` is
        raised.
    """
    if n_brightest is not None:
        if isinstance(n_brightest, bool):
            msg = 'n_brightest must be an integer'
            raise TypeError(msg)
        if n_brightest <= 0:
            msg = 'n_brightest must be > 0'
            raise ValueError(msg)
        bright_int = int(n_brightest)
        if bright_int != n_brightest:
            msg = 'n_brightest must be an integer'
            raise ValueError(msg)
        n_brightest = bright_int
    return n_brightest


def _handle_deprecated_range(old_lower, old_upper, new_range,
                             old_name, new_name, default_range):
    """
    Handle deprecated lower/upper bound parameters replaced by a single
    range parameter.

    Parameters
    ----------
    old_lower : float or `_DeprecatedDefault`
        The deprecated lower-bound parameter value.

    old_upper : float or `_DeprecatedDefault`
        The deprecated upper-bound parameter value.

    new_range : tuple of 2 floats or `None`
        The new range parameter value.

    old_name : str
        The base name of the deprecated parameters (e.g., ``'sharp'``
        for ``'sharplo'`` / ``'sharphi'``).

    new_name : str
        The name of the new range parameter (e.g.,
        ``'sharpness_range'``).

    default_range : tuple of 2 floats
        The default range values when ``new_range`` is `None`.

    Returns
    -------
    result : tuple of 2 floats or `None`
        The resolved range.
    """
    if old_lower is not _DEPR_DEFAULT or old_upper is not _DEPR_DEFAULT:
        msg = (f"The '{old_name}lo' and '{old_name}hi' parameters are "
               'deprecated and will be removed in a future version. '
               f"Use '{new_name}=(lower, upper)' instead.")
        warnings.warn(msg, AstropyDeprecationWarning)
        _default = new_range if new_range is not None else default_range
        lower = (old_lower if old_lower is not _DEPR_DEFAULT
                 else _default[0])
        upper = (old_upper if old_upper is not _DEPR_DEFAULT
                 else _default[1])
        return (lower, upper)
    return new_range


class _DeprecatedDefault:

    def __repr__(self):
        return '<deprecated>'


_DEPR_DEFAULT = _DeprecatedDefault()
