
import abc
import warnings

import numpy as np
from astropy.utils import lazyproperty
from astropy.utils.exceptions import AstropyUserWarning

from photutils.utils._deprecation import deprecated_positional_kwargs
from photutils.utils._quantity_helpers import process_quantities
from photutils.utils._stats import nanmax, nansum

__all__ = ['ProfileBase']


class ProfileBase(metaclass=abc.ABCMeta):

    _xlabel = 'Radius (pixels)'
    _ylabel = 'Profile'

    def __init__(self, data, xycen, radii, *, error=None, mask=None,
                 method='exact', subpixels=5):

        (data, error), unit = process_quantities((data, error),
                                                 ('data', 'error'))

        if error is not None and error.shape != data.shape:
            msg = 'error must have the same shape as data'
            raise ValueError(msg)

        self.data = data
        self.unit = unit
        self.xycen = xycen
        self.radii = self._validate_radii(radii)
        self.error = error
        self.mask = self._compute_mask(data, error, mask)
        self.method = method
        self.subpixels = subpixels
        self.normalization_value = 1.0

    def _validate_radii(self, radii):
        """
        Validate and return the radii array.
        """
        radii = np.array(radii)
        if radii.ndim != 1 or radii.size < 2:
            msg = 'radii must be a 1D array and have at least two values'
            raise ValueError(msg)
        if radii.min() < 0:
            msg = 'minimum radii must be >= 0'
            raise ValueError(msg)

        if not np.all(radii[1:] > radii[:-1]):
            msg = 'radii must be strictly increasing'
            raise ValueError(msg)

        return radii

    def _compute_mask(self, data, error, mask):
        """
        Compute the mask array, automatically masking non-finite data or
        error values.
        """
        badmask = ~np.isfinite(data)
        if error is not None:
            badmask |= ~np.isfinite(error)
        if mask is not None:
            if mask.shape != data.shape:
                msg = 'mask must have the same shape as data'
                raise ValueError(msg)
            badmask &= ~mask
            combined_mask = mask | badmask  # all masked pixels
        else:
            combined_mask = badmask

        if np.any(badmask):
            msg = ('Input data contains non-finite values (e.g., NaN '
                   'or inf) that were automatically masked.')
            warnings.warn(msg, AstropyUserWarning)

        return combined_mask

    @property
    @abc.abstractmethod
    def radius(self):
        """
        The profile radius in pixels as a 1D `~numpy.ndarray`.
        """

    @property
    @abc.abstractmethod
    def profile(self):
        """
        The radial profile as a 1D `~numpy.ndarray`.
        """

    @property
    @abc.abstractmethod
    def profile_error(self):
        """
        The profile errors as a 1D `~numpy.ndarray`.

        If no ``error`` array was provided, an empty array with shape
        ``(0,)`` is returned.
        """

    @lazyproperty
    def _circular_apertures(self):
        pass

    def _compute_photometry(self, apertures):
        pass

    @lazyproperty
    def _photometry(self):
        pass

    @deprecated_positional_kwargs(since='3.0', until='4.0')
    def normalize(self, method='max'):
        """
        Normalize the profile.

        Parameters
        ----------
        method : {'max', 'sum'}, optional
            The method used to normalize the profile:

            * ``'max'`` (default):
              The profile is normalized such that its maximum value is
              1.

            * ``'sum'``:
              The profile is normalized such that its sum (integral) is
              1.
        """
        if method == 'max':
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', RuntimeWarning)
                normalization = nanmax(self.profile)
        elif method == 'sum':
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', RuntimeWarning)
                normalization = nansum(self.profile)
        else:
            msg = "invalid method, must be 'max' or 'sum'"
            raise ValueError(msg)

        if normalization == 0 or not np.isfinite(normalization):
            msg = ('The profile cannot be normalized because the max or '
                   'sum is zero or non-finite.')
            warnings.warn(msg, AstropyUserWarning)
        else:
            self.normalization_value *= normalization

            self.__dict__['profile'] = self.profile / normalization
            self.__dict__['profile_error'] = self.profile_error / normalization
            self._normalize_hook(normalization)

    def _normalize_hook(self, normalization):  # noqa: B027
        """
        Hook called by `normalize` after normalizing ``profile`` and
        ``profile_error``.

        This hook is only called when normalization succeeds (i.e., when
        the normalization value is non-zero and finite).

        Subclasses can override this to normalize additional lazy
        properties (e.g., ``data_profile``).

        Parameters
        ----------
        normalization : float
            The normalization value applied to the profile.
        """

    def unnormalize(self):
        pass

    def _unnormalize_hook(self):  # noqa: B027
        """
        Hook called by `unnormalize` after unnormalizing ``profile`` and
        ``profile_error``, but before resetting ``normalization_value``.

        Subclasses can override this to unnormalize additional lazy
        properties (e.g., ``data_profile``).
        """

    @staticmethod
    def _trim_to_monotonic(xarr, profile, name):
        pass

    def __repr__(self):
        cls_name = self.__class__.__name__
        n_radii = len(self.radii)
        normalized = self.normalization_value != 1.0
        return (f'{cls_name}(xycen={self.xycen}, n_radii={n_radii}, '
                f'normalized={normalized})')

    @deprecated_positional_kwargs(since='3.0', until='4.0')
    def plot(self, ax=None, **kwargs):
        pass

    @deprecated_positional_kwargs(since='3.0', until='4.0')
    def plot_error(self, ax=None, **kwargs):
        pass
