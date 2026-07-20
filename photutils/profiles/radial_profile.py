
import warnings

import numpy as np
from astropy.modeling.fitting import TRFLSQFitter
from astropy.modeling.models import Gaussian1D, Moffat1D
from astropy.stats import gaussian_sigma_to_fwhm
from astropy.utils import lazyproperty
from astropy.utils.exceptions import AstropyUserWarning

from photutils.profiles.core import ProfileBase

__all__ = ['RadialProfile']


class RadialProfile(ProfileBase):

    _ylabel = 'Radial Profile'

    _fit_properties = ('_profile_nanmask', 'gaussian_fit',
                       'gaussian_profile', 'gaussian_fwhm', 'moffat_fit',
                       'moffat_profile', 'moffat_fwhm')

    @lazyproperty
    def radius(self):
        """
        The profile radius (bin centers) in pixels as a 1D
        `~numpy.ndarray`.

        The returned radius values are defined as the arithmetic means
        of the input radial-bins edges (``radii``).

        For logarithmically-spaced input ``radii``, one could instead
        use a radius array defined using the geometric mean of the bin
        edges, i.e. ``np.sqrt(radii[:-1] * radii[1:])``.
        """
        return (self.radii[:-1] + self.radii[1:]) / 2

    @lazyproperty
    def apertures(self):
        pass

    @lazyproperty
    def _flux(self):
        pass

    @lazyproperty
    def _flux_err(self):
        pass

    @lazyproperty
    def area(self):
        pass

    @lazyproperty
    def profile(self):
        pass

    @lazyproperty
    def profile_error(self):
        pass

    @lazyproperty
    def _profile_nanmask(self):
        pass

    @lazyproperty
    def gaussian_fit(self):
        pass

    @lazyproperty
    def gaussian_profile(self):
        pass

    @lazyproperty
    def gaussian_fwhm(self):
        pass

    @lazyproperty
    def moffat_fit(self):
        pass

    @lazyproperty
    def moffat_profile(self):
        pass

    @lazyproperty
    def moffat_fwhm(self):
        pass

    @lazyproperty
    def _data_profile(self):
        pass

    @lazyproperty
    def data_radius(self):
        pass

    @lazyproperty
    def data_profile(self):
        pass

    def _invalidate_fit_cache(self):
        """
        Remove cached Gaussian and Moffat fit lazy properties so they
        are recomputed on next access using the current profile.
        """
        for key in self._fit_properties:
            self.__dict__.pop(key, None)

    def _normalize_hook(self, normalization):
        """
        Also normalize ``data_profile`` if it has been computed, and
        invalidate fit caches so they are recomputed on next access.
        """
        if 'data_profile' in self.__dict__:
            self.__dict__['data_profile'] = self.data_profile / normalization
        self._invalidate_fit_cache()

    def _unnormalize_hook(self):
        pass
