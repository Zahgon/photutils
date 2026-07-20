
import numpy as np
from astropy.utils import lazyproperty
from scipy.interpolate import PchipInterpolator

from photutils.profiles.core import ProfileBase

__all__ = ['CurveOfGrowth', 'EllipticalCurveOfGrowth',
           'EnsquaredCurveOfGrowth']


class CurveOfGrowth(ProfileBase):

    _ylabel = 'Curve of Growth'

    def __init__(self, data, xycen, radii, *, error=None, mask=None,
                 method='exact', subpixels=5):

        if np.min(radii) <= 0:
            msg = 'radii must be > 0'
            raise ValueError(msg)

        super().__init__(data, xycen, radii, error=error, mask=mask,
                         method=method, subpixels=subpixels)

    @lazyproperty
    def radius(self):
        """
        The profile radius in pixels as a 1D `~numpy.ndarray`.

        This is the same as the input ``radii``.

        Note that these are the radii of the circular apertures used
        to measure the profile. Thus, they are the radial values that
        enclose the given flux. They can be used directly to measure the
        encircled energy/flux at a given radius.
        """
        return self.radii

    @lazyproperty
    def apertures(self):
        pass

    @lazyproperty
    def _photometry(self):
        pass

    @lazyproperty
    def profile(self):
        pass

    @lazyproperty
    def profile_error(self):
        pass

    @lazyproperty
    def area(self):
        pass

    def calc_ee_at_radius(self, radius):
        pass

    def calc_radius_at_ee(self, ee):
        pass


class EnsquaredCurveOfGrowth(ProfileBase):

    _xlabel = 'Half-Size (pixels)'
    _ylabel = 'Ensquared Curve of Growth'

    def __init__(self, data, xycen, half_sizes, *, error=None, mask=None,
                 method='exact', subpixels=5):

        if np.min(half_sizes) <= 0:
            msg = 'half_sizes must be > 0'
            raise ValueError(msg)

        super().__init__(data, xycen, half_sizes, error=error, mask=mask,
                         method=method, subpixels=subpixels)
        self.half_sizes = self.radii

    def __repr__(self):
        cls_name = self.__class__.__name__
        n_half_sizes = len(self.half_sizes)
        normalized = self.normalization_value != 1.0
        return (f'{cls_name}(xycen={self.xycen}, '
                f'n_half_sizes={n_half_sizes}, '
                f'normalized={normalized})')

    @lazyproperty
    def half_size(self):
        pass

    @lazyproperty
    def radius(self):
        """
        The profile half-sizes (half side lengths) in pixels as a 1D
        `~numpy.ndarray`.

        This is an alias for `half_size`.
        """
        return self.half_sizes

    @lazyproperty
    def apertures(self):
        pass

    @lazyproperty
    def _photometry(self):
        pass

    @lazyproperty
    def profile(self):
        pass

    @lazyproperty
    def profile_error(self):
        pass

    @lazyproperty
    def area(self):
        pass

    def calc_ee_at_half_size(self, half_size):
        pass

    def calc_half_size_at_ee(self, ee):
        pass


class EllipticalCurveOfGrowth(ProfileBase):

    _xlabel = 'Semimajor Axis (pixels)'
    _ylabel = 'Elliptical Curve of Growth'

    def __init__(self, data, xycen, radii, axis_ratio, *, theta=0.0,
                 error=None, mask=None, method='exact', subpixels=5):

        if np.min(radii) <= 0:
            msg = 'radii must be > 0'
            raise ValueError(msg)

        if not 0 < axis_ratio <= 1:
            msg = 'axis_ratio must be in the range 0 < axis_ratio <= 1'
            raise ValueError(msg)

        self.axis_ratio = axis_ratio
        self.theta = theta

        super().__init__(data, xycen, radii, error=error, mask=mask,
                         method=method, subpixels=subpixels)

    @lazyproperty
    def radius(self):
        """
        The profile semimajor-axis lengths in pixels as a 1D
        `~numpy.ndarray`.

        This is the same as the input ``radii``.

        Note that these are the semimajor-axis lengths of the elliptical
        apertures used to measure the profile. Thus, they are the
        semimajor-axis values that enclose the given flux.
        """
        return self.radii

    @lazyproperty
    def apertures(self):
        pass

    @lazyproperty
    def _photometry(self):
        pass

    @lazyproperty
    def profile(self):
        pass

    @lazyproperty
    def profile_error(self):
        pass

    @lazyproperty
    def area(self):
        pass

    def calc_ee_at_radius(self, radius):
        pass

    def calc_radius_at_ee(self, ee):
        pass
