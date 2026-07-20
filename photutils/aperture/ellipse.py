
import math

import astropy.units as u
import numpy as np
from astropy.coordinates import Angle
from astropy.utils import lazyproperty

from photutils.aperture.attributes import (PixelPositions, PositiveScalar,
                                           PositiveScalarAngle, ScalarAngle,
                                           ScalarAngleOrValue,
                                           SkyCoordPositions)
from photutils.aperture.core import PixelAperture, SkyAperture
from photutils.aperture.mask import ApertureMask
from photutils.geometry import elliptical_overlap_grid
from photutils.utils._deprecation import (deprecated,
                                          deprecated_positional_kwargs)
from photutils.utils._wcs_helpers import (pixel_ellipse_to_sky_svd,
                                          sky_ellipse_to_pixel_svd)

__all__ = [
    'EllipticalAnnulus',
    'EllipticalAperture',
    'EllipticalMaskMixin',
    'SkyEllipticalAnnulus',
    'SkyEllipticalAperture',
]


@deprecated('3.0', until='4.0')
class EllipticalMaskMixin:  # pragma: no cover

    def to_mask(self, method='exact', subpixels=5):
        """
        Return a mask for the aperture.

        Parameters
        ----------
        method : {'exact', 'center', 'subpixel'}, optional
            The method used to determine the overlap of the aperture
            on the pixel grid. Not all options are available for all
            aperture types. Note that the more precise methods are
            generally slower. The following methods are available:

            * ``'exact'`` (default):
              The exact fractional overlap of the aperture and each
              pixel is calculated. The aperture weights will contain
              values between 0 and 1.

            * ``'center'``:
              A pixel is considered to be entirely in or out of the
              aperture depending on whether its center is in or out of
              the aperture. The aperture weights will contain values
              only of 0 (out) and 1 (in).

            * ``'subpixel'``:
              A pixel is divided into subpixels (see the ``subpixels``
              keyword), each of which are considered to be entirely in
              or out of the aperture depending on whether its center is
              in or out of the aperture. If ``subpixels=1``, this method
              is equivalent to ``'center'``. The aperture weights will
              contain values between 0 and 1.

        subpixels : int, optional
            For the ``'subpixel'`` method, resample pixels by this
            factor in each dimension. That is, each pixel is divided
            into ``subpixels**2`` subpixels. This keyword is ignored
            unless ``method='subpixel'``.

        Returns
        -------
        mask : `~photutils.aperture.ApertureMask` or list of \
                `~photutils.aperture.ApertureMask`
            A mask for the aperture. If the aperture is scalar then
            a single `~photutils.aperture.ApertureMask` is returned,
            otherwise a list of `~photutils.aperture.ApertureMask` is
            returned.
        """
        use_exact, subpixels = self._translate_mask_method(method, subpixels)

        if hasattr(self, 'a'):
            a = self.a
            b = self.b
        elif hasattr(self, 'a_in'):  # annulus
            a = self.a_out
            b = self.b_out
        else:
            msg = 'Cannot determine the aperture shape'
            raise ValueError(msg)

        masks = []
        for bbox, edges in zip(self._bbox, self._centered_edges, strict=True):
            ny, nx = bbox.shape
            theta_rad = self.theta.to(u.radian).value
            mask = elliptical_overlap_grid(edges[0], edges[1], edges[2],
                                           edges[3], nx, ny, a, b,
                                           theta_rad, use_exact, subpixels)

            if hasattr(self, 'a_in'):
                mask -= elliptical_overlap_grid(edges[0], edges[1], edges[2],
                                                edges[3], nx, ny, self.a_in,
                                                self.b_in, theta_rad,
                                                use_exact, subpixels)

            masks.append(ApertureMask(mask, bbox))

        if self.isscalar:
            return masks[0]

        return masks

    @staticmethod
    def _calc_extents(semimajor_axis, semiminor_axis, theta):
        pass


def _calc_ellipse_extents(semimajor_axis, semiminor_axis, theta):
    pass


class EllipticalAperture(PixelAperture):

    _params = ('positions', 'a', 'b', 'theta')
    positions = PixelPositions('The center pixel position(s).')
    a = PositiveScalar('The semimajor axis in pixels.')
    b = PositiveScalar('The semiminor axis in pixels.')
    theta = ScalarAngleOrValue('The counterclockwise rotation angle as an '
                               'angular Quantity or value in radians from '
                               'the positive x axis.')

    @deprecated_positional_kwargs(since='3.0', until='4.0')
    def __init__(self, positions, a, b, theta=0.0):
        self.positions = positions
        self.a = a
        self.b = b
        self.theta = theta

    @lazyproperty
    def _xy_extents(self):
        pass

    @lazyproperty
    def area(self):
        pass

    def _to_patch(self, *, origin=(0, 0), **kwargs):
        pass

    def _compute_overlap(self, edges, nx, ny, use_exact, subpixels):
        """
        Compute the overlap of the aperture on the pixel grid.

        Parameters
        ----------
        edges : list of 4 1D `~numpy.ndarray`
            The edges of the pixel grid in the form of
            ``[x_edges, y_edges, x_centers, y_centers]``.

        nx, ny : int
            The number of pixels in the x and y directions.

        use_exact : bool
            Whether to use the exact method for calculating the overlap.

        subpixels : int
            The number of subpixels to use in each dimension for the
            subpixel method.

        Returns
        -------
        overlap : 2D `~numpy.ndarray`
            The overlap of the aperture on the pixel grid. The values
            will be between 0 and 1, where 0 means no overlap and 1
            means full overlap.
        """
        theta_rad = self.theta.to(u.radian).value
        return elliptical_overlap_grid(edges[0], edges[1], edges[2],
                                       edges[3], nx, ny, self.a, self.b,
                                       theta_rad, use_exact, subpixels)

    def to_sky(self, wcs):
        pass


class EllipticalAnnulus(PixelAperture):

    _params = ('positions', 'a_in', 'a_out', 'b_in', 'b_out', 'theta')
    positions = PixelPositions('The center pixel position(s).')
    a_in = PositiveScalar('The inner semimajor axis in pixels.')
    a_out = PositiveScalar('The outer semimajor axis in pixels.')
    b_in = PositiveScalar('The inner semiminor axis in pixels.')
    b_out = PositiveScalar('The outer semiminor axis in pixels.')
    theta = ScalarAngleOrValue('The counterclockwise rotation angle as an '
                               'angular Quantity or value in radians from '
                               'the positive x axis.')

    @deprecated_positional_kwargs(since='3.0', until='4.0')
    def __init__(self, positions, a_in, a_out, b_out, b_in=None, theta=0.0):
        if not a_out > a_in:
            msg = "'a_out' must be greater than 'a_in'"
            raise ValueError(msg)

        self.positions = positions
        self.a_in = a_in
        self.a_out = a_out
        self.b_out = b_out

        if b_in is None:
            b_in = self.b_out * self.a_in / self.a_out
        elif not b_out > b_in:
            msg = "'b_out' must be greater than 'b_in'"
            raise ValueError(msg)
        self.b_in = b_in

        self.theta = theta

    @lazyproperty
    def _xy_extents(self):
        pass

    @lazyproperty
    def area(self):
        pass

    def _to_patch(self, *, origin=(0, 0), **kwargs):
        pass

    def _compute_overlap(self, edges, nx, ny, use_exact, subpixels):
        """
        Compute the overlap of the aperture on the pixel grid.

        Parameters
        ----------
        edges : list of 4 1D `~numpy.ndarray`
            The edges of the pixel grid in the form of
            ``[x_edges, y_edges, x_centers, y_centers]``.

        nx, ny : int
            The number of pixels in the x and y directions.

        use_exact : bool
            Whether to use the exact method for calculating the overlap.

        subpixels : int
            The number of subpixels to use in each dimension for the
            subpixel method.

        Returns
        -------
        overlap : 2D `~numpy.ndarray`
            The overlap of the aperture on the pixel grid. The values
            will be between 0 and 1, where 0 means no overlap and 1
            means full overlap.
        """
        theta_rad = self.theta.to(u.radian).value
        overlap = elliptical_overlap_grid(edges[0], edges[1], edges[2],
                                          edges[3], nx, ny, self.a_out,
                                          self.b_out, theta_rad,
                                          use_exact, subpixels)
        overlap -= elliptical_overlap_grid(edges[0], edges[1], edges[2],
                                           edges[3], nx, ny, self.a_in,
                                           self.b_in, theta_rad,
                                           use_exact, subpixels)
        return overlap

    def to_sky(self, wcs):
        pass


class SkyEllipticalAperture(SkyAperture):

    _params = ('positions', 'a', 'b', 'theta')
    positions = SkyCoordPositions('The center position(s) in sky coordinates.')
    a = PositiveScalarAngle('The semimajor axis in angular units.')
    b = PositiveScalarAngle('The semiminor axis in angular units.')
    theta = ScalarAngle('The position angle in angular units of the ellipse '
                        'semimajor axis.')

    @deprecated_positional_kwargs(since='3.0', until='4.0')
    def __init__(self, positions, a, b, theta=0.0 * u.deg):
        self.positions = positions
        self.a = a
        self.b = b
        self.theta = theta

    def to_pixel(self, wcs):
        """
        Convert the aperture to an `EllipticalAperture` object defined
        in pixel coordinates.

        Parameters
        ----------
        wcs : WCS object
            A world coordinate system (WCS) transformation that
            supports the `astropy shared interface for WCS
            <https://docs.astropy.org/en/stable/wcs/wcsapi.html>`_
            (e.g., `astropy.wcs.WCS`, `gwcs.wcs.WCS`).

        Returns
        -------
        aperture : `EllipticalAperture` object
            An `EllipticalAperture` object.

        Notes
        -----
        The aperture shape parameters are converted using the local WCS
        properties (pixel scale, rotation angle) evaluated at the first
        aperture position. Because aperture objects require scalar shape
        parameters, only a single reference position is used for the
        conversion. For apertures with multiple positions used with a
        WCS that has spatially-varying distortions, this may produce
        inaccurate results for positions far from the first position.
        """
        xpos, ypos = wcs.world_to_pixel(self.positions)
        positions = np.transpose((xpos, ypos))

        skypos = self.positions if self.isscalar else self.positions[0]
        sky_angle_rad = self.theta.to(u.rad).value
        _, pix_width, pix_height, pix_angle = sky_ellipse_to_pixel_svd(
            skypos, wcs,
            2 * self.a.to(u.arcsec).value,
            2 * self.b.to(u.arcsec).value,
            sky_angle_rad)

        a = pix_width / 2
        b = pix_height / 2
        return EllipticalAperture(positions=positions, a=a, b=b,
                                  theta=pix_angle)


class SkyEllipticalAnnulus(SkyAperture):

    _params = ('positions', 'a_in', 'a_out', 'b_in', 'b_out', 'theta')
    positions = SkyCoordPositions('The center position(s) in sky coordinates.')
    a_in = PositiveScalarAngle('The inner semimajor axis in angular units.')
    a_out = PositiveScalarAngle('The outer semimajor axis in angular units.')
    b_in = PositiveScalarAngle('The inner semiminor axis in angular units.')
    b_out = PositiveScalarAngle('The outer semiminor axis in angular units.')
    theta = ScalarAngle('The position angle in angular units of the ellipse '
                        'semimajor axis.')

    @deprecated_positional_kwargs(since='3.0', until='4.0')
    def __init__(self, positions, a_in, a_out, b_out, b_in=None,
                 theta=0.0 * u.deg):
        if not a_out > a_in:
            msg = "'a_out' must be greater than 'a_in'"
            raise ValueError(msg)

        self.positions = positions
        self.a_in = a_in
        self.a_out = a_out
        self.b_out = b_out

        if b_in is None:
            b_in = self.b_out * self.a_in / self.a_out
        elif not b_out > b_in:
            msg = "'b_out' must be greater than 'b_in'"
            raise ValueError(msg)
        self.b_in = b_in

        self.theta = theta

    def to_pixel(self, wcs):
        """
        Convert the aperture to an `EllipticalAnnulus` object defined in
        pixel coordinates.

        Parameters
        ----------
        wcs : WCS object
            A world coordinate system (WCS) transformation that
            supports the `astropy shared interface for WCS
            <https://docs.astropy.org/en/stable/wcs/wcsapi.html>`_
            (e.g., `astropy.wcs.WCS`, `gwcs.wcs.WCS`).

        Returns
        -------
        aperture : `EllipticalAnnulus` object
            An `EllipticalAnnulus` object.

        Notes
        -----
        The aperture shape parameters are converted using the local WCS
        properties (pixel scale, rotation angle) evaluated at the first
        aperture position. Because aperture objects require scalar shape
        parameters, only a single reference position is used for the
        conversion. For apertures with multiple positions used with a
        WCS that has spatially-varying distortions, this may produce
        inaccurate results for positions far from the first position.
        """
        xpos, ypos = wcs.world_to_pixel(self.positions)
        positions = np.transpose((xpos, ypos))

        skypos = self.positions if self.isscalar else self.positions[0]
        sky_angle_rad = self.theta.to(u.rad).value

        _, pix_w_out, pix_h_out, pix_angle = sky_ellipse_to_pixel_svd(
            skypos, wcs,
            2 * self.a_out.to(u.arcsec).value,
            2 * self.b_out.to(u.arcsec).value,
            sky_angle_rad)
        _, pix_w_in, pix_h_in, _ = sky_ellipse_to_pixel_svd(
            skypos, wcs,
            2 * self.a_in.to(u.arcsec).value,
            2 * self.b_in.to(u.arcsec).value,
            sky_angle_rad)

        a_out = pix_w_out / 2
        b_out = pix_h_out / 2
        a_in = pix_w_in / 2
        b_in = pix_h_in / 2
        return EllipticalAnnulus(positions=positions, a_in=a_in,
                                 a_out=a_out, b_out=b_out,
                                 b_in=b_in, theta=pix_angle)
