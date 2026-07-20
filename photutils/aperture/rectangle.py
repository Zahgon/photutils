
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
from photutils.geometry import rectangular_overlap_grid
from photutils.utils._deprecation import (deprecated,
                                          deprecated_positional_kwargs)
from photutils.utils._wcs_helpers import (pixel_to_sky_scales,
                                          sky_to_pixel_scales)

__all__ = [
    'RectangularAnnulus',
    'RectangularAperture',
    'RectangularMaskMixin',
    'SkyRectangularAnnulus',
    'SkyRectangularAperture',
]


@deprecated('3.0', until='4.0')
class RectangularMaskMixin:  # pragma: no cover

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
        _, subpixels = self._translate_mask_method(method, subpixels,
                                                   rectangle=True)

        if hasattr(self, 'w'):
            w = self.w
            h = self.h
        elif hasattr(self, 'w_out'):  # annulus
            w = self.w_out
            h = self.h_out
        else:
            msg = 'Cannot determine the aperture radius'
            raise ValueError(msg)

        masks = []
        for bbox, edges in zip(self._bbox, self._centered_edges, strict=True):
            ny, nx = bbox.shape
            theta_rad = self.theta.to(u.radian).value
            mask = rectangular_overlap_grid(edges[0], edges[1], edges[2],
                                            edges[3], nx, ny, w, h,
                                            theta_rad, 0, subpixels)

            if hasattr(self, 'w_in'):
                mask -= rectangular_overlap_grid(edges[0], edges[1], edges[2],
                                                 edges[3], nx, ny, self.w_in,
                                                 self.h_in, theta_rad,
                                                 0, subpixels)

            masks.append(ApertureMask(mask, bbox))

        if self.isscalar:
            return masks[0]

        return masks

    @staticmethod
    def _calc_extents(width, height, theta):
        pass

    @staticmethod
    def _lower_left_positions(positions, width, height, theta):
        pass


def _calc_rectangle_extents(width, height, theta):
    pass


def _calc_lower_left_positions(positions, width, height, theta):
    pass


class RectangularAperture(PixelAperture):

    _params = ('positions', 'w', 'h', 'theta')
    positions = PixelPositions('The center pixel position(s).')
    w = PositiveScalar('The full width in pixels.')
    h = PositiveScalar('The full height in pixels.')
    theta = ScalarAngleOrValue('The counterclockwise rotation angle as an '
                               'angular Quantity or a value in radians from '
                               'the positive x axis.')
    _is_rectangle = True  # remove when rectangles support "exact" method

    @deprecated_positional_kwargs(since='3.0', until='4.0')
    def __init__(self, positions, w, h, theta=0.0):
        self.positions = positions
        self.w = w
        self.h = h
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
        return rectangular_overlap_grid(edges[0], edges[1], edges[2],
                                        edges[3], nx, ny, self.w,
                                        self.h, theta_rad,
                                        use_exact, subpixels)

    def to_sky(self, wcs):
        pass


class RectangularAnnulus(PixelAperture):

    _params = ('positions', 'w_in', 'w_out', 'h_in', 'h_out', 'theta')
    positions = PixelPositions('The center pixel position(s).')
    w_in = PositiveScalar('The inner full width in pixels.')
    w_out = PositiveScalar('The outer full width in pixels.')
    h_in = PositiveScalar('The inner full height in pixels.')
    h_out = PositiveScalar('The outer full height in pixels.')
    theta = ScalarAngleOrValue('The counterclockwise rotation angle as an '
                               'angular Quantity or a value in radians from '
                               'the positive x axis.')
    _is_rectangle = True  # remove when rectangles support "exact" method

    @deprecated_positional_kwargs(since='3.0', until='4.0')
    def __init__(self, positions, w_in, w_out, h_out, h_in=None, theta=0.0):
        if not w_out > w_in:
            msg = "'w_out' must be greater than 'w_in'"
            raise ValueError(msg)

        self.positions = positions
        self.w_in = w_in
        self.w_out = w_out
        self.h_out = h_out

        if h_in is None:
            h_in = self.w_in * self.h_out / self.w_out
        elif not h_out > h_in:
            msg = "'h_out' must be greater than 'h_in'"
            raise ValueError(msg)
        self.h_in = h_in

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
        overlap = rectangular_overlap_grid(edges[0], edges[1], edges[2],
                                           edges[3], nx, ny, self.w_out,
                                           self.h_out, theta_rad,
                                           use_exact, subpixels)
        overlap -= rectangular_overlap_grid(edges[0], edges[1], edges[2],
                                            edges[3], nx, ny, self.w_in,
                                            self.h_in, theta_rad,
                                            use_exact, subpixels)
        return overlap

    def to_sky(self, wcs):
        pass


class SkyRectangularAperture(SkyAperture):

    _params = ('positions', 'w', 'h', 'theta')
    positions = SkyCoordPositions('The center position(s) in sky coordinates.')
    w = PositiveScalarAngle('The full width in angular units.')
    h = PositiveScalarAngle('The full height in angular units.')
    theta = ScalarAngle('The position angle (in angular units) of the '
                        'rectangle "width" side.')

    @deprecated_positional_kwargs(since='3.0', until='4.0')
    def __init__(self, positions, w, h, theta=0.0 * u.deg):
        self.positions = positions
        self.w = w
        self.h = h
        self.theta = theta

    def to_pixel(self, wcs):
        """
        Convert the aperture to a `RectangularAperture` object defined
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
        aperture : `RectangularAperture` object
            A `RectangularAperture` object.

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
        _, scale_w, scale_h, pixel_angle = sky_to_pixel_scales(
            skypos, wcs, sky_angle_rad)

        w = self.w.to(u.arcsec).value * scale_w
        h = self.h.to(u.arcsec).value * scale_h
        return RectangularAperture(positions=positions, w=w, h=h,
                                   theta=pixel_angle)


class SkyRectangularAnnulus(SkyAperture):

    _params = ('positions', 'w_in', 'w_out', 'h_in', 'h_out', 'theta')
    positions = SkyCoordPositions('The center position(s) in sky coordinates.')
    w_in = PositiveScalarAngle('The inner full width in angular units.')
    w_out = PositiveScalarAngle('The outer full width in angular units.')
    h_in = PositiveScalarAngle('The inner full height in angular units.')
    h_out = PositiveScalarAngle('The outer full height in angular units.')
    theta = ScalarAngle('The position angle (in angular units) of the '
                        'rectangle "width" side.')

    @deprecated_positional_kwargs(since='3.0', until='4.0')
    def __init__(self, positions, w_in, w_out, h_out, h_in=None,
                 theta=0.0 * u.deg):
        if not w_out > w_in:
            msg = "'w_out' must be greater than 'w_in'"
            raise ValueError(msg)

        self.positions = positions
        self.w_in = w_in
        self.w_out = w_out
        self.h_out = h_out

        if h_in is None:
            h_in = self.w_in * self.h_out / self.w_out
        elif not h_out > h_in:
            msg = "'h_out' must be greater than 'h_in'"
            raise ValueError(msg)
        self.h_in = h_in

        self.theta = theta

    def to_pixel(self, wcs):
        """
        Convert the aperture to a `RectangularAnnulus` object defined in
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
        aperture : `RectangularAnnulus` object
            A `RectangularAnnulus` object.

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
        _, scale_w, scale_h, pixel_angle = sky_to_pixel_scales(
            skypos, wcs, sky_angle_rad)

        w_in = self.w_in.to(u.arcsec).value * scale_w
        w_out = self.w_out.to(u.arcsec).value * scale_w
        h_in = self.h_in.to(u.arcsec).value * scale_h
        h_out = self.h_out.to(u.arcsec).value * scale_h
        return RectangularAnnulus(positions=positions, w_in=w_in,
                                  w_out=w_out, h_out=h_out,
                                  h_in=h_in, theta=pixel_angle)
