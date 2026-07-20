
import copy

import numpy as np

from photutils.isophote.geometry import EllipseGeometry
from photutils.isophote.integrator import INTEGRATORS
from photutils.utils._deprecation import (deprecated_getattr,
                                          deprecated_positional_kwargs,
                                          deprecated_renamed_argument)

_DEPRECATED_SAMPLE_ATTRIBUTES = {
    'gradient_error': 'gradient_err',
    'gradient_relative_error': 'gradient_rel_err',
    'nclip': 'n_clip',
}

__all__ = ['EllipseSample']


class EllipseSample:

    @deprecated_positional_kwargs(since='3.0', until='4.0')
    @deprecated_renamed_argument('nclip', 'n_clip', '3.0', until='4.0')
    def __init__(self, image, sma, x0=None, y0=None, astep=0.1, eps=0.2,
                 position_angle=0.0, sclip=3.0, n_clip=0,
                 linear_growth=False,
                 integrmode='bilinear', geometry=None):
        self.image = image
        self.integrmode = integrmode

        if geometry:
            self.geometry = copy.deepcopy(geometry)
            self.geometry.sma = sma
        else:
            _x0 = x0
            _y0 = y0
            if not _x0 or not _y0:
                _x0 = image.shape[1] / 2
                _y0 = image.shape[0] / 2

            self.geometry = EllipseGeometry(_x0, _y0, sma, eps,
                                            position_angle, astep=astep,
                                            linear_growth=linear_growth)

        self.sclip = sclip
        self.n_clip = n_clip

        self.values = None
        self.mean = None
        self.gradient = None
        self.gradient_err = None
        self.gradient_rel_err = None
        self.sector_area = None

        self.total_points = 0
        self.actual_points = 0

    def __getattr__(self, name):
        return deprecated_getattr(self, name,
                                  _DEPRECATED_SAMPLE_ATTRIBUTES,
                                  since='3.0', until='4.0')

    def extract(self):
        """
        Extract sample data by scanning an elliptical path over the
        image array.

        Returns
        -------
        result : 2D `~numpy.ndarray`
            The rows of the array contain the angles, radii, and
            extracted intensity values, respectively.
        """
        if self.values is not None:
            return self.values

        s = self._extract()
        self.values = s
        return s

    def _extract(self, *, phi_min=0.05):

        angles = []
        radii = []
        intensities = []
        sector_areas = []

        self.total_points = 0
        self.actual_points = 0

        integrator = INTEGRATORS[self.integrmode](self.image, self.geometry,
                                                  angles, radii, intensities)

        radius = self.geometry.initial_polar_radius
        phi = self.geometry.initial_polar_angle

        if integrator.is_area():
            integrator.integrate(radius, phi)
            area = integrator.get_sector_area()
            angles = []
            radii = []
            intensities = []
            if area < 1.0:
                integrator = INTEGRATORS['bilinear'](
                    self.image, self.geometry, angles, radii, intensities)
            else:
                integrator = INTEGRATORS[self.integrmode](self.image,
                                                          self.geometry,
                                                          angles, radii,
                                                          intensities)

        while phi <= np.pi * 2.0 + phi_min:
            integrator.integrate(radius, phi)

            sector_areas.append(integrator.get_sector_area())

            self.total_points += 1

            phistep_ = integrator.get_polar_angle_step()
            phi += min(phistep_, 0.5)
            radius = self.geometry.radius(phi)

        self.sector_area = np.mean(np.array(sector_areas))

        angles, radii, intensities = self._sigma_clip(angles, radii,
                                                      intensities)

        self.actual_points = len(angles)

        return np.array([np.array(angles), np.array(radii),
                         np.array(intensities)])

    def _sigma_clip(self, angles, radii, intensities):
        if self.n_clip > 0:
            for _ in range(self.n_clip):
                angles, radii, intensities = self._iter_sigma_clip(
                    angles[:], radii[:], intensities[:])

        return np.array(angles), np.array(radii), np.array(intensities)

    def _iter_sigma_clip(self, angles, radii, intensities):
        r_angles = []
        r_radii = []
        r_intensities = []

        values = np.array(intensities)
        mean = np.mean(values)
        sig = np.std(values)
        lower = mean - self.sclip * sig
        upper = mean + self.sclip * sig

        count = 0
        for k, intensity in enumerate(intensities):
            if lower <= intensity < upper:
                r_angles.append(angles[k])
                r_radii.append(radii[k])
                r_intensities.append(intensity)
                count += 1

        return r_angles, r_radii, r_intensities

    @deprecated_positional_kwargs(since='3.0', until='4.0')
    def update(self, fixed_parameters=None):
        """
        Update this `~photutils.isophote.EllipseSample` instance.

        This method calls the
        :meth:`~photutils.isophote.EllipseSample.extract` method to get
        the values that match the current ``geometry`` attribute, and
        then computes the mean intensity, local gradient, and other
        associated quantities.

        Parameters
        ----------
        fixed_parameters : `None` or array_like, optional
            An array of the fixed parameters. Must have 4 elements,
            corresponding to x center, y center, PA, and EPS.
        """
        if fixed_parameters is None:
            fixed_parameters = np.array([False, False, False, False])
        self.geometry.fix = fixed_parameters

        step = self.geometry.astep

        s = self.extract()
        self.mean = np.mean(s[2])

        gradient, gradient_err = self._get_gradient(step)

        previous_gradient = self.gradient
        if not previous_gradient:
            previous_gradient = gradient + gradient_err

        if gradient >= (previous_gradient / 3.0):  # gradient is negative!
            gradient, gradient_err = self._get_gradient(2 * step)

        if gradient >= (previous_gradient / 3.0):
            gradient = previous_gradient * 0.8
            gradient_err = None

        self.gradient = gradient
        self.gradient_err = gradient_err
        if gradient_err and gradient < 0.0:
            self.gradient_rel_err = gradient_err / np.abs(gradient)
        else:
            self.gradient_rel_err = None

    def _get_gradient(self, step):
        gradient_sma = (1.0 + step) * self.geometry.sma

        gradient_sample = EllipseSample(
            self.image, gradient_sma, x0=self.geometry.x0,
            y0=self.geometry.y0, astep=self.geometry.astep, sclip=self.sclip,
            n_clip=self.n_clip, eps=self.geometry.eps,
            position_angle=self.geometry.pa,
            linear_growth=self.geometry.linear_growth,
            integrmode=self.integrmode)

        sg = gradient_sample.extract()
        mean_g = np.mean(sg[2])
        gradient = (mean_g - self.mean) / self.geometry.sma / step

        s = self.extract()
        sigma = np.std(s[2])
        sigma_g = np.std(sg[2])

        gradient_err = (np.sqrt(sigma**2 / len(s[2])
                                + sigma_g**2 / len(sg[2]))
                        / self.geometry.sma / step)

        return gradient, gradient_err

    def coordinates(self):
        pass


class CentralEllipseSample(EllipseSample):

    @deprecated_positional_kwargs(since='3.0', until='4.0')
    def update(self, fixed_parameters=None):  # noqa: ARG002
        """
        Update this `~photutils.isophote.EllipseSample` instance with
        the intensity integrated at the (x0, y0) center position using
        bilinear integration. The local gradient is set to `None`.

        Parameters
        ----------
        fixed_parameters : `None` or array_like, optional
            An array of the fixed parameters. Must have 4 elements,
            corresponding to x center, y center, PA, and EPS. This
            keyword is ignored in this subclass.
        """
        s = self.extract()
        self.mean = s[2][0]

        self.gradient = None
        self.gradient_err = None
        self.gradient_rel_err = None

    def _extract(self):
        angles = []
        radii = []
        intensities = []

        integrator = INTEGRATORS['bilinear'](self.image, self.geometry,
                                             angles, radii, intensities)
        integrator.integrate(0.0, 0.0)

        self.total_points = 1
        self.actual_points = 1

        return np.array([np.array(angles), np.array(radii),
                         np.array(intensities)])
