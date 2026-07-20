
import math

import numpy as np
from astropy import log

from photutils.utils._deprecation import deprecated_positional_kwargs

__all__ = ['EllipseGeometry']


IN_MASK = [
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
]

OUT_MASK = [
    [1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1],
    [1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1],
    [1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1],
    [1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1],
    [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
    [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
    [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
    [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
    [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
    [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
    [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
    [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
    [1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1],
    [1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1],
    [1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1],
    [1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1],
]


def _area(sma, eps, phi, r):
    """
    Compute elliptical sector area.
    """
    aux = r * math.cos(phi) / sma
    signal = aux / abs(aux)
    if abs(aux) >= 1.0:
        aux = signal
    return abs(sma**2 * (1.0 - eps) / 2.0 * math.acos(aux))


class EllipseGeometry:

    @deprecated_positional_kwargs(since='3.0', until='4.0')
    def __init__(self, x0, y0, sma, eps, pa, astep=0.1, linear_growth=False,
                 fix_center=False, fix_pa=False, fix_eps=False):
        self.x0 = x0
        self.y0 = y0
        self.sma = sma
        self.eps = eps
        self.pa = pa

        self.astep = astep
        self.linear_growth = linear_growth

        self.fix = np.array([fix_center, fix_center, fix_pa, fix_eps])

        self._phi_min = 0.05
        self._phi_max = 0.2

        sma1, sma2 = self.bounding_ellipses()
        inner_sma = min((sma2 - sma1), 3.0)
        self._area_factor = (sma2 - sma1) * inner_sma

        if self.sma > 0.0:
            self.sector_angular_width = max(min((inner_sma / self.sma),
                                                self._phi_max), self._phi_min)
            self.initial_polar_angle = self.sector_angular_width / 2.0
            self.initial_polar_radius = self.radius(self.initial_polar_angle)

    @deprecated_positional_kwargs(since='3.0', until='4.0')
    def find_center(self, image, threshold=0.1, verbose=True):
        pass

    def radius(self, angle):
        """
        Calculate the polar radius for a given polar angle.

        Parameters
        ----------
        angle : float
            The polar angle (radians).

        Returns
        -------
        radius : float
            The polar radius (pixels).
        """
        return (self.sma * (1.0 - self.eps)
                / np.sqrt(((1.0 - self.eps) * np.cos(angle))**2
                          + (np.sin(angle))**2))

    def initialize_sector_geometry(self, phi):
        """
        Initialize geometry attributes associated with an elliptical
        sector at the given polar angle ``phi``.

        This function computes:

        * the four vertices that define the elliptical sector on the
          pixel array.
        * the sector area (saved in the ``sector_area`` attribute)
        * the sector angular width (saved in ``sector_angular_width``
          attribute)

        Parameters
        ----------
        phi : float
            The polar angle (radians) where the sector is located.

        Returns
        -------
        x, y : 1D `~numpy.ndarray`
            The x and y coordinates of each vertex as 1D arrays.
        """
        sma1, sma2 = self.bounding_ellipses()
        eps_ = 1.0 - self.eps

        self._phi1 = phi - self.sector_angular_width / 2.0
        r1 = (sma1 * eps_ / math.sqrt((eps_ * math.cos(self._phi1))**2
                                      + (math.sin(self._phi1))**2))
        r2 = (sma2 * eps_ / math.sqrt((eps_ * math.cos(self._phi1))**2
                                      + (math.sin(self._phi1))**2))

        self._phi2 = phi + self.sector_angular_width / 2.0
        r3 = (sma2 * eps_ / math.sqrt((eps_ * math.cos(self._phi2))**2
                                      + (math.sin(self._phi2))**2))

        r4 = (sma1 * eps_ / math.sqrt((eps_ * math.cos(self._phi2))**2
                                      + (math.sin(self._phi2))**2))

        sa1 = _area(sma1, self.eps, self._phi1, r1)
        sa2 = _area(sma2, self.eps, self._phi1, r2)
        sa3 = _area(sma2, self.eps, self._phi2, r3)
        sa4 = _area(sma1, self.eps, self._phi2, r4)
        self.sector_area = abs((sa3 - sa2) - (sa4 - sa1))

        self.sector_angular_width = max(min((self._area_factor / (r3 - r4)
                                             / r4), self._phi_max),
                                        self._phi_min)

        vertex_x = np.zeros(shape=4, dtype=float)
        vertex_y = np.zeros(shape=4, dtype=float)

        vertex_x[0:2] = np.array([r1, r2]) * math.cos(self._phi1 + self.pa)
        vertex_x[2:4] = np.array([r4, r3]) * math.cos(self._phi2 + self.pa)
        vertex_y[0:2] = np.array([r1, r2]) * math.sin(self._phi1 + self.pa)
        vertex_y[2:4] = np.array([r4, r3]) * math.sin(self._phi2 + self.pa)
        vertex_x += self.x0
        vertex_y += self.y0

        return vertex_x, vertex_y

    def bounding_ellipses(self):
        """
        Compute the semimajor axis of the two ellipses that bound the
        annulus where integrations take place.

        Returns
        -------
        sma1, sma2 : float
            The smaller and larger values of semimajor axis length that
            define the annulus bounding ellipses.
        """
        if self.linear_growth:
            a1 = self.sma - self.astep / 2.0
            a2 = self.sma + self.astep / 2.0
        else:
            a1 = self.sma * (1.0 - self.astep / 2.0)
            a2 = self.sma * (1.0 + self.astep / 2.0)

        return a1, a2

    def polar_angle_sector_limits(self):
        """
        Return the two polar angles that bound the sector.

        The two bounding polar angles become available only after
        calling the
        :meth:`~photutils.isophote.EllipseGeometry.initialize_sector_geometry`
        method.

        Returns
        -------
        phi1, phi2 : float
            The smaller and larger values of polar angle that bound the
            current sector.
        """
        return self._phi1, self._phi2

    def to_polar(self, x, y):
        r"""
        Return the radius and polar angle in the ellipse coordinate
        system given (x, y) pixel image coordinates.

        This function takes care of the different definitions for
        position angle (PA) and polar angle (phi):

        .. math::

            -\pi < PA < \pi

            0 < phi < 2 \pi

        Note that radius can be anything. The solution is not tied to
        the semimajor axis length, but to the center position and tilt
        angle.

        Parameters
        ----------
        x, y : float
            The (x, y) image coordinates.

        Returns
        -------
        radius, angle : float
            The ellipse radius and polar angle.
        """

        if isinstance(x, (int, float)):
            return self._to_polar_scalar(x, y)

        return self._to_polar_vectorized(x, y)

    def _to_polar_scalar(self, x, y):
        x1 = x - self.x0
        y1 = y - self.y0

        radius = x1**2 + y1**2
        if radius > 0.0:
            radius = math.sqrt(radius)
            angle = math.asin(abs(y1) / radius)
        else:
            radius = 0.0
            angle = 1.0

        if x1 >= 0.0 and y1 < 0.0:
            angle = 2 * np.pi - angle
        elif x1 < 0.0 and y1 >= 0.0:
            angle = np.pi - angle
        elif x1 < 0.0 and y1 < 0.0:
            angle = np.pi + angle

        pa1 = self.pa
        if self.pa < 0.0:
            pa1 = self.pa + 2 * np.pi
        angle = angle - pa1
        if angle < 0.0:
            angle = angle + 2 * np.pi

        return radius, angle

    def _to_polar_vectorized(self, x, y):
        x1 = np.atleast_2d(x) - self.x0
        y1 = np.atleast_2d(y) - self.y0

        radius = x1**2 + y1**2
        angle = np.ones(radius.shape)

        imask = (radius > 0.0)
        radius[imask] = np.sqrt(radius[imask])
        angle[imask] = np.arcsin(np.abs(y1[imask]) / radius[imask])
        radius[~imask] = 0.0
        angle[~imask] = 1.0

        idx = (x1 >= 0.0) & (y1 < 0.0)
        angle[idx] = 2 * np.pi - angle[idx]
        idx = (x1 < 0.0) & (y1 >= 0.0)
        angle[idx] = np.pi - angle[idx]
        idx = (x1 < 0.0) & (y1 < 0.0)
        angle[idx] = np.pi + angle[idx]

        pa1 = self.pa
        if self.pa < 0.0:
            pa1 = self.pa + 2 * np.pi
        angle = angle - pa1
        angle[angle < 0] += 2 * np.pi

        return radius, angle

    def update_sma(self, step):
        pass

    def reset_sma(self, step):
        pass
