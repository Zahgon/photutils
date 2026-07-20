
import astropy.units as u
import numpy as np
from astropy.modeling import Fittable2DModel, Parameter
from astropy.modeling.utils import ellipse_extent
from astropy.units import UnitsError
from scipy.special import erf, j1, jn_zeros

__all__ = [
    'AiryDiskPSF',
    'CircularGaussianPRF',
    'CircularGaussianPSF',
    'CircularGaussianSigmaPRF',
    'GaussianPRF',
    'GaussianPSF',
    'MoffatPSF',
]

FLOAT_EPSILON = float(np.finfo(np.float32).tiny)
GAUSSIAN_FWHM_TO_SIGMA = 1.0 / (2.0 * np.sqrt(2.0 * np.log(2.0)))


def _gaussian_amplitude(flux, xsigma, ysigma):
    pass


class GaussianPSF(Fittable2DModel):

    flux = Parameter(
        default=1, description='Total integrated flux over the entire PSF.')
    x_0 = Parameter(
        default=0, description='Position of the peak along the x axis')
    y_0 = Parameter(
        default=0, description='Position of the peak along the y axis')
    x_fwhm = Parameter(
        default=1,
        bounds=(FLOAT_EPSILON, None),
        fixed=True,
        description='FWHM of the Gaussian along the x axis')
    y_fwhm = Parameter(
        default=1,
        bounds=(FLOAT_EPSILON, None),
        fixed=True,
        description='FWHM of the Gaussian along the y axis')
    theta = Parameter(
        default=0.0, description=('CCW rotation angle either as a float (in '
                                  'degrees) or a Quantity angle (optional)'),
        fixed=True)

    def __init__(self, *, flux=flux.default, x_0=x_0.default, y_0=y_0.default,
                 x_fwhm=x_fwhm.default, y_fwhm=y_fwhm.default,
                 theta=theta.default, bbox_factor=5.5, **kwargs):
        super().__init__(flux=flux, x_0=x_0, y_0=y_0, x_fwhm=x_fwhm,
                         y_fwhm=y_fwhm, theta=theta, **kwargs)
        self.bbox_factor = bbox_factor

    @property
    def amplitude(self):
        pass

    @property
    def x_sigma(self):
        pass

    @property
    def y_sigma(self):
        pass

    def _calc_bounding_box(self, *, factor=5.5):
        """
        Calculate a bounding box defining the limits of the model.

        The limits are adjusted for rotation.

        Parameters
        ----------
        factor : float, optional
            The multiple of the x and y standard deviations (sigma) used
            to define the limits.

        Returns
        -------
        bbox : tuple
            A bounding box defining the ((y_min, y_max), (x_min, x_max))
            limits of the model.
        """
        a = factor * self.x_sigma
        b = factor * self.y_sigma
        dx, dy = ellipse_extent(a, b, self.theta)
        return ((self.y_0 - dy, self.y_0 + dy), (self.x_0 - dx, self.x_0 + dx))

    @property
    def bounding_box(self):
        """
        The bounding box of the model.

        Examples
        --------
        >>> from photutils.psf import GaussianPSF
        >>> model = GaussianPSF(x_0=0, y_0=0, x_fwhm=2, y_fwhm=3)
        >>> model.bounding_box  # doctest: +FLOAT_CMP
        ModelBoundingBox(
            intervals={
                x: Interval(lower=-4.671269901584105, upper=4.671269901584105)
                y: Interval(lower=-7.006904852376157, upper=7.006904852376157)
            }
            model=GaussianPSF(inputs=('x', 'y'))
            order='C'
        )
        >>> model.bbox_factor = 7
        >>> model.bounding_box  # doctest: +FLOAT_CMP
        ModelBoundingBox(
            intervals={
                x: Interval(lower=-5.945252602016134, upper=5.945252602016134)
                y: Interval(lower=-8.9178789030242, upper=8.9178789030242)
            }
            model=GaussianPSF(inputs=('x', 'y'))
            order='C'
        )
        """
        return self._calc_bounding_box(factor=self.bbox_factor)

    def evaluate(self, x, y, flux, x_0, y_0, x_fwhm, y_fwhm, theta):
        """
        Calculate the value of the 2D Gaussian model at the input
        coordinates for the given model parameters.

        Parameters
        ----------
        x, y : float or array_like
            The x and y coordinates at which to evaluate the model.

        flux : float
            Total integrated flux over the entire PSF.

        x_0, y_0 : float
            Position of the peak along the x and y axes.

        x_fwhm, y_fwhm : float
            FWHM of the Gaussian along the x and y axes.

        theta : float
            The counterclockwise rotation angle either as a float (in
            degrees) or a `~astropy.units.Quantity` angle (optional).

        Returns
        -------
        result : `~numpy.ndarray`
            The value of the model evaluated at the input coordinates.
        """
        if not isinstance(theta, u.Quantity):
            theta = np.deg2rad(theta)
        cost2 = np.cos(theta) ** 2
        sint2 = np.sin(theta) ** 2
        sin2t = np.sin(2.0 * theta)
        xstd = x_fwhm * GAUSSIAN_FWHM_TO_SIGMA
        ystd = y_fwhm * GAUSSIAN_FWHM_TO_SIGMA
        xstd2 = xstd ** 2
        ystd2 = ystd ** 2
        xdiff = x - x_0
        ydiff = y - y_0
        a = 0.5 * ((cost2 / xstd2) + (sint2 / ystd2))
        b = 0.5 * ((sin2t / xstd2) - (sin2t / ystd2))
        c = 0.5 * ((sint2 / xstd2) + (cost2 / ystd2))

        if isinstance(xstd, u.Quantity):
            xstd = xstd.value
            ystd = ystd.value

        amplitude = flux / (2 * np.pi * xstd * ystd)
        return amplitude * np.exp(
            -(a * xdiff**2) - (b * xdiff * ydiff) - (c * ydiff**2))

    @staticmethod
    def fit_deriv(x, y, flux, x_0, y_0, x_fwhm, y_fwhm, theta):
        pass

    @property
    def input_units(self):
        pass

    def _parameter_units_for_data_units(self, inputs_unit, outputs_unit):
        pass


class CircularGaussianPSF(Fittable2DModel):

    flux = Parameter(
        default=1, description='Total integrated flux over the entire PSF.')
    x_0 = Parameter(
        default=0, description='Position of the peak along the x axis')
    y_0 = Parameter(
        default=0, description='Position of the peak along the y axis')
    fwhm = Parameter(
        default=1,
        bounds=(FLOAT_EPSILON, None),
        fixed=True,
        description='FWHM of the Gaussian')

    def __init__(self, *, flux=flux.default, x_0=x_0.default, y_0=y_0.default,
                 fwhm=fwhm.default, bbox_factor=5.5, **kwargs):
        super().__init__(flux=flux, x_0=x_0, y_0=y_0, fwhm=fwhm, **kwargs)
        self.bbox_factor = bbox_factor

    @property
    def amplitude(self):
        pass

    @property
    def sigma(self):
        pass

    def _calc_bounding_box(self, *, factor=5.5):
        """
        Calculate a bounding box defining the limits of the model.

        Parameters
        ----------
        factor : float, optional
            The multiple of the standard deviations (sigma) used to
            define the limits.

        Returns
        -------
        bbox : tuple
            A bounding box defining the ((y_min, y_max), (x_min, x_max))
            limits of the model.
        """
        delta = factor * self.sigma
        return ((self.y_0 - delta, self.y_0 + delta),
                (self.x_0 - delta, self.x_0 + delta))

    @property
    def bounding_box(self):
        """
        The bounding box of the model.

        Examples
        --------
        >>> from photutils.psf import CircularGaussianPSF
        >>> model = CircularGaussianPSF(x_0=0, y_0=0, fwhm=2)
        >>> model.bounding_box  # doctest: +FLOAT_CMP
        ModelBoundingBox(
            intervals={
                x: Interval(lower=-4.671269901584105, upper=4.671269901584105)
                y: Interval(lower=-4.671269901584105, upper=4.671269901584105)
            }
            model=CircularGaussianPSF(inputs=('x', 'y'))
            order='C'
        )
        >>> model.bbox_factor = 7
        >>> model.bounding_box  # doctest: +FLOAT_CMP
        ModelBoundingBox(
            intervals={
                x: Interval(lower=-5.945252602016134, upper=5.945252602016134)
                y: Interval(lower=-5.945252602016134, upper=5.945252602016134)
            }
            model=CircularGaussianPSF(inputs=('x', 'y'))
            order='C'
        )
        """
        return self._calc_bounding_box(factor=self.bbox_factor)

    def evaluate(self, x, y, flux, x_0, y_0, fwhm):
        """
        Calculate the value of the 2D Gaussian model at the input
        coordinates for the given model parameters.

        Parameters
        ----------
        x, y : float or array_like
            The x and y coordinates at which to evaluate the model.

        flux : float
            Total integrated flux over the entire PSF.

        x_0, y_0 : float
            Position of the peak along the x and y axes.

        fwhm : float
            FWHM of the Gaussian.

        Returns
        -------
        result : `~numpy.ndarray`
            The value of the model evaluated at the input coordinates.
        """
        sigma2 = (fwhm * GAUSSIAN_FWHM_TO_SIGMA) ** 2

        sigma2_norm = sigma2
        if isinstance(sigma2, u.Quantity):
            sigma2_norm = sigma2.value

        amplitude = flux / (2 * np.pi * sigma2_norm)
        return amplitude * np.exp(-0.5 * ((x - x_0) ** 2 + (y - y_0) ** 2)
                                  / sigma2)

    @staticmethod
    def fit_deriv(x, y, flux, x_0, y_0, fwhm):
        pass

    @property
    def input_units(self):
        pass

    def _parameter_units_for_data_units(self, inputs_unit, outputs_unit):
        pass


class GaussianPRF(Fittable2DModel):

    flux = Parameter(
        default=1, description='Total integrated flux over the entire PSF.')
    x_0 = Parameter(
        default=0, description='Position of the peak along the x axis')
    y_0 = Parameter(
        default=0, description='Position of the peak along the y axis')
    x_fwhm = Parameter(
        default=1,
        bounds=(FLOAT_EPSILON, None),
        fixed=True,
        description='FWHM of the Gaussian along the x axis')
    y_fwhm = Parameter(
        default=1,
        bounds=(FLOAT_EPSILON, None),
        fixed=True,
        description='FWHM of the Gaussian along the y axis')
    theta = Parameter(
        default=0.0, description=('CCW rotation angle either as a float (in '
                                  'degrees) or a Quantity angle (optional)'),
        fixed=True)

    def __init__(self, *, flux=flux.default, x_0=x_0.default, y_0=y_0.default,
                 x_fwhm=x_fwhm.default, y_fwhm=y_fwhm.default,
                 theta=theta.default, bbox_factor=5.5, **kwargs):
        super().__init__(flux=flux, x_0=x_0, y_0=y_0, x_fwhm=x_fwhm,
                         y_fwhm=y_fwhm, theta=theta, **kwargs)
        self.bbox_factor = bbox_factor

    @property
    def amplitude(self):
        pass

    @property
    def x_sigma(self):
        pass

    @property
    def y_sigma(self):
        pass

    def _calc_bounding_box(self, *, factor=5.5):
        """
        Calculate a bounding box defining the limits of the model.

        The limits are adjusted for rotation.

        Parameters
        ----------
        factor : float, optional
            The multiple of the x and y FWHMs used to define the limits.
            zzzz

        Returns
        -------
        bbox : tuple
            A bounding box defining the ((y_min, y_max), (x_min, x_max))
            limits of the model.
        """
        a = factor * self.x_sigma
        b = factor * self.y_sigma
        dx, dy = ellipse_extent(a, b, self.theta)
        return ((self.y_0 - dy, self.y_0 + dy), (self.x_0 - dx, self.x_0 + dx))

    @property
    def bounding_box(self):
        """
        The bounding box of the model.

        Examples
        --------
        >>> from photutils.psf import GaussianPRF
        >>> model = GaussianPRF(x_0=0, y_0=0, x_fwhm=2, y_fwhm=3)
        >>> model.bounding_box  # doctest: +FLOAT_CMP
        ModelBoundingBox(
            intervals={
                x: Interval(lower=-4.671269901584105, upper=4.671269901584105)
                y: Interval(lower=-7.006904852376157, upper=7.006904852376157)
            }
            model=GaussianPRF(inputs=('x', 'y'))
            order='C'
        )
        >>> model.bbox_factor = 7
        >>> model.bounding_box  # doctest: +FLOAT_CMP
        ModelBoundingBox(
            intervals={
                x: Interval(lower=-5.945252602016134, upper=5.945252602016134)
                y: Interval(lower=-8.9178789030242, upper=8.9178789030242)
            }
            model=GaussianPRF(inputs=('x', 'y'))
            order='C'
        )
        """
        return self._calc_bounding_box(factor=self.bbox_factor)

    def evaluate(self, x, y, flux, x_0, y_0, x_fwhm, y_fwhm, theta):
        """
        Calculate the value of the 2D Gaussian model at the input
        coordinates for the given model parameters.

        Parameters
        ----------
        x, y : float or array_like
            The x and y coordinates at which to evaluate the model.

        flux : float
            Total integrated flux over the entire PSF.

        x_0, y_0 : float
            Position of the peak along the x and y axes.

        x_fwhm, y_fwhm : float
            FWHM of the Gaussian along the x and y axes.

        theta : float
            The counterclockwise rotation angle either as a float (in
            degrees) or a `~astropy.units.Quantity` angle (optional).

        Returns
        -------
        result : `~numpy.ndarray`
            The value of the model evaluated at the input coordinates.
        """
        if not isinstance(theta, u.Quantity):
            theta = np.deg2rad(theta)

        x_sigma = x_fwhm * GAUSSIAN_FWHM_TO_SIGMA
        y_sigma = y_fwhm * GAUSSIAN_FWHM_TO_SIGMA
        dx = x - x_0
        dy = y - y_0
        cost = np.cos(theta)
        sint = np.sin(theta)
        x0 = dx * cost + dy * sint
        y0 = -dx * sint + dy * cost

        dpix = 0.5
        if isinstance(x0, u.Quantity):
            dpix <<= x0.unit

        return (flux / 4.0
                * ((erf((x0 + dpix) / (np.sqrt(2) * x_sigma))
                    - erf((x0 - dpix) / (np.sqrt(2) * x_sigma)))
                   * (erf((y0 + dpix) / (np.sqrt(2) * y_sigma))
                      - erf((y0 - dpix) / (np.sqrt(2) * y_sigma)))))

    @property
    def input_units(self):
        pass

    def _parameter_units_for_data_units(self, inputs_unit, outputs_unit):
        pass


class CircularGaussianPRF(Fittable2DModel):

    flux = Parameter(
        default=1, description='Total integrated flux over the entire PSF.')
    x_0 = Parameter(
        default=0, description='Position of the peak along the x axis')
    y_0 = Parameter(
        default=0, description='Position of the peak along the y axis')
    fwhm = Parameter(
        default=1,
        bounds=(FLOAT_EPSILON, None),
        fixed=True,
        description='FWHM of the Gaussian')

    def __init__(self, *, flux=flux.default, x_0=x_0.default, y_0=y_0.default,
                 fwhm=fwhm.default, bbox_factor=5.5, **kwargs):
        super().__init__(flux=flux, x_0=x_0, y_0=y_0, fwhm=fwhm, **kwargs)
        self.bbox_factor = bbox_factor

    @property
    def amplitude(self):
        pass

    @property
    def sigma(self):
        pass

    def _calc_bounding_box(self, *, factor=5.5):
        """
        Calculate a bounding box defining the limits of the model.

        Parameters
        ----------
        factor : float, optional
            The multiple of the standard deviations (sigma) used to
            define the limits.

        Returns
        -------
        bbox : tuple
            A bounding box defining the ((y_min, y_max), (x_min, x_max))
            limits of the model.
        """
        delta = factor * self.sigma
        return ((self.y_0 - delta, self.y_0 + delta),
                (self.x_0 - delta, self.x_0 + delta))

    @property
    def bounding_box(self):
        """
        The bounding box of the model.

        Examples
        --------
        >>> from photutils.psf import CircularGaussianPRF
        >>> model = CircularGaussianPRF(x_0=0, y_0=0, fwhm=2)
        >>> model.bounding_box  # doctest: +FLOAT_CMP
        ModelBoundingBox(
            intervals={
                x: Interval(lower=-4.671269901584105, upper=4.671269901584105)
                y: Interval(lower=-4.671269901584105, upper=4.671269901584105)
            }
            model=CircularGaussianPRF(inputs=('x', 'y'))
            order='C'
        )
        >>> model.bbox_factor = 7
        >>> model.bounding_box  # doctest: +FLOAT_CMP
        ModelBoundingBox(
            intervals={
                x: Interval(lower=-5.945252602016134, upper=5.945252602016134)
                y: Interval(lower=-5.945252602016134, upper=5.945252602016134)
            }
            model=CircularGaussianPRF(inputs=('x', 'y'))
            order='C'
        )
        """
        return self._calc_bounding_box(factor=self.bbox_factor)

    def evaluate(self, x, y, flux, x_0, y_0, fwhm):
        """
        Calculate the value of the 2D Gaussian model at the input
        coordinates for the given model parameters.

        Parameters
        ----------
        x, y : float or array_like
            The x and y coordinates at which to evaluate the model.

        flux : float
            Total integrated flux over the entire PSF.

        x_0, y_0 : float
            Position of the peak along the x and y axes.

        fwhm : float
            FWHM of the Gaussian.

        Returns
        -------
        result : `~numpy.ndarray`
            The value of the model evaluated at the input coordinates.
        """
        x0 = x - x_0
        y0 = y - y_0
        sigma = fwhm * GAUSSIAN_FWHM_TO_SIGMA

        dpix = 0.5
        if isinstance(x0, u.Quantity):
            dpix <<= x0.unit

        return (flux / 4.0
                * ((erf((x0 + dpix) / (np.sqrt(2) * sigma))
                    - erf((x0 - dpix) / (np.sqrt(2) * sigma)))
                   * (erf((y0 + dpix) / (np.sqrt(2) * sigma))
                      - erf((y0 - dpix) / (np.sqrt(2) * sigma)))))

    @property
    def input_units(self):
        pass

    def _parameter_units_for_data_units(self, inputs_unit, outputs_unit):
        pass


class CircularGaussianSigmaPRF(Fittable2DModel):

    flux = Parameter(
        default=1, description='Total integrated flux over the entire PSF.')
    x_0 = Parameter(
        default=0, description='Position of the peak along the x axis')
    y_0 = Parameter(
        default=0, description='Position of the peak along the y axis')
    sigma = Parameter(
        default=1,
        bounds=(FLOAT_EPSILON, None),
        fixed=True,
        description='Sigma (standard deviation) of the Gaussian')

    def __init__(self, *, flux=flux.default, x_0=x_0.default, y_0=y_0.default,
                 sigma=sigma.default, bbox_factor=5.5, **kwargs):
        super().__init__(sigma=sigma, x_0=x_0, y_0=y_0, flux=flux, **kwargs)
        self.bbox_factor = bbox_factor

    @property
    def amplitude(self):
        pass

    @property
    def fwhm(self):
        pass

    def _calc_bounding_box(self, *, factor=5.5):
        """
        Calculate a bounding box defining the limits of the model.

        Parameters
        ----------
        factor : float, optional
            The multiple of the standard deviations (sigma) used to
            define the limits.

        Returns
        -------
        bbox : tuple
            A bounding box defining the ((y_min, y_max), (x_min, x_max))
            limits of the model.
        """
        delta = factor * self.sigma
        return ((self.y_0 - delta, self.y_0 + delta),
                (self.x_0 - delta, self.x_0 + delta))

    @property
    def bounding_box(self):
        """
        The bounding box of the model.

        Examples
        --------
        >>> from photutils.psf import CircularGaussianPRF
        >>> model = CircularGaussianPRF(x_0=0, y_0=0, fwhm=2)
        >>> model.bounding_box  # doctest: +FLOAT_CMP
        ModelBoundingBox(
            intervals={
                x: Interval(lower=-4.671269901584105, upper=4.671269901584105)
                y: Interval(lower=-4.671269901584105, upper=4.671269901584105)
            }
            model=CircularGaussianPRF(inputs=('x', 'y'))
            order='C'
        )
        >>> model.bbox_factor = 7
        >>> model.bounding_box  # doctest: +FLOAT_CMP
        ModelBoundingBox(
            intervals={
                x: Interval(lower=-5.945252602016134, upper=5.945252602016134)
                y: Interval(lower=-5.945252602016134, upper=5.945252602016134)
            }
            model=CircularGaussianPRF(inputs=('x', 'y'))
            order='C'
        )
        """
        return self._calc_bounding_box(factor=self.bbox_factor)

    def evaluate(self, x, y, flux, x_0, y_0, sigma):
        """
        Calculate the value of the 2D Gaussian model at the input
        coordinates for the given model parameters.

        Parameters
        ----------
        x, y : float or array_like
            The coordinates at which to evaluate the model.

        flux : float
            The total flux of the star.

        x_0, y_0 : float
            The position of the star.

        sigma : float
            The width of the Gaussian PRF.

        Returns
        -------
        evaluated_model : `~numpy.ndarray`
            The evaluated model.
        """
        dpix = 0.5
        if isinstance(x_0, u.Quantity):
            dpix *= x_0.unit

        return (flux / 4
                * ((erf((x - x_0 + dpix) / (np.sqrt(2) * sigma))
                    - erf((x - x_0 - dpix) / (np.sqrt(2) * sigma)))
                   * (erf((y - y_0 + dpix) / (np.sqrt(2) * sigma))
                      - erf((y - y_0 - dpix) / (np.sqrt(2) * sigma)))))

    @property
    def input_units(self):
        pass

    def _parameter_units_for_data_units(self, inputs_unit, outputs_unit):
        pass


class MoffatPSF(Fittable2DModel):

    flux = Parameter(
        default=1, description='Total integrated flux over the entire PSF.')
    x_0 = Parameter(
        default=0, description='Position of the peak along the x axis')
    y_0 = Parameter(
        default=0, description='Position of the peak along the y axis')
    alpha = Parameter(
        default=1,
        bounds=(FLOAT_EPSILON, None),
        fixed=True,
        description='Characteristic radius of the Moffat profile')
    beta = Parameter(
        default=2,
        bounds=(1.0 + FLOAT_EPSILON, None),
        fixed=True,
        description='Power-law index of the Moffat profile')

    def __init__(self, *, flux=flux.default, x_0=x_0.default, y_0=y_0.default,
                 alpha=alpha.default, beta=beta.default, bbox_factor=10.0,
                 **kwargs):
        super().__init__(flux=flux, x_0=x_0, y_0=y_0, alpha=alpha, beta=beta,
                         **kwargs)
        self.bbox_factor = bbox_factor

    @property
    def fwhm(self):
        pass

    def _calc_bounding_box(self, *, factor=10.0):
        """
        Calculate a bounding box defining the limits of the model.

        Parameters
        ----------
        factor : float, optional
            The multiple of the FWHM used to define the limits.

        Returns
        -------
        bbox : tuple
            A bounding box defining the ((y_min, y_max), (x_min, x_max))
            limits of the model.
        """
        delta = factor * self.fwhm
        return ((self.y_0 - delta, self.y_0 + delta),
                (self.x_0 - delta, self.x_0 + delta))

    @property
    def bounding_box(self):
        """
        The bounding box of the model.

        Examples
        --------
        >>> from photutils.psf import MoffatPSF
        >>> model = MoffatPSF(x_0=0, y_0=0, alpha=2, beta=3)
        >>> model.bounding_box  # doctest: +FLOAT_CMP
        ModelBoundingBox(
            intervals={
                x: Interval(lower=-20.39298114135835, upper=20.39298114135835)
                y: Interval(lower=-20.39298114135835, upper=20.39298114135835)
            }
            model=MoffatPSF(inputs=('x', 'y'))
            order='C'
        )
        >>> model.bbox_factor = 7
        >>> model.bounding_box  # doctest: +FLOAT_CMP
        ModelBoundingBox(
            intervals={
                x: Interval(lower=-14.27508679895084, upper=14.27508679895084)
                y: Interval(lower=-14.27508679895084, upper=14.27508679895084)
            }
            model=MoffatPSF(inputs=('x', 'y'))
            order='C'
        )
        """
        return self._calc_bounding_box(factor=self.bbox_factor)

    def evaluate(self, x, y, flux, x_0, y_0, alpha, beta):
        """
        Calculate the value of the 2D Moffat model at the input
        coordinates for the given model parameters.

        Parameters
        ----------
        x, y : float or array_like
            The x and y coordinates at which to evaluate the model.

        flux : float
            Total integrated flux over the entire PSF.

        x_0, y_0 : float
            Position of the peak along the x and y axes.

        alpha : float, optional
            The characteristic radius of the Moffat profile.

        beta : float, optional
            The asymptotic power-law slope of the Moffat profile wings
            at large radial distances. Larger values provide less flux
            in the profile wings. For large ``beta``, this profile
            approaches a Gaussian profile. ``beta`` must be greater
            than 1. If ``beta`` is set to 1, then the Moffat profile is
            a Lorentz function, whose integral is infinite. For this
            normalized model, if ``beta`` is set to 1, then the profile
            will be zero everywhere.

        Returns
        -------
        result : `~numpy.ndarray`
            The value of the model evaluated at the input coordinates.
        """
        alpha2 = alpha.copy()
        if isinstance(alpha, u.Quantity):
            alpha2 = alpha.value

        amp = flux * (beta - 1) / (np.pi * alpha2 ** 2)
        r2 = (x - x_0) ** 2 + (y - y_0) ** 2
        return amp * (1 + (r2 / alpha**2)) ** (-beta)

    @property
    def input_units(self):
        pass

    def _parameter_units_for_data_units(self, inputs_unit, outputs_unit):
        pass


class AiryDiskPSF(Fittable2DModel):

    flux = Parameter(
        default=1, description='Total integrated flux over the entire PSF.')
    x_0 = Parameter(
        default=0, description='Position of the peak along the x axis')
    y_0 = Parameter(
        default=0, description='Position of the peak along the y axis')
    radius = Parameter(
        default=1,
        bounds=(FLOAT_EPSILON, None),
        fixed=True,
        description='Radius of the Airy disk at the first zero')

    _rz = jn_zeros(1, 1)[0] / np.pi

    def __init__(self, *, flux=flux.default, x_0=x_0.default, y_0=y_0.default,
                 radius=radius.default, bbox_factor=10.0, **kwargs):
        super().__init__(flux=flux, x_0=x_0, y_0=y_0, radius=radius, **kwargs)
        self.bbox_factor = bbox_factor

    @property
    def fwhm(self):
        pass

    def _calc_bounding_box(self, *, factor=10.0):
        """
        Calculate a bounding box defining the limits of the model.

        Parameters
        ----------
        factor : float, optional
            The multiple of the FWHM used to define the limits.

        Returns
        -------
        bbox : tuple
            A bounding box defining the ((y_min, y_max), (x_min, x_max))
            limits of the model.
        """
        delta = factor * self.fwhm
        return ((self.y_0 - delta, self.y_0 + delta),
                (self.x_0 - delta, self.x_0 + delta))

    @property
    def bounding_box(self):
        """
        The bounding box of the model.

        Examples
        --------
        >>> from photutils.psf import AiryDiskPSF
        >>> model = AiryDiskPSF(x_0=0, y_0=0, radius=3)
        >>> model.bounding_box  # doctest: +FLOAT_CMP
        ModelBoundingBox(
            intervals={
                x: Interval(lower=-25.30997880648709, upper=25.30997880648709)
                y: Interval(lower=-25.30997880648709, upper=25.30997880648709)
            }
            model=AiryDiskPSF(inputs=('x', 'y'))
            order='C'
        )
        >>> model.bbox_factor = 7
        >>> model.bounding_box  # doctest: +FLOAT_CMP
        ModelBoundingBox(
            intervals={
                x: Interval(lower=-17.71698516454096, upper=17.71698516454096)
                y: Interval(lower=-17.71698516454096, upper=17.71698516454096)
            }
            model=AiryDiskPSF(inputs=('x', 'y'))
            order='C'
        )
        """
        return self._calc_bounding_box(factor=self.bbox_factor)

    def evaluate(self, x, y, flux, x_0, y_0, radius):
        """
        Calculate the value of the 2D Airy disk model at the input
        coordinates for the given model parameters.

        Parameters
        ----------
        x, y : float or array_like
            The x and y coordinates at which to evaluate the model.

        flux : float
            Total integrated flux over the entire PSF.

        x_0, y_0 : float
            Position of the peak along the x and y axes.

        radius : float, optional
            The radius of the Airy disk at the first zero.

        Returns
        -------
        result : `~numpy.ndarray`
            The value of the model evaluated at the input coordinates.
        """
        r = np.sqrt((x - x_0) ** 2 + (y - y_0) ** 2) / (radius / self._rz)

        if isinstance(r, u.Quantity):
            r = r.to_value(u.dimensionless_unscaled)

        z = np.ones(r.shape)
        rt = np.pi * r[r > 0]
        z[r > 0] = (2.0 * j1(rt) / rt) ** 2

        if isinstance(flux, u.Quantity):
            z <<= u.dimensionless_unscaled

        normalization = (4.0 / np.pi) * (radius / self._rz) ** 2
        if isinstance(normalization, u.Quantity):
            normalization = normalization.value

        z *= (flux / normalization)

        return z

    @property
    def input_units(self):
        pass

    def _parameter_units_for_data_units(self, inputs_unit, outputs_unit):
        pass
