
import copy

import numpy as np
from astropy.modeling import Fittable2DModel, Parameter
from astropy.utils.decorators import lazyproperty
from scipy.interpolate import RectBivariateSpline

from photutils.utils._parameters import as_pair

__all__ = ['ImagePSF']


class ImagePSF(Fittable2DModel):

    flux = Parameter(default=1,
                     description='Intensity scaling factor of the image.')
    x_0 = Parameter(default=0,
                    description=('Position of a feature in the image along '
                                 'the x axis'))
    y_0 = Parameter(default=0,
                    description=('Position of a feature in the image along '
                                 'the y axis'))

    def __init__(self, data, *, flux=flux.default, x_0=x_0.default,
                 y_0=y_0.default, origin=None, oversampling=1,
                 fill_value=0.0, **kwargs):

        self._validate_data(data)
        self.data = data
        self.origin = origin
        self.oversampling = as_pair('oversampling', oversampling,
                                    lower_bound=(0, 0))
        self.fill_value = fill_value

        super().__init__(flux, x_0, y_0, **kwargs)

    @staticmethod
    def _validate_data(data):
        if not isinstance(data, np.ndarray):
            msg = 'Input data must be a 2D numpy array'
            raise TypeError(msg)

        if data.ndim != 2:
            msg = 'Input data must be a 2D numpy array'
            raise ValueError(msg)

        if not np.all(np.isfinite(data)):
            msg = 'All elements of input data must be finite'
            raise ValueError(msg)

        if np.any(np.array(data.shape) < 4):
            msg = 'The length of the x and y axes must both be at least 4'
            raise ValueError(msg)

    def __str__(self):
        keywords = [('PSF shape (oversampled pixels)', self.data.shape),
                    ('Origin', self.origin.tolist()),
                    ('Oversampling', self.oversampling.tolist()),
                    ('Fill Value', self.fill_value),
                    ]
        return self._format_str(keywords=keywords)

    def __repr__(self):
        kwargs = {'origin': self.origin.tolist(),
                  'oversampling': self.oversampling.tolist(),
                  'fill_value': self.fill_value}
        return self._format_repr(kwargs=kwargs)

    def copy(self):
        """
        Return a copy of this model where only the model parameters are
        copied.

        All other copied model attributes are references to the original
        model. This prevents copying the image data, which may be a
        large array.

        This method is useful if one is interested in only changing
        the model parameters in a model copy. It is used in the PSF
        photometry classes during model fitting.

        Use the `deepcopy` method if you want to copy all the model
        attributes, including the PSF image data.

        Returns
        -------
        result : `ImagePSF`
            A copy of this model with only the model parameters copied.
        """
        newcls = object.__new__(self.__class__)

        for key, val in self.__dict__.items():
            if key in self.param_names:  # copy only the parameter values
                newcls.__dict__[key] = copy.copy(val)
            else:
                newcls.__dict__[key] = val

        return newcls

    def deepcopy(self):
        """
        Return a deep copy of this model.

        Returns
        -------
        result : `ImagePSF`
            A deep copy of this model.
        """
        return copy.deepcopy(self)

    @property
    def shape(self):
        pass

    @property
    def origin(self):
        pass

    @origin.setter
    def origin(self, origin):
        pass

    @lazyproperty
    def interpolator(self):
        """
        The interpolating spline function.

        The interpolator is computed with a 3rd-degree
        `~scipy.interpolate.RectBivariateSpline` (kx=3, ky=3, s=0) using
        the input image data. The interpolator is used to evaluate
        the model at arbitrary locations, including fractional pixel
        positions.

        Notes
        -----
        This property can be overridden in a subclass to define custom
        interpolators.
        """
        x = np.arange(self.data.shape[1])
        y = np.arange(self.data.shape[0])
        return RectBivariateSpline(x, y, self.data.T, kx=3, ky=3, s=0)

    def _calc_bounding_box(self):
        """
        Set a bounding box defining the limits of the model.

        Returns
        -------
        bbox : tuple
            A bounding box defining the ((y_min, y_max), (x_min, x_max))
            limits of the model.
        """
        dy, dx = np.array(self.data.shape) / 2 / self.oversampling

        xshift = np.array(self.data.shape[1] - 1) / 2 - self.origin[0]
        yshift = np.array(self.data.shape[0] - 1) / 2 - self.origin[1]
        xshift /= self.oversampling[1]
        yshift /= self.oversampling[0]

        return ((self.y_0 - dy + yshift, self.y_0 + dy + yshift),
                (self.x_0 - dx + xshift, self.x_0 + dx + xshift))

    @property
    def bounding_box(self):
        """
        The bounding box of the model.

        Examples
        --------
        >>> from photutils.psf import ImagePSF
        >>> psf_data = np.arange(30, dtype=float).reshape(5, 6)
        >>> psf_data /= np.sum(psf_data)
        >>> model = ImagePSF(psf_data, flux=1, x_0=0, y_0=0)
        >>> model.bounding_box  # doctest: +FLOAT_CMP
        ModelBoundingBox(
            intervals={
                x: Interval(lower=-3.0, upper=3.0)
                y: Interval(lower=-2.5, upper=2.5)
            }
            model=ImagePSF(inputs=('x', 'y'))
            order='C'
        )
        """
        return self._calc_bounding_box()

    def evaluate(self, x, y, flux, x_0, y_0):
        """
        Calculate the value of the image model at the input coordinates
        for the given model parameters.

        Parameters
        ----------
        x, y : float or array_like
            The x and y coordinates at which to evaluate the model.

        flux : float
            The total flux of the source, assuming the input image
            was properly normalized.

        x_0, y_0 : float
            The x and y positions of the feature in the image in the
            output coordinate grid on which the model is evaluated.

        Returns
        -------
        result : `~numpy.ndarray`
            The value of the model evaluated at the input coordinates.
        """
        xi = self.oversampling[1] * (np.asarray(x, dtype=float) - x_0)
        yi = self.oversampling[0] * (np.asarray(y, dtype=float) - y_0)
        xi += self._origin[0]
        yi += self._origin[1]

        evaluated_model = flux * self.interpolator(xi, yi, grid=False)

        if self.fill_value is not None:
            ny, nx = self.data.shape
            invalid = (xi < 0) | (xi > nx - 1) | (yi < 0) | (yi > ny - 1)
            evaluated_model[invalid] = self.fill_value

        return evaluated_model
