
import abc
import inspect
import warnings
from copy import deepcopy

import numpy as np
from astropy.coordinates import SkyCoord
from astropy.utils import lazyproperty

from photutils.aperture.bounding_box import BoundingBox
from photutils.aperture.mask import ApertureMask
from photutils.utils._deprecation import deprecated_positional_kwargs

__all__ = ['Aperture', 'PixelAperture', 'SkyAperture']


class Aperture(metaclass=abc.ABCMeta):

    _params = ()

    def __len__(self):
        if self.isscalar:
            msg = f'A scalar {self.__class__.__name__!r} object has no len()'
            raise TypeError(msg)
        return self.shape[0]

    def __getitem__(self, index):
        if self.isscalar:
            msg = (f'A scalar {self.__class__.__name__!r} object cannot be '
                   'indexed')
            raise TypeError(msg)

        kwargs = {}
        for param in self._params:
            if param == 'positions':
                kwargs[param] = getattr(self, param)[index]
            else:
                kwargs[param] = getattr(self, param)
        return self.__class__(**kwargs)

    def __iter__(self):
        for i in range(len(self)):
            yield self.__getitem__(i)

    def _positions_str(self, *, prefix=None):
        pass

    def __repr__(self):
        prefix = f'{self.__class__.__name__}'
        cls_info = []
        for param in self._params:
            if param == 'positions':
                cls_info.append(self._positions_str(prefix=prefix))
            else:
                cls_info.append(f'{param}={getattr(self, param)}')
        cls_info = ', '.join(cls_info)
        return f'<{prefix}({cls_info})>'

    def __str__(self):
        cls_info = [('Aperture', self.__class__.__name__)]
        for param in self._params:
            if param == 'positions':
                prefix = 'positions'
                cls_info.append((prefix,
                                 self._positions_str(prefix=prefix + ': ')))
            else:
                cls_info.append((param, getattr(self, param)))
        fmt = [f'{key}: {val}' for key, val in cls_info]
        return '\n'.join(fmt)

    def __eq__(self, other):
        """
        Equality operator for `Aperture`.

        All Aperture properties are compared for strict equality except
        for Quantity parameters, which allow for different units if they
        are directly convertible.
        """
        if not isinstance(other, self.__class__):
            return False

        self_params = list(self._params)
        other_params = list(other._params)

        if self_params != other_params:
            return False

        try:
            for param in self_params:
                if np.any(getattr(self, param) != getattr(other, param)):
                    return False
        except TypeError:
            return False

        return True

    def __ne__(self, other):
        """
        Inequality operator for `Aperture`.
        """
        return not self == other

    @property
    def _lazyproperties(self):
        pass

    def copy(self):
        """
        Make a deep copy of this object.

        Returns
        -------
        result : `Aperture`
            A deep copy of the Aperture object.
        """
        params_copy = {}
        for param in list(self._params):
            params_copy[param] = deepcopy(getattr(self, param))
        return self.__class__(**params_copy)

    @abc.abstractmethod
    def positions(self):
        """
        The aperture positions, as an array of (x, y) coordinates or a
        `~astropy.coordinates.SkyCoord`.
        """

    @lazyproperty
    def shape(self):
        pass

    @lazyproperty
    def isscalar(self):
        """
        Whether the instance is scalar (i.e., a single position).
        """
        return self.shape == ()


class PixelAperture(Aperture):

    @lazyproperty
    def _default_patch_properties(self):
        pass

    @staticmethod
    def _translate_mask_method(method, subpixels, *, rectangle=False):
        """
        Translate the mask method and subpixels parameters to the values
        used by the low-level `photutils.geometry` functions.

        Parameters
        ----------
        method : {'exact', 'center', 'subpixel'}
            The mask method.

        subpixels : int
            The number of subpixels for subpixel method.

        rectangle : bool, optional
            Whether the aperture is a rectangular aperture. This is
            used to approximate the "exact" method for rectangular
            apertures, which is not currently supported by the low-level
            `photutils.geometry` functions.

        Returns
        -------
        use_exact : int
            Whether to use exact method (1) or not (0).

        subpixels : int
            The number of subpixels for subpixel method.
        """
        if method not in ('center', 'subpixel', 'exact'):
            msg = f'Invalid mask method: {method}'
            raise ValueError(msg)

        if rectangle and method == 'exact':
            method = 'subpixel'
            subpixels = 32

        if ((method == 'subpixel')
                and (not isinstance(subpixels, int) or subpixels <= 0)):
            msg = 'subpixels must be a strictly positive integer'
            raise ValueError(msg)

        if method == 'center':
            use_exact = 0
            subpixels = 1
        elif method == 'subpixel':
            use_exact = 0
        elif method == 'exact':
            use_exact = 1
            subpixels = 1

        return use_exact, subpixels

    @property
    @abc.abstractmethod
    def _xy_extents(self):
        """
        The (x, y) extents of the aperture measured from the center
        position.

        In other words, the (x, y) extents are half of the aperture
        minimal bounding box size in each dimension.
        """

    @lazyproperty
    def _positions(self):
        pass

    @lazyproperty
    def _bbox(self):
        pass

    @lazyproperty
    def bbox(self):
        pass

    @lazyproperty
    def _centered_edges(self):
        pass

    @property
    @abc.abstractmethod
    def area(self):
        """
        The exact geometric area of the aperture shape.

        Use the `area_overlap` method to return the area of overlap
        between the data and the aperture, taking into account the
        aperture mask method, masked data pixels (``mask`` keyword), and
        partial/no overlap of the aperture with the data.

        Returns
        -------
        area : float
            The aperture area.

        See Also
        --------
        area_overlap
        """

    def area_overlap(self, data, *, mask=None, method='exact', subpixels=5):
        pass

    @deprecated_positional_kwargs(since='3.0', until='4.0')
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
        use_exact, subpixels = self._translate_mask_method(
            method, subpixels, rectangle=getattr(self, '_is_rectangle', False))

        masks = []
        for bbox, edges in zip(self._bbox, self._centered_edges, strict=True):
            ny, nx = bbox.shape
            overlap = self._compute_overlap(
                edges, nx, ny, use_exact, subpixels)
            masks.append(ApertureMask(overlap, bbox))

        if self.isscalar:
            return masks[0]

        return masks

    @abc.abstractmethod
    def _compute_overlap(self, edges, nx, ny, use_exact, subpixels):
        """
        Compute the overlap of the aperture for a single position.

        Parameters
        ----------
        edges : tuple of float
            The ``(xmin, xmax, ymin, ymax)`` pixel edges centered at
            the origin.

        nx, ny : int
            The number of pixels in x and y.

        use_exact : int
            Whether to use exact method (1) or not (0).

        subpixels : int
            The number of subpixels for subpixel method.

        Returns
        -------
        overlap : 2D `~numpy.ndarray`
            The overlap array.
        """

    @deprecated_positional_kwargs(since='3.0', until='4.0')
    def do_photometry(self, data, error=None, mask=None, method='exact',
                      subpixels=5):
        """
        Perform aperture photometry on the input data.

        Parameters
        ----------
        data : array_like or `~astropy.units.Quantity` instance
            The 2D array on which to perform photometry. ``data`` should
            be background subtracted.

        error : array_like or `~astropy.units.Quantity`, optional
            The pixel-wise Gaussian 1-sigma errors of the input
            ``data``. ``error`` is assumed to include *all* sources
            of error, including the Poisson error of the sources (see
            `~photutils.utils.calc_total_error`). ``error`` must have
            the same shape as the input ``data``.

        mask : array_like (bool), optional
            A boolean mask with the same shape as ``data`` where a
            `True` value indicates the corresponding element of ``data``
            is masked. Masked data are excluded from all calculations.

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
        aperture_sums : `~numpy.ndarray` or `~astropy.units.Quantity`
            The sum within each aperture.

        aperture_sum_errs : `~numpy.ndarray` or `~astropy.units.Quantity`
            The errors on the sum within each aperture.

        Notes
        -----
        `RectangularAperture` and `RectangularAnnulus` photometry with
        the "exact" method uses a subpixel approximation by subdividing
        each data pixel by a factor of 1024 (``subpixels = 32``). For
        rectangular aperture widths and heights in the range from
        2 to 100 pixels, this subpixel approximation gives results
        typically within 0.001 percent or better of the exact value.
        The differences can be larger for smaller apertures (e.g.,
        aperture sizes of one pixel or smaller). For such small sizes,
        it is recommended to set ``method='subpixel'`` with a larger
        ``subpixels`` size.
        """
        data = np.asanyarray(data)
        if data.ndim != 2:
            msg = 'data must be a 2D array'
            raise ValueError(msg)

        if error is not None:
            error = np.asanyarray(error)
            if error.shape != data.shape:
                msg = 'error and data must have the same shape'
                raise ValueError(msg)

        unit = {getattr(arr, 'unit', None) for arr in (data, error)
                if arr is not None}
        if len(unit) > 1:
            msg = ('If data or error has units, then they both must have '
                   'the same units')
            raise ValueError(msg)

        unit = unit.pop()
        if unit is not None:
            unit = data.unit
            data = data.value

            if error is not None:
                error = error.value

        apermasks = self.to_mask(method=method, subpixels=subpixels)
        if self.isscalar:
            apermasks = (apermasks,)

        aperture_sums = []
        aperture_sum_errs = []
        for apermask in apermasks:
            (slc_large,
             aper_weights,
             pixel_mask) = apermask._get_overlap_cutouts(data.shape, mask=mask)

            if slc_large is None:
                aperture_sums.append(np.nan)
                aperture_sum_errs.append(np.nan)
                continue

            with warnings.catch_warnings():
                warnings.simplefilter('ignore', RuntimeWarning)

                values = (data[slc_large] * aper_weights)[pixel_mask]
                aperture_sums.append(values.sum())

                if error is not None:
                    variance = (error[slc_large]**2 * aper_weights)[pixel_mask]
                    aperture_sum_errs.append(np.sqrt(variance.sum()))

        aperture_sums = np.array(aperture_sums)
        aperture_sum_errs = np.array(aperture_sum_errs)

        if unit is not None:
            aperture_sums <<= unit
            aperture_sum_errs <<= unit

        return aperture_sums, aperture_sum_errs

    @staticmethod
    def _make_annulus_path(patch_inner, patch_outer):
        pass

    def _define_patch_params(self, *, origin=(0, 0), **kwargs):
        pass

    @abc.abstractmethod
    def _to_patch(self, *, origin=(0, 0), **kwargs):
        """
        Return a `~matplotlib.patches.Patch` for the aperture.

        Parameters
        ----------
        origin : array_like, optional
            The ``(x, y)`` position of the origin of the displayed
            image.

        **kwargs : dict, optional
            Any keyword arguments accepted by
            `matplotlib.patches.Patch`.

        Returns
        -------
        patch : `~matplotlib.patches.Patch` or list of \
                `~matplotlib.patches.Patch`
            A patch for the aperture. If the aperture is scalar then a
            single `~matplotlib.patches.Patch` is returned, otherwise a
            list of `~matplotlib.patches.Patch` is returned.
        """

    @deprecated_positional_kwargs(since='3.0', until='4.0')
    def plot(self, ax=None, origin=(0, 0), **kwargs):
        pass

    @abc.abstractmethod
    def to_sky(self, wcs):
        """
        Convert the aperture to a `SkyAperture` object defined in
        celestial coordinates.

        Parameters
        ----------
        wcs : WCS object
            A world coordinate system (WCS) transformation that
            supports the `astropy shared interface for WCS
            <https://docs.astropy.org/en/stable/wcs/wcsapi.html>`_
            (e.g., `astropy.wcs.WCS`, `gwcs.wcs.WCS`).

        Returns
        -------
        aperture : `SkyAperture` object
            A `SkyAperture` object.
        """


class SkyAperture(Aperture):

    @abc.abstractmethod
    def to_pixel(self, wcs):
        """
        Convert the aperture to a `PixelAperture` object defined in
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
        aperture : `PixelAperture` object
            A `PixelAperture` object.
        """


def _aperture_metadata(aperture, *, index=''):
    """
    Return a dictionary of aperture metadata.

    Parameters
    ----------
    aperture : `Aperture`
        An aperture object.

    index : str, optional
        A string that will be prepended to each metadata key.

    Returns
    -------
    meta : dict
        A dictionary of aperture metadata
    """
    params = aperture._params
    meta = {}
    meta[f'aperture{index}'] = aperture.__class__.__name__
    for param in params:
        if param != 'positions':
            meta[f'aperture{index}_{param}'] = getattr(aperture, param)
    return meta
