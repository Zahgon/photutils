
import functools
import inspect
import warnings
from copy import deepcopy

import astropy.units as u
import numpy as np
from astropy.nddata import NDData, StdDevUncertainty
from astropy.stats import (SigmaClip, biweight_location, biweight_midvariance,
                           mad_std)
from astropy.utils import lazyproperty
from astropy.utils.exceptions import AstropyUserWarning

from photutils.aperture import Aperture, SkyAperture, region_to_aperture
from photutils.aperture.core import _aperture_metadata
from photutils.morphology import gini as gini_func
from photutils.utils._deprecation import (create_empty_deprecated_qtable,
                                          deprecated_getattr,
                                          deprecated_positional_kwargs)
from photutils.utils._misc import _get_meta
from photutils.utils._moments import _image_moments
from photutils.utils._quantity_helpers import process_quantities

__all__ = ['ApertureStats']


DEFAULT_COLUMNS = ['id', 'x_centroid', 'y_centroid', 'sky_centroid',
                   'sum', 'sum_err', 'sum_aper_area', 'center_aper_area',
                   'min', 'max', 'mean', 'median', 'mode', 'std',
                   'mad_std', 'var', 'biweight_location',
                   'biweight_midvariance', 'fwhm', 'semimajor_axis',
                   'semiminor_axis', 'orientation', 'eccentricity']

_DEPRECATED_ATTRIBUTES: dict = {
    'covar_sigx2': 'covariance_xx',
    'covar_sigxy': 'covariance_xy',
    'covar_sigy2': 'covariance_yy',
    'cxx': 'ellipse_cxx',
    'cxy': 'ellipse_cxy',
    'cyy': 'ellipse_cyy',
    'data_sumcutout': 'data_sum_cutout',
    'error_sumcutout': 'error_sum_cutout',
    'get_id': 'select_id',
    'get_ids': 'select_ids',
    'semimajor_sigma': 'semimajor_axis',
    'semiminor_sigma': 'semiminor_axis',
    'xcentroid': 'x_centroid',
    'ycentroid': 'y_centroid',
}


def as_scalar(method):
    pass


class ApertureStats:

    def __init__(self, data, aperture, *, error=None, mask=None, wcs=None,
                 sigma_clip=None, sum_method='exact', subpixels=5,
                 local_bkg=None):

        if isinstance(data, NDData):
            data, error, mask, wcs = self._unpack_nddata(data, error, mask,
                                                         wcs)

        inputs = (data, error, local_bkg)
        names = ('data', 'error', 'local_bkg')
        inputs, unit = process_quantities(inputs, names)
        (data, error, local_bkg) = inputs

        self._data = self._validate_array(data, 'data', shape=False)
        self._data_unit = unit
        self._input_aperture = self._validate_aperture(aperture)
        aperture_meta = _aperture_metadata(aperture)  # use input aperture

        if isinstance(aperture, SkyAperture) and wcs is None:
            msg = 'A wcs is required when using a SkyAperture'
            raise ValueError(msg)

        if not isinstance(aperture, Aperture):
            aperture = region_to_aperture(aperture)
        self.aperture = aperture

        self._error = self._validate_array(error, 'error')
        self._mask = self._validate_array(mask, 'mask')
        self._wcs = wcs

        if sigma_clip is not None and not isinstance(sigma_clip, SigmaClip):
            msg = 'sigma_clip must be a SigmaClip instance'
            raise TypeError(msg)
        self.sigma_clip = sigma_clip

        self.sum_method = sum_method
        self.subpixels = subpixels

        self._local_bkg = np.zeros(self.n_apertures)  # no local bkg
        if local_bkg is not None:
            local_bkg = np.atleast_1d(local_bkg)
            if local_bkg.ndim != 1:
                msg = 'local_bkg must be a 1D array'
                raise ValueError(msg)

            n_local_bkg = len(local_bkg)
            if n_local_bkg not in (1, self.n_apertures):
                msg = ('local_bkg must be scalar or have the same length '
                       'as the input aperture')
                raise ValueError(msg)
            local_bkg = np.broadcast_to(local_bkg, self.n_apertures)

            if np.any(~np.isfinite(local_bkg)):
                msg = ('local_bkg must not contain any non-finite '
                       '(e.g., inf or NaN) values')
                raise ValueError(msg)
            self._local_bkg = local_bkg  # always an iterable

        self._ids = np.arange(self.n_apertures) + 1
        self.default_columns = DEFAULT_COLUMNS
        self.meta = _get_meta()
        self.meta.update(aperture_meta)

    @staticmethod
    def _unpack_nddata(data, error, mask, wcs):
        nddata_attr = {'error': error, 'mask': mask, 'wcs': wcs}
        for key, value in nddata_attr.items():
            if value is not None:
                msg = (f'The {key!r} keyword will be ignored. Its value '
                       'is obtained from the input NDData object.')
                warnings.warn(msg, AstropyUserWarning)

        mask = data.mask
        wcs = data.wcs

        if isinstance(data.uncertainty, StdDevUncertainty):
            if data.uncertainty.unit is None:
                error = data.uncertainty.array
            else:
                error = data.uncertainty.array * data.uncertainty.unit

        if data.unit is not None:
            data = u.Quantity(data.data, unit=data.unit)
        else:
            data = data.data

        return data, error, mask, wcs

    @staticmethod
    def _validate_aperture(aperture):
        try:
            from regions import Region
            aper_types = (Aperture, Region)
        except ImportError:
            aper_types = Aperture

        if not isinstance(aperture, aper_types):
            msg = 'aperture must be an Aperture or Region object'
            raise TypeError(msg)
        return aperture

    def _validate_array(self, array, name, *, ndim=2, shape=True):
        if name == 'mask' and array is np.ma.nomask:
            array = None
        if array is not None:
            array = np.asanyarray(array)
            if array.ndim != ndim:
                msg = f'{name} must be a {ndim}D array'
                raise ValueError(msg)
            if shape and array.shape != self._data.shape:
                msg = f'data and {name} must have the same shape'
                raise ValueError(msg)
        return array

    @property
    def _lazyproperties(self):
        pass

    @property
    def properties(self):
        pass

    def __getitem__(self, index):
        if self.isscalar:
            msg = (f'A scalar {self.__class__.__name__!r} object cannot '
                   'be indexed')
            raise TypeError(msg)

        newcls = object.__new__(self.__class__)

        init_attr = ('_data', '_data_unit', '_error', '_mask', '_wcs',
                     'sigma_clip', 'sum_method', 'subpixels',
                     'default_columns', 'meta')
        for attr in init_attr:
            setattr(newcls, attr, getattr(self, attr))

        attrs = ('aperture', '_ids')
        for attr in attrs:
            setattr(newcls, attr, getattr(self, attr)[index])

        keys = set(self.__dict__.keys()) & set(self._lazyproperties)
        keys.add('_local_bkg')  # iterable defined in __init__
        for key in keys:
            value = self.__dict__[key]

            if np.isscalar(value):
                continue

            try:
                if (newcls.isscalar and key.startswith('_')
                        and key != '_pixel_aperture'):
                    if isinstance(value, np.ndarray):
                        val = value[:, np.newaxis][index]
                    else:
                        val = [value[index]]
                else:
                    val = value[index]
            except TypeError:
                arr = np.empty(len(value), dtype=object)
                arr[:] = list(value)
                val = arr[index].tolist()

            newcls.__dict__[key] = val
        return newcls

    def __str__(self):
        cls_name = f'<{self.__class__.__module__}.{self.__class__.__name__}>'
        with np.printoptions(threshold=25, edgeitems=5):
            fmt = [f'Length: {self.n_apertures}']
        return f'{cls_name}\n' + '\n'.join(fmt)

    def __repr__(self):
        return self.__str__()

    def __len__(self):
        if self.isscalar:
            msg = f'Scalar {self.__class__.__name__!r} object has no len()'
            raise TypeError(msg)
        return self.n_apertures

    def __iter__(self):
        for item in range(len(self)):
            yield self.__getitem__(item)

    def __getattr__(self, name):
        return deprecated_getattr(self, name, _DEPRECATED_ATTRIBUTES,
                                  since='3.0', until='4.0')

    @lazyproperty
    def isscalar(self):
        """
        Whether the instance is scalar (e.g., a single aperture
        position).
        """
        return self._pixel_aperture.isscalar

    def copy(self):
        """
        Return a deep copy of this object.

        Returns
        -------
        result : `ApertureStats`
            A deep copy of this object.
        """
        return deepcopy(self)

    @lazyproperty
    def _null_object(self):
        pass

    @lazyproperty
    def _null_value(self):
        pass

    @property
    @as_scalar
    def id(self):
        pass

    @property
    def ids(self):
        pass

    def select_id(self, id_num):
        pass

    def select_ids(self, id_nums):
        pass

    @deprecated_positional_kwargs(since='3.0', until='4.0')
    def to_table(self, *, columns=None):
        pass

    @lazyproperty
    def n_apertures(self):
        pass

    @lazyproperty
    def _pixel_aperture(self):
        pass

    @lazyproperty
    def _aperture_masks_center(self):
        pass

    @lazyproperty
    def _aperture_masks(self):
        pass

    @lazyproperty
    def _overlap_slices(self):
        pass

    @lazyproperty
    def _data_cutouts(self):
        pass

    def _make_aperture_cutouts(self, aperture_masks):
        pass

    @lazyproperty
    def _aperture_cutouts_center(self):
        pass

    @lazyproperty
    def _aperture_cutouts(self):
        pass

    @lazyproperty
    def _mask_cutout_center(self):
        pass

    @lazyproperty
    def _mask_cutout(self):
        pass

    def _make_masked_array_center(self, array):
        pass

    def _make_masked_array(self, array):
        pass

    @lazyproperty
    @as_scalar
    def data_cutout(self):
        pass

    @lazyproperty
    @as_scalar
    def data_sum_cutout(self):
        pass

    @lazyproperty
    def _variance_cutout_center(self):
        pass

    @lazyproperty
    def _variance_cutout(self):
        pass

    @lazyproperty
    @as_scalar
    def error_sum_cutout(self):
        pass

    @lazyproperty
    def _weight_cutout_center(self):
        pass

    @lazyproperty
    def _weight_cutout(self):
        pass

    @lazyproperty
    def _moment_data_cutout(self):
        pass

    @lazyproperty
    def _all_masked(self):
        pass

    @lazyproperty
    def _overlap(self):
        pass

    def _get_values(self, array):
        pass

    @lazyproperty
    def _data_values_center(self):
        pass

    @lazyproperty
    @as_scalar
    def moments(self):
        pass

    @lazyproperty
    @as_scalar
    def moments_central(self):
        pass

    @lazyproperty
    @as_scalar
    def cutout_centroid(self):
        pass

    @lazyproperty
    @as_scalar
    def centroid(self):
        pass

    @lazyproperty
    def _x_centroid(self):
        pass

    @lazyproperty
    @as_scalar
    def x_centroid(self):
        pass

    @lazyproperty
    def _y_centroid(self):
        pass

    @lazyproperty
    @as_scalar
    def y_centroid(self):
        pass

    @lazyproperty
    @as_scalar
    def sky_centroid(self):
        pass

    @lazyproperty
    @as_scalar
    def sky_centroid_icrs(self):
        pass

    @lazyproperty
    def _bbox(self):
        pass

    @lazyproperty
    @as_scalar
    def bbox(self):
        pass

    @lazyproperty
    @as_scalar
    def _bbox_bounds(self):
        pass

    @lazyproperty
    @as_scalar
    def bbox_xmin(self):
        pass

    @lazyproperty
    @as_scalar
    def bbox_xmax(self):
        pass

    @lazyproperty
    @as_scalar
    def bbox_ymin(self):
        pass

    @lazyproperty
    @as_scalar
    def bbox_ymax(self):
        pass

    def _calculate_stats(self, stat_func, *, unit=None):
        """
        Apply the input ``stat_func`` to the 1D array of unmasked data
        values in the aperture.

        Units are applied if the input ``data`` has units.

        Parameters
        ----------
        stat_func : callable
            The callable to apply to the 1D `~numpy.ndarray` of unmasked
            data values.

        unit : `None` or `astropy.unit.Unit`, optional
            The unit to apply to the output data. This is used only
            if the input ``data`` has units. If `None` then the input
            ``data`` unit will be used.
        """
        result = np.array([stat_func(arr) for arr in self._data_values_center])
        if unit is None:
            unit = self._data_unit
        if unit is not None:
            result <<= unit
        return result

    @lazyproperty
    @as_scalar
    def center_aper_area(self):
        pass

    @lazyproperty
    @as_scalar
    def sum_aper_area(self):
        pass

    @lazyproperty
    @as_scalar
    def sum(self):
        pass

    @lazyproperty
    @as_scalar
    def sum_err(self):
        pass

    @lazyproperty
    @as_scalar
    def min(self):
        pass

    @lazyproperty
    @as_scalar
    def max(self):
        pass

    @lazyproperty
    @as_scalar
    def mean(self):
        """
        The mean of the unmasked pixel values within the aperture.
        """
        return self._calculate_stats(np.mean)

    @lazyproperty
    @as_scalar
    def median(self):
        pass

    @lazyproperty
    @as_scalar
    def mode(self):
        pass

    @lazyproperty
    @as_scalar
    def std(self):
        """
        The standard deviation of the unmasked pixel values within the
        aperture.
        """
        return self._calculate_stats(np.std)

    @lazyproperty
    @as_scalar
    def mad_std(self):
        r"""
        The standard deviation calculated using
        the `median absolute deviation (MAD)
        <https://en.wikipedia.org/wiki/Median_absolute_deviation>`_.

        The standard deviation estimator is given by:

        .. math::

            \sigma \approx \frac{\textrm{MAD}}{\Phi^{-1}(3/4)}
                \approx 1.4826 \ \textrm{MAD}

        where :math:`\Phi^{-1}(P)` is the normal inverse cumulative
        distribution function evaluated at probability :math:`P = 3/4`.
        """
        return self._calculate_stats(mad_std)

    @lazyproperty
    @as_scalar
    def var(self):
        pass

    @lazyproperty
    @as_scalar
    def biweight_location(self):
        """
        The biweight location of the unmasked pixel values within the
        aperture.

        See `astropy.stats.biweight_location`.
        """
        return self._calculate_stats(biweight_location)

    @lazyproperty
    @as_scalar
    def biweight_midvariance(self):
        """
        The biweight midvariance of the unmasked pixel values within the
        aperture.

        See `astropy.stats.biweight_midvariance`
        """
        unit = self._data_unit
        if unit is not None:
            unit **= 2
        return self._calculate_stats(biweight_midvariance, unit=unit)

    @lazyproperty
    @as_scalar
    def inertia_tensor(self):
        pass

    @lazyproperty
    def _covariance(self):
        pass

    @lazyproperty
    @as_scalar
    def covariance(self):
        pass

    @lazyproperty
    @as_scalar
    def covariance_eigvals(self):
        pass

    @lazyproperty
    @as_scalar
    def semimajor_axis(self):
        pass

    @lazyproperty
    @as_scalar
    def semiminor_axis(self):
        pass

    @lazyproperty
    @as_scalar
    def fwhm(self):
        pass

    @lazyproperty
    @as_scalar
    def orientation(self):
        pass

    @lazyproperty
    @as_scalar
    def eccentricity(self):
        pass

    @lazyproperty
    @as_scalar
    def elongation(self):
        pass

    @lazyproperty
    @as_scalar
    def ellipticity(self):
        pass

    @lazyproperty
    @as_scalar
    def covariance_xx(self):
        pass

    @lazyproperty
    @as_scalar
    def covariance_yy(self):
        pass

    @lazyproperty
    @as_scalar
    def covariance_xy(self):
        pass

    @lazyproperty
    @as_scalar
    def ellipse_cxx(self):
        pass

    @lazyproperty
    @as_scalar
    def ellipse_cyy(self):
        pass

    @lazyproperty
    @as_scalar
    def ellipse_cxy(self):
        pass

    @lazyproperty
    @as_scalar
    def gini(self):
        r"""
        The `Gini coefficient
        <https://en.wikipedia.org/wiki/Gini_coefficient>`_ of the
        unmasked pixel values within the aperture.

        The Gini coefficient of the distribution of absolute flux values
        is calculated using the prescription from `Lotz et al. 2004
        <https://ui.adsabs.harvard.edu/abs/2004AJ....128..163L/abstract>`_
        (Eq. 6) as:

        .. math::

            G = \frac{1}{\overline{|x|} \, n \, (n - 1)}
                \sum^{n}_{i} (2i - n - 1) \left | x_i \right |

        where :math:`\overline{|x|}` is the mean of the absolute value
        of all pixel values :math:`x_i`. If the sum of all pixel values
        is zero, the Gini coefficient is zero.

        Negative pixel values are used via their absolute value. Invalid
        values (NaN and inf) in the input are automatically excluded
        from the calculation. If only a single finite pixel remains
        after filtering, the Gini coefficient is 0.0.
        """
        return np.array([gini_func(arr) for arr in self._data_values_center])
