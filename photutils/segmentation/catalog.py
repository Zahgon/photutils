
import functools
import inspect
import math
import warnings
from copy import deepcopy

import astropy.units as u
import numpy as np
from astropy.stats import SigmaClip, gaussian_fwhm_to_sigma
from astropy.utils import lazyproperty
from scipy.ndimage import map_coordinates
from scipy.optimize import root_scalar

from photutils.aperture import (BoundingBox, CircularAperture,
                                EllipticalAperture, RectangularAnnulus)
from photutils.background import SExtractorBackground
from photutils.geometry import circular_overlap_grid, elliptical_overlap_grid
from photutils.morphology import gini as gini_func
from photutils.segmentation.core import SegmentationImage
from photutils.segmentation.utils import _mask_to_mirrored_value
from photutils.utils._deprecation import (_get_future_column_names,
                                          create_empty_deprecated_qtable,
                                          deprecated_getattr,
                                          deprecated_positional_kwargs,
                                          deprecated_renamed_argument)
from photutils.utils._misc import _get_meta
from photutils.utils._progress_bars import add_progress_bar
from photutils.utils._quantity_helpers import process_quantities
from photutils.utils.cutouts import CutoutImage

__all__ = ['SourceCatalog']


DEFAULT_COLUMNS = ['label', 'x_centroid', 'y_centroid', 'sky_centroid',
                   'bbox_xmin', 'bbox_xmax', 'bbox_ymin', 'bbox_ymax',
                   'area', 'semimajor_axis', 'semiminor_axis', 'orientation',
                   'eccentricity', 'min_value', 'max_value', 'segment_flux',
                   'segment_flux_err', 'kron_flux', 'kron_flux_err']

_DEPRECATED_ATTRIBUTES = {
    'add_extra_property': 'add_property',
    'apermask_method': 'aperture_mask_method',
    'background': 'background_cutout',
    'background_ma': 'background_cutout_masked',
    'convdata': 'conv_data_cutout',
    'convdata_ma': 'conv_data_cutout_masked',
    'covar_sigx2': 'covariance_xx',
    'covar_sigxy': 'covariance_xy',
    'covar_sigy2': 'covariance_yy',
    'cutout_maxval_index': 'cutout_max_value_index',
    'cutout_minval_index': 'cutout_min_value_index',
    'cxx': 'ellipse_cxx',
    'cxy': 'ellipse_cxy',
    'cyy': 'ellipse_cyy',
    'data': 'data_cutout',
    'data_ma': 'data_cutout_masked',
    'error': 'error_cutout',
    'error_ma': 'error_cutout_masked',
    'extra_properties': 'custom_properties',
    'fluxfrac_radius': 'flux_radius',
    'get_label': 'select_label',
    'get_labels': 'select_labels',
    'kron_fluxerr': 'kron_flux_err',
    'localbkg_width': 'local_bkg_width',
    'maxval_index': 'max_value_index',
    'maxval_xindex': 'max_value_xindex',
    'maxval_yindex': 'max_value_yindex',
    'minval_index': 'min_value_index',
    'minval_xindex': 'min_value_xindex',
    'minval_yindex': 'min_value_yindex',
    'nlabels': 'n_labels',
    'remove_extra_properties': 'remove_properties',
    'remove_extra_property': 'remove_property',
    'rename_extra_property': 'rename_property',
    'segment': 'segment_cutout',
    'segment_fluxerr': 'segment_flux_err',
    'segment_ma': 'segment_cutout_masked',
    'semimajor_sigma': 'semimajor_axis',
    'semiminor_sigma': 'semiminor_axis',
    'xcentroid': 'x_centroid',
    'xcentroid_quad': 'x_centroid_quad',
    'xcentroid_win': 'x_centroid_win',
    'ycentroid': 'y_centroid',
    'ycentroid_quad': 'y_centroid_quad',
    'ycentroid_win': 'y_centroid_win',
}

_DEPRECATED_META_KEYS = {
    'localbkg_width': 'local_bkg_width',
    'apermask_method': 'aperture_mask_method',
}


def as_scalar(method):
    pass


def use_detcat(method):
    pass


class SourceCatalog:

    @deprecated_renamed_argument('segment_img', 'segmentation_image', '3.0',
                                 until='4.0')
    @deprecated_renamed_argument('localbkg_width', 'local_bkg_width',
                                 '3.0', until='4.0')
    @deprecated_renamed_argument('apermask_method', 'aperture_mask_method',
                                 '3.0', until='4.0')
    @deprecated_renamed_argument('detection_cat', 'detection_catalog', '3.0',
                                 until='4.0')
    def __init__(self, data, segmentation_image, *, convolved_data=None,
                 error=None, mask=None, background=None, wcs=None,
                 local_bkg_width=0, aperture_mask_method='correct',
                 kron_params=(2.5, 1.4, 0.0), detection_catalog=None,
                 progress_bar=False):

        inputs = (data, convolved_data, error, background)
        names = ('data', 'convolved_data', 'error', 'background')
        inputs, unit = process_quantities(inputs, names)
        (data, convolved_data, error, background) = inputs

        self._data_unit = unit
        self._data = self._validate_array(data, 'data', shape=False)
        self._convolved_data = self._validate_array(convolved_data,
                                                    'convolved_data')
        self._segmentation_image = self._validate_segmentation_image(
            segmentation_image)
        self._error = self._validate_array(error, 'error')
        self._mask = self._validate_array(mask, 'mask')
        self._background = self._validate_array(background, 'background')
        self.wcs = wcs
        self.local_bkg_width = self._validate_local_bkg_width(
            local_bkg_width)
        self.aperture_mask_method = self._validate_aperture_mask_method(
            aperture_mask_method)
        self.kron_params = self._validate_kron_params(kron_params)
        self.progress_bar = progress_bar

        self._slices = self._segmentation_image.slices
        self._labels = self._segmentation_image.labels

        if self._labels.shape == (0,):
            msg = 'segmentation_image must have at least one non-zero label'
            raise ValueError(msg)

        self._detection_catalog = self._validate_detection_catalog(
            detection_catalog)
        attrs = ('wcs', 'aperture_mask_method', 'kron_params')
        if self._detection_catalog is not None:
            for attr in attrs:
                setattr(self, attr, getattr(self._detection_catalog, attr))

        if convolved_data is None:
            self._convolved_data = self._data

        self._aperture_mask_kwargs = {
            'circ': {'method': 'exact'},
            'kron': {'method': 'exact'},
            'flux_radius': {'method': 'exact'},
            'cen_win': {'method': 'center'},
        }

        self.default_columns = DEFAULT_COLUMNS
        self._custom_properties = []
        self._flux_radius_cache = {}
        self.meta = _get_meta()
        self._update_meta()

    def _validate_segmentation_image(self, segmentation_image):
        if not isinstance(segmentation_image, SegmentationImage):
            msg = 'segmentation_image must be a SegmentationImage'
            raise TypeError(msg)
        if segmentation_image.shape != self._data.shape:
            msg = 'segmentation_image and data must have the same shape'
            raise ValueError(msg)
        return segmentation_image

    def _validate_array(self, array, name, *, shape=True):
        if name == 'mask' and array is np.ma.nomask:
            array = None
        if array is not None:
            array = np.asanyarray(array)
            if array.ndim != 2:
                msg = f'{name} must be a 2D array'
                raise ValueError(msg)
            if shape and array.shape != self._data.shape:
                msg = f'data and {name} must have the same shape'
                raise ValueError(msg)
        return array

    @staticmethod
    def _validate_local_bkg_width(local_bkg_width):
        if local_bkg_width < 0:
            msg = 'local_bkg_width must be >= 0'
            raise ValueError(msg)
        local_bkg_width_int = int(local_bkg_width)
        if local_bkg_width_int != local_bkg_width:
            msg = 'local_bkg_width must be an integer'
            raise ValueError(msg)
        return local_bkg_width_int

    @staticmethod
    def _validate_aperture_mask_method(aperture_mask_method):
        if aperture_mask_method not in ('none', 'mask', 'correct'):
            msg = 'Invalid aperture_mask_method value'
            raise ValueError(msg)
        return aperture_mask_method

    @staticmethod
    def _validate_kron_params(kron_params):
        if np.ndim(kron_params) != 1:
            msg = 'kron_params must be 1D'
            raise ValueError(msg)
        nparams = len(kron_params)
        if nparams not in (2, 3):
            msg = 'kron_params must have 2 or 3 elements'
            raise ValueError(msg)
        if kron_params[0] <= 0:
            msg = 'kron_params[0] must be > 0'
            raise ValueError(msg)
        if kron_params[1] <= 0:
            msg = 'kron_params[1] must be > 0'
            raise ValueError(msg)
        if nparams == 3 and kron_params[2] < 0:
            msg = 'kron_params[2] must be >= 0'
            raise ValueError(msg)
        return tuple(kron_params)

    def _validate_detection_catalog(self, detection_catalog):
        if detection_catalog is None:
            return None

        if not isinstance(detection_catalog, SourceCatalog):
            msg = 'detection_catalog must be a SourceCatalog instance'
            raise TypeError(msg)
        if not np.array_equal(detection_catalog._segmentation_image,
                              self._segmentation_image):
            msg = ('detection_catalog must have same segmentation_image '
                   'as the input segmentation_image')
            raise ValueError(msg)
        return detection_catalog

    def _update_meta(self):
        meta_values = {}
        attrs = ('local_bkg_width', 'aperture_mask_method', 'kron_params')
        for attr in attrs:
            meta_values[attr] = getattr(self, attr)

        if not _get_future_column_names():
            for old_name, new_name in _DEPRECATED_META_KEYS.items():
                if new_name in meta_values:
                    meta_values[old_name] = meta_values[new_name]

        self.meta.update(meta_values)

    def _set_semode(self):
        pass

    @property
    def _properties(self):
        pass

    @property
    def properties(self):
        pass

    @property
    def _lazyproperties(self):
        pass

    @staticmethod
    def _index_object_list(lst, index):
        pass

    def __getitem__(self, index):
        if self.isscalar:
            msg = (f'A scalar {self.__class__.__name__!r} object cannot '
                   'be indexed')
            raise TypeError(msg)

        newcls = object.__new__(self.__class__)

        init_attr = ('_data', '_segmentation_image', '_error', '_mask',
                     '_background', 'wcs', '_data_unit', '_convolved_data',
                     'local_bkg_width', 'aperture_mask_method',
                     'kron_params', 'default_columns', '_custom_properties',
                     'meta', '_aperture_mask_kwargs', 'progress_bar')
        for attr in init_attr:
            setattr(newcls, attr, getattr(self, attr))

        attr = '_labels'
        setattr(newcls, attr, getattr(self, attr)[index])

        attr = '_detection_catalog'
        if getattr(self, attr) is None:
            setattr(newcls, attr, None)
        else:
            setattr(newcls, attr, getattr(self, attr)[index])

        attr = '_slices'
        value = self._index_object_list(getattr(self, attr), index)
        setattr(newcls, attr, value)

        newcls._flux_radius_cache = {key: value[index]
                                     for key, value
                                     in self._flux_radius_cache.items()}

        keys = (set(self.__dict__.keys())
                & (set(self._lazyproperties) | set(self._custom_properties)))
        for key in keys:
            value = self.__dict__[key]

            if np.isscalar(value):
                continue

            try:
                if newcls.isscalar and key.startswith('_'):
                    if isinstance(value, np.ndarray):
                        val = value[:, np.newaxis][index]
                    else:
                        val = [value[index]]
                else:
                    val = value[index]
            except TypeError:
                val = self._index_object_list(value, index)

            newcls.__dict__[key] = val
        return newcls

    def __str__(self):
        cls_name = f'<{self.__class__.__module__}.{self.__class__.__name__}>'
        with np.printoptions(threshold=25, edgeitems=5):
            fmt = [f'Length: {self.n_labels}', f'labels: {self.labels}']
        return f'{cls_name}\n' + '\n'.join(fmt)

    def __repr__(self):
        return self.__str__()

    def __len__(self):
        if self.isscalar:
            msg = f'Scalar {self.__class__.__name__!r} object has no len()'
            raise TypeError(msg)
        return self.n_labels

    def __iter__(self):
        for item in range(len(self)):
            yield self.__getitem__(item)

    def __getattr__(self, name):
        return deprecated_getattr(self, name, _DEPRECATED_ATTRIBUTES,
                                  since='3.0', until='4.0')

    @lazyproperty
    def isscalar(self):
        """
        Whether the instance is scalar (e.g., a single source).
        """
        return self._labels.shape == ()

    @staticmethod
    def _has_len(value):
        pass

    def copy(self):
        """
        Return a deep copy of this SourceCatalog.

        Returns
        -------
        result : `SourceCatalog`
            A deep copy of this object.
        """
        return deepcopy(self)

    @property
    def custom_properties(self):
        pass

    @deprecated_positional_kwargs(since='3.0', until='4.0')
    def add_property(self, name, value, overwrite=False):
        pass

    def remove_property(self, name):
        pass

    def remove_properties(self, names):
        pass

    def rename_property(self, name, new_name):
        pass

    @lazyproperty
    def _null_objects(self):
        pass

    @lazyproperty
    def _null_values(self):
        pass

    @lazyproperty
    def _data_cutouts(self):
        pass

    @lazyproperty
    def _segmentation_image_cutouts(self):
        pass

    @lazyproperty
    def _mask_cutouts(self):
        pass

    @lazyproperty
    def _error_cutouts(self):
        pass

    @lazyproperty
    def _convdata_cutouts(self):
        pass

    @lazyproperty
    def _background_cutouts(self):
        pass

    @staticmethod
    def _make_cutout_data_mask(data_cutout, mask_cutout):
        pass

    def _make_cutout_data_masks(self, data_cutouts, mask_cutouts):
        pass

    @lazyproperty
    def _cutout_segment_masks(self):
        pass

    @lazyproperty
    def _cutout_data_masks(self):
        pass

    @lazyproperty
    def _cutout_total_masks(self):
        pass

    @lazyproperty
    def _moment_data_cutouts(self):
        pass

    def _prepare_cutouts(self, arrays, *, units=True, masked=False,
                         dtype=None):
        pass

    def select_label(self, label):
        pass

    def select_labels(self, labels):
        pass

    @deprecated_positional_kwargs(since='3.0', until='4.0')
    def to_table(self, columns=None):
        pass

    @lazyproperty
    def n_labels(self):
        pass

    @property
    @as_scalar
    def label(self):
        pass

    @property
    def labels(self):
        pass

    @property
    @as_scalar
    def slices(self):
        pass

    @lazyproperty
    def _slices_iter(self):
        pass

    @lazyproperty
    @as_scalar
    def segment_cutout(self):
        pass

    @lazyproperty
    @as_scalar
    def segment_cutout_masked(self):
        pass

    @lazyproperty
    @as_scalar
    def data_cutout(self):
        pass

    @lazyproperty
    @as_scalar
    def data_cutout_masked(self):
        pass

    @lazyproperty
    @as_scalar
    def conv_data_cutout(self):
        pass

    @lazyproperty
    @as_scalar
    def conv_data_cutout_masked(self):
        pass

    @lazyproperty
    @as_scalar
    def error_cutout(self):
        pass

    @lazyproperty
    @as_scalar
    def error_cutout_masked(self):
        pass

    @lazyproperty
    @as_scalar
    def background_cutout(self):
        pass

    @lazyproperty
    @as_scalar
    def background_cutout_masked(self):
        pass

    @lazyproperty
    def _all_masked(self):
        pass

    def _get_values(self, array):
        pass

    @staticmethod
    def _reduceat(values, ufunc, *, transform=None):
        pass

    @lazyproperty
    def _data_values(self):
        pass

    @lazyproperty
    def _error_values(self):
        pass

    @lazyproperty
    def _background_values(self):
        pass

    @lazyproperty
    @use_detcat
    @as_scalar
    def moments(self):
        pass

    @lazyproperty
    @use_detcat
    @as_scalar
    def moments_central(self):
        pass

    @lazyproperty
    @use_detcat
    @as_scalar
    def cutout_centroid(self):
        pass

    @lazyproperty
    @use_detcat
    @as_scalar
    def centroid(self):
        pass

    @lazyproperty
    @use_detcat
    def _x_centroid(self):
        pass

    @lazyproperty
    @use_detcat
    @as_scalar
    def x_centroid(self):
        pass

    @lazyproperty
    @use_detcat
    def _y_centroid(self):
        pass

    @lazyproperty
    @use_detcat
    @as_scalar
    def y_centroid(self):
        pass

    @lazyproperty
    @use_detcat
    @as_scalar
    def centroid_win(self):
        pass

    @lazyproperty
    @use_detcat
    @as_scalar
    def x_centroid_win(self):
        pass

    @lazyproperty
    @use_detcat
    @as_scalar
    def y_centroid_win(self):
        pass

    @lazyproperty
    @use_detcat
    @as_scalar
    def cutout_centroid_win(self):
        pass

    @lazyproperty
    @use_detcat
    @as_scalar
    def cutout_centroid_quad(self):
        pass

    @lazyproperty
    @use_detcat
    @as_scalar
    def centroid_quad(self):
        pass

    @lazyproperty
    @use_detcat
    @as_scalar
    def x_centroid_quad(self):
        pass

    @lazyproperty
    @use_detcat
    @as_scalar
    def y_centroid_quad(self):
        pass

    @lazyproperty
    @use_detcat
    @as_scalar
    def sky_centroid(self):
        pass

    @lazyproperty
    @use_detcat
    @as_scalar
    def sky_centroid_icrs(self):
        pass

    @lazyproperty
    @use_detcat
    @as_scalar
    def sky_centroid_win(self):
        pass

    @lazyproperty
    @use_detcat
    @as_scalar
    def sky_centroid_quad(self):
        pass

    @lazyproperty
    @use_detcat
    def _bbox(self):
        pass

    @lazyproperty
    @use_detcat
    @as_scalar
    def bbox(self):
        pass

    @lazyproperty
    @use_detcat
    @as_scalar
    def bbox_xmin(self):
        pass

    @lazyproperty
    @use_detcat
    @as_scalar
    def bbox_xmax(self):
        pass

    @lazyproperty
    @use_detcat
    @as_scalar
    def bbox_ymin(self):
        pass

    @lazyproperty
    @use_detcat
    @as_scalar
    def bbox_ymax(self):
        pass

    @lazyproperty
    @use_detcat
    def _bbox_corner_ll(self):
        pass

    @lazyproperty
    @use_detcat
    def _bbox_corner_ul(self):
        pass

    @lazyproperty
    @use_detcat
    def _bbox_corner_lr(self):
        pass

    @lazyproperty
    @use_detcat
    def _bbox_corner_ur(self):
        pass

    @lazyproperty
    @use_detcat
    @as_scalar
    def sky_bbox_ll(self):
        pass

    @lazyproperty
    @use_detcat
    @as_scalar
    def sky_bbox_ul(self):
        pass

    @lazyproperty
    @use_detcat
    @as_scalar
    def sky_bbox_lr(self):
        pass

    @lazyproperty
    @use_detcat
    @as_scalar
    def sky_bbox_ur(self):
        pass

    @lazyproperty
    @as_scalar
    def min_value(self):
        pass

    @lazyproperty
    @as_scalar
    def max_value(self):
        pass

    @lazyproperty
    @as_scalar
    def cutout_min_value_index(self):
        pass

    @lazyproperty
    @as_scalar
    def cutout_max_value_index(self):
        pass

    @lazyproperty
    @as_scalar
    def min_value_index(self):
        pass

    @lazyproperty
    @as_scalar
    def max_value_index(self):
        pass

    @lazyproperty
    @as_scalar
    def min_value_xindex(self):
        pass

    @lazyproperty
    @as_scalar
    def min_value_yindex(self):
        pass

    @lazyproperty
    @as_scalar
    def max_value_xindex(self):
        pass

    @lazyproperty
    @as_scalar
    def max_value_yindex(self):
        pass

    @lazyproperty
    @as_scalar
    def segment_flux(self):
        pass

    @lazyproperty
    @as_scalar
    def segment_flux_err(self):
        pass

    @lazyproperty
    @as_scalar
    def background_sum(self):
        pass

    @lazyproperty
    @as_scalar
    def background_mean(self):
        pass

    @lazyproperty
    @as_scalar
    def background_centroid(self):
        pass

    @lazyproperty
    @use_detcat
    @as_scalar
    def segment_area(self):
        pass

    @lazyproperty
    @use_detcat
    @as_scalar
    def area(self):
        pass

    @lazyproperty
    @use_detcat
    @as_scalar
    def equivalent_radius(self):
        pass

    @lazyproperty
    @use_detcat
    @as_scalar
    def perimeter(self):
        pass

    @lazyproperty
    @use_detcat
    @as_scalar
    def inertia_tensor(self):
        pass

    @lazyproperty
    @use_detcat
    def _covariance(self):
        pass

    @lazyproperty
    @use_detcat
    @as_scalar
    def covariance(self):
        pass

    @lazyproperty
    @use_detcat
    @as_scalar
    def covariance_eigvals(self):
        pass

    @lazyproperty
    @use_detcat
    @as_scalar
    def semimajor_axis(self):
        pass

    @lazyproperty
    @use_detcat
    @as_scalar
    def semiminor_axis(self):
        pass

    @lazyproperty
    @use_detcat
    @as_scalar
    def fwhm(self):
        pass

    @lazyproperty
    @use_detcat
    @as_scalar
    def orientation(self):
        pass

    @lazyproperty
    @use_detcat
    @as_scalar
    def eccentricity(self):
        pass

    @lazyproperty
    @use_detcat
    @as_scalar
    def elongation(self):
        pass

    @lazyproperty
    @use_detcat
    @as_scalar
    def ellipticity(self):
        pass

    @lazyproperty
    @use_detcat
    @as_scalar
    def covariance_xx(self):
        pass

    @lazyproperty
    @use_detcat
    @as_scalar
    def covariance_yy(self):
        pass

    @lazyproperty
    @use_detcat
    @as_scalar
    def covariance_xy(self):
        pass

    @lazyproperty
    @use_detcat
    @as_scalar
    def ellipse_cxx(self):
        pass

    @lazyproperty
    @use_detcat
    @as_scalar
    def ellipse_cyy(self):
        pass

    @lazyproperty
    @use_detcat
    @as_scalar
    def ellipse_cxy(self):
        pass

    @lazyproperty
    @use_detcat
    @as_scalar
    def gini(self):
        r"""
        The `Gini coefficient
        <https://en.wikipedia.org/wiki/Gini_coefficient>`_ of the
        source.

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
        return np.array([gini_func(arr) for arr in self._data_values])

    @lazyproperty
    def _local_background_apertures(self):
        pass

    @lazyproperty
    @use_detcat
    @as_scalar
    def local_background_aperture(self):
        pass

    @lazyproperty
    def _local_background(self):
        pass

    @lazyproperty
    @as_scalar
    def local_background(self):
        pass

    def _aperture_to_mask(self, aperture, **kwargs):
        pass

    def _make_aperture_data(self, label, x_centroid, y_centroid, aperture_bbox,
                            local_background, *, make_error=True):
        pass

    def _make_circular_apertures(self, radius):
        pass

    @as_scalar
    def make_circular_apertures(self, radius):
        pass

    @as_scalar
    @deprecated_positional_kwargs(since='3.0', until='4.0')
    def plot_circular_apertures(self, radius, ax=None, origin=(0, 0),
                                **kwargs):
        pass

    @deprecated_positional_kwargs(since='3.0', until='4.0')
    def circular_photometry(self, radius, name=None, overwrite=False):
        pass

    def _make_elliptical_apertures(self, *, scale=6.0):
        pass

    @lazyproperty
    @use_detcat
    def _measured_kron_radius(self):
        pass

    @as_scalar
    def _calc_kron_radius(self, kron_params):
        pass

    @lazyproperty
    @use_detcat
    @as_scalar
    def kron_radius(self):
        pass

    def _make_kron_apertures(self, kron_params):
        pass

    @lazyproperty
    @use_detcat
    @as_scalar
    def kron_aperture(self):
        pass

    @as_scalar
    @deprecated_positional_kwargs(since='3.0', until='4.0')
    def make_kron_apertures(self, kron_params=None):
        pass

    @as_scalar
    @deprecated_positional_kwargs(since='3.0', until='4.0')
    def plot_kron_apertures(self, kron_params=None, ax=None, origin=(0, 0),
                            **kwargs):
        pass

    def _aperture_photometry(self, apertures, *, desc='', **kwargs):
        pass

    def _calc_kron_photometry(self, *, kron_params=None):
        pass

    @deprecated_positional_kwargs(since='3.0', until='4.0')
    def kron_photometry(self, kron_params, name=None, overwrite=False):
        pass

    @lazyproperty
    def _kron_photometry(self):
        pass

    @lazyproperty
    @as_scalar
    def kron_flux(self):
        pass

    @lazyproperty
    @as_scalar
    def kron_flux_err(self):
        pass

    @lazyproperty
    @use_detcat
    def _max_circular_kron_radius(self):
        pass

    @staticmethod
    def _flux_radius_fcn(radius, clean_data, grid_params, normflux):
        pass

    @lazyproperty
    @use_detcat
    def _flux_radius_optimizer_args(self):
        pass

    @as_scalar
    @deprecated_positional_kwargs(since='3.0', until='4.0')
    def flux_radius(self, fraction, name=None, overwrite=False):
        pass

    @as_scalar
    def make_cutouts(self, shape, *, array=None, mode='partial',
                     fill_value=np.nan):
        pass
