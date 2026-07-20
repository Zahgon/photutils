
import copy
import warnings

import astropy.units as u
import numpy as np
from astropy.nddata import NDData, block_replicate, reshape_as_blocks
from astropy.utils import lazyproperty
from astropy.utils.exceptions import AstropyUserWarning
from scipy.ndimage import generic_filter

from photutils.aperture import RectangularAperture
from photutils.background.core import (SIGMA_CLIP, SExtractorBackground,
                                       StdBackgroundRMS)
from photutils.background.interpolators import (BkgIDWInterpolator,
                                                _BkgZoomInterpolator)
from photutils.utils import ShepardIDWInterpolator
from photutils.utils._deprecation import (deprecated,
                                          deprecated_renamed_argument)
from photutils.utils._parameters import as_pair, create_default_sigmaclip
from photutils.utils._repr import make_repr
from photutils.utils._stats import nanmedian, nanmin

__all__ = ['Background2D']

__doctest_skip__ = ['Background2D']


class Background2D:

    @deprecated_renamed_argument('bkgrms_estimator', 'bkg_rms_estimator',
                                 '3.0', until='4.0')
    @deprecated_renamed_argument('interpolator', None, '3.0', until='4.0')
    def __init__(self, data, box_size, *, mask=None, coverage_mask=None,
                 fill_value=0.0, exclude_percentile=10.0, filter_size=(3, 3),
                 filter_threshold=None, sigma_clip=SIGMA_CLIP,
                 bkg_estimator=None, bkg_rms_estimator=None,
                 interpolator=None):

        if isinstance(data, (u.Quantity, NDData)):  # includes CCDData
            self._unit = data.unit
            data = data.data
        else:
            self._unit = None

        self._data = self._validate_array(data, 'data', shape=False)
        self._data_dtype = self._data.dtype
        self._data_shape = self._data.shape
        if np.all(~np.isfinite(self._data)):
            msg = ('Input data contains all non-finite (NaN or infinity) '
                   'values. Cannot compute a background.')
            raise ValueError(msg)

        self._mask = self._validate_array(mask, 'mask')
        self._has_mask = self._mask is not None
        self.coverage_mask = self._validate_array(coverage_mask,
                                                  'coverage_mask')

        self.box_size = as_pair('box_size', box_size, lower_bound=(0, 0),
                                upper_bound=data.shape)

        self.fill_value = fill_value
        if exclude_percentile < 0 or exclude_percentile > 100:
            msg = 'exclude_percentile must be between 0 and 100 (inclusive)'
            raise ValueError(msg)
        self.exclude_percentile = exclude_percentile
        self.filter_size = as_pair('filter_size', filter_size,
                                   lower_bound=(0, 0), check_odd=True)
        self.filter_threshold = filter_threshold

        if sigma_clip is SIGMA_CLIP:
            sigma_clip = create_default_sigmaclip(sigma=SIGMA_CLIP.sigma,
                                                  maxiters=SIGMA_CLIP.maxiters)
        self.sigma_clip = sigma_clip

        if interpolator is None:
            interpolator = _BkgZoomInterpolator()
        self.interpolator = interpolator

        if bkg_estimator is None:
            bkg_estimator = SExtractorBackground(sigma_clip=None)
        if bkg_rms_estimator is None:
            bkg_rms_estimator = StdBackgroundRMS(sigma_clip=None)

        bkg_estimator = copy.copy(bkg_estimator)
        bkg_rms_estimator = copy.copy(bkg_rms_estimator)
        if hasattr(bkg_estimator, 'sigma_clip'):
            bkg_estimator.sigma_clip = None
        if hasattr(bkg_rms_estimator, 'sigma_clip'):
            bkg_rms_estimator.sigma_clip = None
        self.bkg_estimator = bkg_estimator
        self.bkg_rms_estimator = bkg_rms_estimator

        self._box_npixels = None

        interp_dtype = self._data.dtype
        if interp_dtype.kind != 'f':
            interp_dtype = np.float32
        self._interp_kwargs = {'shape': self._data.shape,
                               'dtype': interp_dtype,
                               'box_size': self.box_size}

        (self._bkg_stats,
         self._bkgrms_stats,
         self._ngood) = self._calculate_stats()

        self._min_bkg_stats = nanmin(self._bkg_stats)

        self._mesh_nan_mask = np.isnan(self._bkg_stats)

        if isinstance(self.interpolator, BkgIDWInterpolator):
            self._interp_kwargs['mesh_yxcen'] = self._calculate_mesh_yxcen()
            self._interp_kwargs['mesh_nan_mask'] = self._mesh_nan_mask

    def _repr_str_params(self):
        pass

    def __repr__(self):
        params, overrides = self._repr_str_params()
        return make_repr(self, params, overrides=overrides)

    def __str__(self):
        params, overrides = self._repr_str_params()
        return make_repr(self, params, overrides=overrides, long=True)

    def _validate_array(self, array, name, *, shape=True):
        """
        Validate the input data, mask, and coverage_mask arrays.
        """
        if name in ('mask', 'coverage_mask') and array is np.ma.nomask:
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

    def _apply_units(self, data):
        pass

    def _combine_input_masks(self):
        """
        Combine the input mask and coverage_mask.
        """
        if self._mask is None and self.coverage_mask is None:
            return None
        if self._mask is None:
            return self.coverage_mask
        if self.coverage_mask is None:
            return self._mask

        return np.logical_or(self._mask, self.coverage_mask)

    def _combine_all_masks(self, mask):
        """
        Combine the input masks (mask and coverage_mask) with the mask
        of invalid data values.
        """
        input_mask = self._combine_input_masks()

        msg = ('Input data contains non-finite (NaN or infinity) values, '
               'which were automatically masked.')

        if input_mask is None:
            if np.any(mask):
                warnings.warn(msg, AstropyUserWarning)

            total_mask = mask
        else:
            condition = np.logical_and(np.logical_not(input_mask), mask)
            if np.any(condition):
                warnings.warn(msg, AstropyUserWarning)

            total_mask = np.logical_or(input_mask, mask)

        if np.all(total_mask):
            msg = 'All input pixels are masked. Cannot compute a background.'
            raise ValueError(msg)

        return total_mask

    @lazyproperty
    def _good_npixels_threshold(self):
        pass

    def _sigmaclip_boxes(self, data, axis):
        """
        Sigma clip the boxes along the specified axis.

        This method sigma clips the boxes along the specified axis and
        returns the sigma-clipped data. The input ``data`` is typically
        a 4D array where the first two dimensions represent the y and x
        positions of the boxes and the last two dimensions represent the
        y and x positions within each box.

        We perform sigma clipping as a separate step to avoid performing
        sigma clipping for both the background and background RMS
        estimators.

        Parameters
        ----------
        data : `~numpy.ndarray`
            The 4D array of box data.

        axis : int or tuple of int
            The axis or axes along which to sigma clip the data.

        Returns
        -------
        data : `~numpy.ndarray`
            The sigma-clipped data.
        """
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', category=AstropyUserWarning)
            if self.sigma_clip is not None:
                data = self.sigma_clip(data, axis=axis, masked=False,
                                       copy=False)

        return data

    def _compute_box_statistics(self, data, *, axis=None):
        """
        Compute the background and background RMS statistics in each
        box.

        Parameters
        ----------
        data : `~numpy.ndarray`
            The 4D array of box data.

        axis : int or tuple of int, optional
            The axis or axes along which to compute the statistics.

        Returns
        -------
        bkg : 2D `~numpy.ndarray` or float
            The background statistics in each box.

        bkgrms : 2D `~numpy.ndarray` or float
            The background RMS statistics in each box.
        """
        data = self._sigmaclip_boxes(data, axis=axis)

        bkg = self.bkg_estimator(data, axis=axis)
        bkgrms = self.bkg_rms_estimator(data, axis=axis)

        ngood = np.count_nonzero(~np.isnan(data), axis=axis)
        box_mask = ngood <= self._good_npixels_threshold

        if np.ndim(bkg) == 0:
            if box_mask:  # single corner box
                bkg = np.float32(np.nan)
                bkgrms = np.float32(np.nan)
        else:
            bkg[box_mask] = np.nan
            bkgrms[box_mask] = np.nan

        return bkg, bkgrms, ngood

    def _calculate_stats(self):
        """
        Calculate the background and background RMS statistics in each
        box.

        Returns
        -------
        bkg : 2D `~numpy.ndarray`
            The background statistics in each box.

        bkgrms : 2D `~numpy.ndarray`
            The background RMS statistics in each box.

        ngood : 2D `~numpy.ndarray`
            The number of unmasked pixels in each box.
        """
        if self._data.dtype.kind != 'f':
            self._data = self._data.astype(np.float32)

        mask = self._combine_all_masks(~np.isfinite(self._data))

        self._box_npixels = np.prod(self.box_size)
        nboxes = self._data.shape // self.box_size
        y1, x1 = nboxes * self.box_size

        core = reshape_as_blocks(self._data[:y1, :x1].copy(), self.box_size)
        core_mask = reshape_as_blocks(mask[:y1, :x1], self.box_size)
        core = core.reshape((*nboxes, -1))
        core_mask = core_mask.reshape((*nboxes, -1))
        core[core_mask] = np.nan
        bkg, bkgrms, ngood = self._compute_box_statistics(core, axis=-1)

        extra_row = y1 < self._data.shape[0]
        extra_col = x1 < self._data.shape[1]
        if extra_row or extra_col:
            if extra_row:
                row_data = self._data[y1:, :x1].copy()
                row_mask = mask[y1:, :x1]
                row_data[row_mask] = np.nan
                row_data = reshape_as_blocks(row_data, (1, self.box_size[1]))
                row_data = np.moveaxis(row_data, 0, -1)
                row_data = row_data.reshape((*row_data.shape[:-2], -1))
                row_bkg, row_bkgrms, row_ngood = self._compute_box_statistics(
                    row_data, axis=-1)

            if extra_col:
                col_data = self._data[:y1, x1:].copy()
                col_mask = mask[:y1, x1:]
                col_data[col_mask] = np.nan
                col_data = reshape_as_blocks(col_data, (self.box_size[0], 1))
                col_data = np.transpose(col_data, (0, 3, 1, 2))
                col_data = col_data.reshape((*col_data.shape[:-2], -1))
                col_bkg, col_bkgrms, col_ngood = self._compute_box_statistics(
                    col_data, axis=-1)

            if extra_row and extra_col:
                corner_data = self._data[y1:, x1:].copy()
                corner_mask = mask[y1:, x1:]
                corner_data[corner_mask] = np.nan
                crn_bkg, crn_bkgrms, crn_ngood = self._compute_box_statistics(
                    corner_data, axis=None)
                col_bkg = np.vstack((col_bkg, crn_bkg))
                col_bkgrms = np.vstack((col_bkgrms, crn_bkgrms))
                col_ngood = np.vstack((col_ngood, crn_ngood))

            if extra_row:
                bkg = np.vstack([bkg, row_bkg[:, 0]])
                bkgrms = np.vstack([bkgrms, row_bkgrms[:, 0]])
                ngood = np.vstack([ngood, row_ngood[:, 0]])

            if extra_col:
                bkg = np.hstack([bkg, col_bkg])
                bkgrms = np.hstack([bkgrms, col_bkgrms])
                ngood = np.hstack([ngood, col_ngood])

        if np.all(np.isnan(bkg)):
            msg = (f'All boxes contain <= {self._good_npixels_threshold} '
                   f'unmasked or finite pixels ({self.box_size=}, '
                   f'{self.exclude_percentile=}). Please check your data '
                   'or increase "exclude_percentile" to allow more boxes to '
                   'be included.')
            raise ValueError(msg)

        del self._data
        del self._mask

        return bkg, bkgrms, ngood

    def _interpolate_grid(self, data, *, n_neighbors=10, eps=0.0, power=1.0,
                          regularization=0.0):
        pass

    def _selective_filter(self, data):
        pass

    def _filter_grid(self, data):
        pass

    def _calculate_mesh_yxcen(self):
        """
        Calculate the y and x positions of the centers of the low-
        resolution background and background RMS meshes with respect to
        the input data array.

        This is used by the IDW interpolator to expand the low-
        resolution mesh to the full-size image. It is also used to plot
        the mesh boxes on the input image.
        """
        mesh_idx = np.where(~self._mesh_nan_mask)  # good mesh indices
        box_cen = (self.box_size - 1) / 2.0
        return (mesh_idx * self.box_size[:, None]) + box_cen[:, None]

    def _try_free_bkg_stats(self):
        pass

    @lazyproperty
    def background_mesh(self):
        pass

    @lazyproperty
    def background_rms_mesh(self):
        pass

    @property
    @deprecated(since='3.0', alternative='n_pixels_mesh', until='4.0')
    def npixels_mesh(self):
        pass

    @property
    def n_pixels_mesh(self):
        pass

    @property
    @deprecated(since='3.0', alternative='n_pixels_map', until='4.0')
    def npixels_map(self):
        pass

    @property
    def n_pixels_map(self):
        pass

    @lazyproperty
    def background_median(self):
        pass

    @lazyproperty
    def background_rms_median(self):
        pass

    def _calculate_image(self, data):
        pass

    @property
    def background(self):
        pass

    @property
    def background_rms(self):
        pass

    def plot_meshes(self, *, ax=None, marker='+', markersize=None,
                    color='blue', alpha=None, outlines=False, **kwargs):
        pass
