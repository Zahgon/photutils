
import inspect
import warnings
from collections import defaultdict
from copy import copy, deepcopy

import numpy as np
from astropy.utils import lazyproperty
from astropy.utils.exceptions import AstropyUserWarning
from scipy.ndimage import find_objects, grey_dilation
from scipy.signal import fftconvolve

from photutils.aperture import BoundingBox
from photutils.aperture.converters import _shapely_polygon_to_region
from photutils.utils._deprecation import (deprecated_getattr,
                                          deprecated_positional_kwargs)
from photutils.utils._optional_deps import HAS_RASTERIO, HAS_SHAPELY
from photutils.utils._parameters import as_pair
from photutils.utils.colormaps import make_random_cmap

__all__ = ['Segment', 'SegmentationImage']

_SEGM_DEPRECATED_ATTRIBUTES = {
    'nlabels': 'n_labels',
    'data_ma': 'data_masked',
    'deblended_labels_map': 'deblended_label_to_parent',
    'deblended_labels_inverse_map': 'parent_to_deblended_labels',
}

_SEGMENT_DEPRECATED_ATTRIBUTES = {
    'data_ma': 'data_masked',
}


class SegmentationImage:

    def __init__(self, data):
        if not isinstance(data, np.ndarray):
            msg = 'Input data must be a numpy array'
            raise TypeError(msg)
        self.data = data
        self._deblend_label_map = {}  # set by source deblender

    def __str__(self):
        cls_name = f'<{self.__class__.__module__}.{self.__class__.__name__}>'

        params = ['shape', 'n_labels']
        cls_info = [(param, getattr(self, param)) for param in params]
        cls_info.append(('labels', self.labels))
        with np.printoptions(threshold=25, edgeitems=5):
            fmt = [f'{key}: {val}' for key, val in cls_info]

        return f'{cls_name}\n' + '\n'.join(fmt)

    def __repr__(self):
        return self.__str__()

    def __getattr__(self, name):
        return deprecated_getattr(self, name, _SEGM_DEPRECATED_ATTRIBUTES,
                                  since='3.0', until='4.0')

    def __getitem__(self, key):
        """
        Slice the segmentation image, returning a new SegmentationImage
        object.
        """
        if (isinstance(key, tuple) and len(key) == 2
                and all(isinstance(key[i], slice) for i in (0, 1))):
            result = self.data[key]
            if result.size == 0:
                msg = ('The sliced result is empty; cannot create '
                       'a SegmentationImage with zero size')
                raise ValueError(msg)
            return SegmentationImage(result)

        msg = f'{key!r} is not a valid 2D slice object'
        raise TypeError(msg)

    def __array__(self):
        """
        Array representation of the segmentation array (e.g., for
        matplotlib).
        """
        return self._data

    @staticmethod
    def _get_labels(data):
        """
        Return a sorted array of the non-zero labels in the segmentation
        image.

        Parameters
        ----------
        data : array_like (int)
            A segmentation array where source regions are labeled by
            different positive integer values. A value of zero is
            reserved for the background.

        Returns
        -------
        result : `~numpy.ndarray`
            An array of non-zero label numbers.

        Notes
        -----
        This is a static method so it can be used in
        :meth:`remove_masked_labels` on a masked version of the
        segmentation array.
        """
        return np.unique(data[data != 0])

    @lazyproperty
    def segments(self):
        pass

    @lazyproperty
    def deblended_labels(self):
        pass

    @lazyproperty
    def deblended_label_to_parent(self):
        pass

    @lazyproperty
    def parent_to_deblended_labels(self):
        pass

    @property
    def data(self):
        pass

    @property
    def _lazyproperties(self):
        pass

    def _reset_lazyproperties(self):
        pass

    @data.setter
    def data(self, value):
        pass

    @lazyproperty
    def data_masked(self):
        pass

    @lazyproperty
    def shape(self):
        pass

    @lazyproperty
    def _ndim(self):
        pass

    @lazyproperty
    def labels(self):
        pass

    @lazyproperty
    def n_labels(self):
        pass

    @lazyproperty
    def max_label(self):
        pass

    def get_index(self, label):
        pass

    def get_indices(self, labels):
        """
        Find the indices of the input ``labels``.

        Parameters
        ----------
        labels : int, array_like (1D, int)
            The label numbers(s) to find.

        Returns
        -------
        indices : int `~numpy.ndarray`
            An integer array of indices with the same shape as
            ``labels``. If ``labels`` is a scalar, then the returned
            index will also be a scalar.

        Raises
        ------
        ValueError
            If any input ``labels`` are invalid.
        """
        self.check_labels(labels)
        return np.searchsorted(self.labels, labels)

    @lazyproperty
    def _raw_slices(self):
        pass

    @lazyproperty
    def slices(self):
        pass

    @lazyproperty
    def bbox(self):
        pass

    @lazyproperty
    def background_area(self):
        pass

    @lazyproperty
    def areas(self):
        pass

    def get_area(self, label):
        pass

    def get_areas(self, labels):
        pass

    def _make_polygon(self, label, slc):
        pass

    def _make_segment(self, label):
        pass

    def get_segment(self, label):
        pass

    def get_segments(self, labels):
        pass

    @lazyproperty
    def is_consecutive(self):
        pass

    @lazyproperty
    def missing_labels(self):
        pass

    def copy(self):
        """
        Return a deep copy of this object.

        Returns
        -------
        result : `SegmentationImage`
            A deep copy of this object.
        """
        return deepcopy(self)

    def check_label(self, label):
        pass

    def check_labels(self, labels):
        """
        Check that the input label(s) are valid label numbers within the
        segmentation array.

        Parameters
        ----------
        labels : int, 1D array_like (int)
            The label(s) to check.

        Raises
        ------
        ValueError
            If any input ``labels`` are invalid.
        """
        labels = np.atleast_1d(labels)
        bad_labels = set()

        valid_mask = np.isin(labels, self.labels)
        bad_labels.update(labels[~valid_mask])

        if bad_labels:
            bad_labels = sorted(bad_labels)
            label_str = 'label'
            conj_str = 'is'
            if len(bad_labels) > 1:
                label_str = 'labels'
                conj_str = 'are'
            msg = f'{label_str} {bad_labels} {conj_str} invalid'
            raise ValueError(msg)

    def _make_cmap(self, n_colors, *, background_color='#000000ff',
                   seed=None):
        pass

    @deprecated_positional_kwargs(since='3.0', until='4.0')
    def make_cmap(self, background_color='#000000ff', seed=None):
        pass

    @deprecated_positional_kwargs(since='3.0', until='4.0')
    def reset_cmap(self, seed=None):
        pass

    @lazyproperty
    def cmap(self):
        pass

    def _update_deblend_label_map(self, relabel_map):
        """
        Update the deblended label map based on the input
        ``relabel_map``.

        Parameters
        ----------
        relabel_map : `~numpy.ndarray`
            An array mapping the original label numbers to the new label
            numbers.
        """
        for parent_label, child_labels in self._deblend_label_map.items():
            self._deblend_label_map[parent_label] = relabel_map[child_labels]

    @deprecated_positional_kwargs(since='3.0', until='4.0')
    def reassign_label(self, label, new_label, relabel=False):
        pass

    @deprecated_positional_kwargs(since='3.0', until='4.0')
    def reassign_labels(self, labels, new_label, relabel=False):
        pass

    @deprecated_positional_kwargs(since='3.0', until='4.0')
    def relabel_consecutive(self, start_label=1):
        pass

    @deprecated_positional_kwargs(since='3.0', until='4.0')
    def keep_label(self, label, relabel=False):
        pass

    @deprecated_positional_kwargs(since='3.0', until='4.0')
    def keep_labels(self, labels, relabel=False):
        pass

    @deprecated_positional_kwargs(since='3.0', until='4.0')
    def remove_label(self, label, relabel=False):
        pass

    @deprecated_positional_kwargs(since='3.0', until='4.0')
    def remove_labels(self, labels, relabel=False):
        pass

    @deprecated_positional_kwargs(since='3.0', until='4.0')
    def remove_border_labels(self, border_width, partial_overlap=True,
                             relabel=False):
        pass

    @deprecated_positional_kwargs(since='3.0', until='4.0')
    def remove_masked_labels(self, mask, partial_overlap=True,
                             relabel=False):
        pass

    def make_source_mask(self, *, size=None, footprint=None):
        pass

    @lazyproperty
    def _geojson_polygons(self):
        pass

    @lazyproperty
    def polygons(self):
        pass

    def get_polygon(self, label):
        pass

    def get_polygons(self, labels):
        pass

    @staticmethod
    def _convert_ring_to_path(ring):
        pass

    def _convert_shapely_to_pathpatch(self, geometry, *, origin=(0, 0),
                                      scale=1.0, **kwargs):
        pass

    def to_patches(self, *, origin=(0, 0), scale=1.0, **kwargs):
        pass

    def get_patch(self, label, *, origin=(0, 0), scale=1.0, **kwargs):
        pass

    def get_patches(self, labels, *, origin=(0, 0), scale=1.0, **kwargs):
        pass

    def plot_patches(self, *, ax=None, origin=(0, 0), scale=1.0, labels=None,
                     **kwargs):
        pass

    def to_regions(self, *, group=False, **kwargs):
        pass

    def get_region(self, label, **kwargs):
        pass

    def get_regions(self, labels, **kwargs):
        pass

    @deprecated_positional_kwargs(since='3.0', until='4.0')
    def imshow(self, ax=None, figsize=None, dpi=None, cmap=None, alpha=None):
        pass

    @deprecated_positional_kwargs(since='3.0', until='4.0')
    def imshow_map(self, ax=None, figsize=None, dpi=None, cmap=None,
                   alpha=None, max_labels=25, cbar_labelsize=None):
        pass


class Segment:

    def __init__(self, segment_data, label, slices, bbox, area, *,
                 polygon=None):
        self._segment_data_cutout = np.copy(segment_data[slices])
        self._segment_data_shape = segment_data.shape
        self.label = label
        self.slices = slices
        self.bbox = bbox
        self.area = area
        self.polygon = polygon

    def __str__(self):
        cls_name = f'<{self.__class__.__module__}.{self.__class__.__name__}>'

        params = ['label', 'slices', 'area']
        cls_info = [(param, getattr(self, param)) for param in params]

        fmt = [f'{key}: {val}' for key, val in cls_info]

        return f'{cls_name}\n' + '\n'.join(fmt)

    def __repr__(self):
        return self.__str__()

    def __getattr__(self, name):
        return deprecated_getattr(self, name,
                                  _SEGMENT_DEPRECATED_ATTRIBUTES,
                                  since='3.0', until='4.0')

    def _repr_svg_(self):
        pass

    def __array__(self):
        """
        Array representation of the labeled region (e.g., for
        matplotlib).
        """
        return self.data

    @lazyproperty
    def data(self):
        pass

    @lazyproperty
    def data_masked(self):
        pass

    @deprecated_positional_kwargs(since='3.0', until='4.0')
    def make_cutout(self, data, masked_array=False):
        pass
