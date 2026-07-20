
import warnings

import astropy.units as u
import numpy as np

from photutils.utils._deprecation import deprecated_positional_kwargs

__all__ = ['ApertureMask']


class ApertureMask:

    def __init__(self, data, bbox):
        self.data = np.asanyarray(data)
        if self.data.shape != bbox.shape:
            msg = 'mask data and bounding box must have the same shape'
            raise ValueError(msg)
        self.bbox = bbox
        self._mask = (self.data == 0)

    def __array__(self, dtype=None, *, copy=None):
        """
        Array representation of the mask data array (e.g., for
        matplotlib).
        """
        return np.array(self.data, dtype=dtype, copy=copy)

    @property
    def shape(self):
        pass

    def get_overlap_slices(self, shape):
        """
        Get slices for the overlapping part of the aperture mask and a
        2D array.

        Parameters
        ----------
        shape : 2-tuple of int
            The shape of the 2D array.

        Returns
        -------
        slices_large : tuple of slices or `None`
            A tuple of slice objects for each axis of the large array,
            such that ``large_array[slices_large]`` extracts the region
            of the large array that overlaps with the small array.
            `None` is returned if there is no overlap of the bounding
            box with the given image shape.

        slices_small : tuple of slices or `None`
            A tuple of slice objects for each axis of the aperture mask
            array such that ``small_array[slices_small]`` extracts the
            region that is inside the large array. `None` is returned if
            there is no overlap of the bounding box with the given image
            shape.
        """
        return self.bbox.get_overlap_slices(shape)

    @deprecated_positional_kwargs(since='3.0', until='4.0')
    def to_image(self, shape, dtype=float):
        pass

    @deprecated_positional_kwargs(since='3.0', until='4.0')
    def cutout(self, data, fill_value=0.0, copy=False):
        pass

    @deprecated_positional_kwargs(since='3.0', until='4.0')
    def multiply(self, data, fill_value=0.0):
        pass

    def _get_overlap_cutouts(self, shape, *, mask=None):
        """
        Get the aperture mask weights, pixel mask, and slice for the
        overlap with the input shape.

        If input, the ``mask`` is included in the output pixel mask
        cutout.

        Parameters
        ----------
        shape : tuple of int
            The shape of data.

        mask : array_like (bool), optional
            A boolean mask with the same shape as ``shape`` where a
            `True` value indicates a masked pixel.

        Returns
        -------
        slices_large : tuple of slices or `None`
            A tuple of slice objects for each axis of the large array
            of given ``shape``, such that ``large_array[slices_large]``
            extracts the region of the large array that overlaps with
            the small array. `None` is returned if there is no overlap
            of the bounding box with the given image shape.

        aper_weights: 2D float `~numpy.ndarray`
            The cutout aperture mask weights for the overlap.

        pixel_mask: 2D bool `~numpy.ndarray`
            The cutout pixel mask for the overlap.

        Notes
        -----
        This method is separate from ``get_values`` to facilitate
        applying the same slices, aper_weights, and pixel_mask to
        multiple associated arrays (e.g., data and error arrays). It is
        used in this way by the `PixelAperture.do_photometry` method.
        """
        if mask is not None and mask.shape != shape:
            msg = 'mask and data must have the same shape'
            raise ValueError(msg)

        slc_large, slc_small = self.get_overlap_slices(shape)
        if slc_large is None:  # no overlap
            return None, None, None

        aper_weights = self.data[slc_small]
        pixel_mask = (aper_weights > 0)  # good pixels

        if mask is not None:
            pixel_mask &= ~mask[slc_large]

        return slc_large, aper_weights, pixel_mask

    @deprecated_positional_kwargs(since='3.0', until='4.0')
    def get_values(self, data, mask=None):
        pass
