
import numpy as np
from astropy.nddata import extract_array, overlap_slices
from astropy.utils import lazyproperty

from photutils.utils._deprecation import deprecated_positional_kwargs

__all__ = ['CutoutImage']


class CutoutImage:

    @deprecated_positional_kwargs(since='3.0', until='4.0')
    def __init__(self, data, position, shape, mode='trim', fill_value=np.nan,
                 copy=False):
        self.position = position
        self.input_shape = tuple(shape)
        self.mode = mode
        self.fill_value = fill_value
        self.copy = copy

        data = np.asanyarray(data)
        self._overlap_slices = overlap_slices(data.shape, shape, position,
                                              mode=mode)
        self.data = self._make_cutout(data)
        self.shape = self.data.shape

    def _make_cutout(self, data):
        """
        Create the cutout data array.

        Parameters
        ----------
        data : `~numpy.ndarray`
            The 2D data array from which to extract the cutout array.

        Returns
        -------
        cutout_data : `~numpy.ndarray`
            The 2D cutout data array.
        """
        cutout_data = extract_array(data, self.input_shape, self.position,
                                    mode=self.mode, fill_value=self.fill_value,
                                    return_position=False)
        if self.copy:
            cutout_data = np.copy(cutout_data)
        return cutout_data

    def __array__(self, dtype=None, *, copy=None):
        """
        Array representation of the cutout data array (e.g., for
        matplotlib).

        Parameters
        ----------
        dtype : `~numpy.dtype`, optional
            The data type of the output array. If `None`, then the
            data type of the cutout data array is used.

        copy : bool, optional
            If `True`, then a copy of the underlying data array
            is returned.
        """
        return np.array(self.data, dtype=dtype, copy=copy)

    def __str__(self):
        cls_name = f'<{self.__class__.__module__}.{self.__class__.__name__}>'
        props = f'Shape: {self.data.shape}'
        return f'{cls_name}\n' + props

    def __repr__(self):
        return (f'{self.__class__.__name__}(position={self.position}, '
                f'shape={self.shape})')

    @lazyproperty
    def slices_original(self):
        pass

    @lazyproperty
    def slices_cutout(self):
        pass

    def _calc_bbox(self, slices):
        pass

    @lazyproperty
    def bbox_original(self):
        pass

    @lazyproperty
    def bbox_cutout(self):
        pass

    def _calc_xyorigin(self, slices):
        pass

    @lazyproperty
    def xyorigin(self):
        pass


def _make_cutouts(data, xpos, ypos, cutout_shape, *, fill_value=0.0):
    """
    Make 2D cutouts from a data array at the given positions.

    Positions are rounded to the nearest integer pixel. Pixels that fall
    outside the image boundary are filled with ``fill_value``.

    Parameters
    ----------
    data : 2D `~numpy.ndarray`
        The 2D image array.

    xpos : 1D `~numpy.ndarray`
        The x pixel positions of the cutout centers.

    ypos : 1D `~numpy.ndarray`
        The y pixel positions of the cutout centers.

    cutout_shape : tuple of int
        The ``(ny, nx)`` shape of each cutout.

    fill_value : float, optional
        The value used to fill pixels that fall outside the image
        boundary. The default is 0.0. Use ``np.nan`` when out-of-bounds
        pixels must be distinguishable from real data (e.g., for
        sigma-clipped statistics on partial cutouts).

    Returns
    -------
    cutouts : 3D `~numpy.ndarray`
        A 3D array of shape ``(n_sources, ny, nx)`` containing the
        cutout data.

    overlap_mask : 3D `~numpy.ndarray` of bool
        A boolean array with the same shape as ``cutouts``. `True`
        indicates a pixel that came from ``data``. `False` indicates
        a pixel that was filled with ``fill_value`` because it fell
        outside the image boundary.

        Per-source overlap status can be derived from this mask:

        * Fully inside the image: ``overlap_mask[i].all()``
        * No overlap (entirely outside): ``~overlap_mask[i].any()``
        * Partial overlap: neither of the above
    """
    data = np.asarray(data)
    if data.ndim != 2:
        msg = 'data must be a 2D array'
        raise ValueError(msg)

    xpos = np.atleast_1d(np.asarray(xpos))
    ypos = np.atleast_1d(np.asarray(ypos))
    if xpos.ndim != 1 or ypos.ndim != 1:
        msg = 'xpos and ypos must be 1D arrays'
        raise ValueError(msg)

    if len(xpos) != len(ypos):
        msg = 'xpos and ypos must have the same length'
        raise ValueError(msg)

    if len(cutout_shape) != 2:
        msg = 'cutout_shape must have exactly 2 elements'
        raise ValueError(msg)

    ky, kx = cutout_shape
    hy, hx = ky // 2, kx // 2

    yc = np.round(ypos).astype(int)
    xc = np.round(xpos).astype(int)

    dy = np.arange(ky) - hy
    dx = np.arange(kx) - hx
    y_idx = yc[:, np.newaxis, np.newaxis] + dy[np.newaxis, :, np.newaxis]
    x_idx = xc[:, np.newaxis, np.newaxis] + dx[np.newaxis, np.newaxis, :]

    overlap_mask = ((y_idx >= 0) & (y_idx < data.shape[0])
                    & (x_idx >= 0) & (x_idx < data.shape[1]))

    y_safe = np.clip(y_idx, 0, data.shape[0] - 1)
    x_safe = np.clip(x_idx, 0, data.shape[1] - 1)

    cutouts = np.where(overlap_mask, data[y_safe, x_safe], fill_value)

    return cutouts, overlap_mask
