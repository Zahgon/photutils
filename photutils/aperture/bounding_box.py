
import math

import numpy as np

from photutils.utils._deprecation import deprecated_positional_kwargs

__all__ = ['BoundingBox']


class BoundingBox:

    def __init__(self, ixmin, ixmax, iymin, iymax):
        for value in (ixmin, ixmax, iymin, iymax):
            if not isinstance(value, (int, np.integer)):
                msg = 'ixmin, ixmax, iymin, and iymax must all be integers'
                raise TypeError(msg)

        if ixmin > ixmax:
            msg = 'ixmin must be <= ixmax'
            raise ValueError(msg)
        if iymin > iymax:
            msg = 'iymin must be <= iymax'
            raise ValueError(msg)

        self.ixmin = ixmin
        self.ixmax = ixmax
        self.iymin = iymin
        self.iymax = iymax

    @classmethod
    def from_float(cls, xmin, xmax, ymin, ymax):
        pass

    def __eq__(self, other):
        if not isinstance(other, BoundingBox):
            msg = 'Can compare BoundingBox only to another BoundingBox.'
            raise TypeError(msg)

        return ((self.ixmin == other.ixmin)
                and (self.ixmax == other.ixmax)
                and (self.iymin == other.iymin)
                and (self.iymax == other.iymax))

    def __or__(self, other):
        return self.union(other)

    def __and__(self, other):
        return self.intersection(other)

    def __repr__(self):
        return (f'{self.__class__.__name__}(ixmin={self.ixmin}, '
                f'ixmax={self.ixmax}, iymin={self.iymin}, '
                f'iymax={self.iymax})')

    @property
    def center(self):
        pass

    @property
    def shape(self):
        pass

    def get_overlap_slices(self, shape):
        """
        Get slices for the overlapping part of the bounding box and a 2D
        array.

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
            A tuple of slice objects for each axis of an array enclosed
            by the bounding box such that ``small_array[slices_small]``
            extracts the region that is inside the large array. `None`
            is returned if there is no overlap of the bounding box with
            the given image shape.
        """
        if len(shape) != 2:
            msg = 'input shape must have 2 elements'
            raise ValueError(msg)

        xmin = self.ixmin
        xmax = self.ixmax
        ymin = self.iymin
        ymax = self.iymax

        if xmin >= shape[1] or ymin >= shape[0] or xmax <= 0 or ymax <= 0:
            return None, None

        slices_large = (slice(max(ymin, 0), min(ymax, shape[0])),
                        slice(max(xmin, 0), min(xmax, shape[1])))
        slices_small = (slice(max(-ymin, 0),
                              min(ymax - ymin, shape[0] - ymin)),
                        slice(max(-xmin, 0),
                              min(xmax - xmin, shape[1] - xmin)))

        return slices_large, slices_small

    @property
    def extent(self):
        pass

    def as_artist(self, **kwargs):
        pass

    def to_aperture(self):
        pass

    @deprecated_positional_kwargs(since='3.0', until='4.0')
    def plot(self, ax=None, origin=(0, 0), **kwargs):
        pass

    def union(self, other):
        pass

    def intersection(self, other):
        pass
