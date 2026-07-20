
import numpy as np
from astropy.units import Quantity
from astropy.utils.decorators import deprecated
from scipy.ndimage import zoom

from photutils.utils import ShepardIDWInterpolator
from photutils.utils._repr import make_repr

__all__ = ['BkgIDWInterpolator', 'BkgZoomInterpolator']


class _BkgZoomInterpolator:

    def __init__(self, *, order=3, mode='reflect', cval=0.0, clip=True):
        self.order = order
        self.mode = mode
        self.cval = cval
        self.clip = clip

    def __repr__(self):
        params = ('order', 'mode', 'cval', 'clip')
        return make_repr(self, params)

    def __call__(self, data, **kwargs):
        """
        Resize the 2D mesh array.

        Parameters
        ----------
        data : 2D `~numpy.ndarray`
            The low-resolution 2D mesh array.

        **kwargs : dict
            Additional keyword arguments passed to the interpolator.

        Returns
        -------
        result : 2D `~numpy.ndarray`
            The resized background or background RMS image.

        Notes
        -----
        If ``data`` is an `~astropy.units.Quantity`, units are stripped
        before interpolation. Unit re-assignment is the caller's
        responsibility.
        """
        data = np.asanyarray(data)
        if isinstance(data, Quantity):
            data = data.value
        if np.ptp(data) == 0:
            return np.full(kwargs['shape'], np.min(data),
                           dtype=kwargs['dtype'])

        zoom_factor = kwargs['box_size']
        result = zoom(data, zoom_factor, order=self.order, mode=self.mode,
                      cval=self.cval, grid_mode=True)
        result = result[0:kwargs['shape'][0], 0:kwargs['shape'][1]]

        if self.clip:
            minval = np.min(data)
            maxval = np.max(data)
            np.clip(result, minval, maxval, out=result)  # clip in place

        return result


@deprecated(since='3.0', message=('BkgZoomInterpolator is deprecated and will '
                                  'be removed in version 4.0.'))
class BkgZoomInterpolator(_BkgZoomInterpolator):

    def __init__(self, *, order=3, mode='reflect', cval=0.0, clip=True):
        super().__init__(order=order, mode=mode, cval=cval, clip=clip)


@deprecated(since='3.0', message=('BkgIDWInterpolator is deprecated and will '
                                  'be removed in a version 4.0.'))
class BkgIDWInterpolator:

    def __init__(self, *, leafsize=10, n_neighbors=10, power=1.0,
                 regularization=0.0):
        self.leafsize = leafsize
        self.n_neighbors = n_neighbors
        self.power = power
        self.regularization = regularization

    def __repr__(self):
        params = ('leafsize', 'n_neighbors', 'power', 'regularization')
        return make_repr(self, params)

    def __call__(self, data, **kwargs):
        """
        Resize the 2D mesh array.

        Parameters
        ----------
        data : 2D `~numpy.ndarray`
            The low-resolution 2D mesh array.

        **kwargs : dict
            Additional keyword arguments passed to the interpolator.

        Returns
        -------
        result : 2D `~numpy.ndarray`
            The resized background or background RMS image.

        Notes
        -----
        If ``data`` is an `~astropy.units.Quantity`, units are stripped
        before interpolation. Unit re-assignment is the caller's
        responsibility.
        """
        data = np.asanyarray(data)
        if isinstance(data, Quantity):
            data = data.value
        if np.ptp(data) == 0:
            return np.full(kwargs['shape'], np.min(data),
                           dtype=kwargs['dtype'])

        yxcen = np.column_stack(kwargs['mesh_yxcen'])
        good_idx = np.where(~kwargs['mesh_nan_mask'])
        data = data[good_idx]
        interp_func = ShepardIDWInterpolator(yxcen, data,
                                             leafsize=self.leafsize)

        yi, xi = np.mgrid[0:kwargs['shape'][0], 0:kwargs['shape'][1]]
        yx_indices = np.column_stack((yi.ravel(), xi.ravel()))
        data = interp_func(yx_indices, n_neighbors=self.n_neighbors,
                           power=self.power,
                           regularization=self.regularization)

        return data.reshape(kwargs['shape'])
