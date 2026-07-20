
import numpy as np
from scipy.spatial import cKDTree

from photutils.utils._deprecation import (deprecated_positional_kwargs,
                                          deprecated_renamed_argument)

__all__ = ['ShepardIDWInterpolator']


class ShepardIDWInterpolator:

    @deprecated_positional_kwargs(since='3.0', until='4.0')
    def __init__(self, coordinates, values, weights=None, leafsize=10):
        coordinates = np.asarray(coordinates)
        if coordinates.ndim == 0:  # scalar coordinate
            coordinates = np.atleast_2d(coordinates)

        if coordinates.ndim == 1:
            coordinates = np.transpose(np.atleast_2d(coordinates))

        if coordinates.ndim > 2:
            coordinates = np.reshape(coordinates, (-1, coordinates.shape[-1]))

        values = np.asanyarray(values).ravel()

        ncoords = coordinates.shape[0]
        if ncoords < 1:
            msg = 'coordinates must have at least one data point'
            raise ValueError(msg)

        if values.shape[0] != ncoords:
            msg = 'The number of values must match the number of coordinates.'
            raise ValueError(msg)

        if weights is not None:
            weights = np.asanyarray(weights).ravel()
            if weights.shape[0] != ncoords:
                msg = ('The number of weights must match the number of '
                       'coordinates.')
                raise ValueError(msg)
            if np.any(weights < 0.0):
                msg = 'All weight values must be non-negative numbers.'
                raise ValueError(msg)

        self.coordinates = coordinates
        self.ncoords = ncoords
        self.coords_ndim = coordinates.shape[1]
        self.values = values
        self.weights = weights
        self.kdtree = cKDTree(coordinates, leafsize=leafsize)

    @deprecated_renamed_argument('reg', 'regularization', '3.0',
                                 until='4.0')
    @deprecated_positional_kwargs(since='3.0', until='4.0')
    def __call__(self, positions, n_neighbors=8, eps=0.0, power=1.0,
                 regularization=0.0, conf_dist=1.0e-12, dtype=float):
        """
        Evaluate the interpolator at the given positions.

        Parameters
        ----------
        positions : float, 1D array_like, or NxM array_like
            Coordinates of the position(s) at which the interpolator
            should be evaluated. In general, it is expected that these
            coordinates are in a form of an NxM-like array where N is
            the number of points and M is dimension of the coordinate
            space. When M=1 (1D space), then the ``positions`` parameter
            may be input as a 1D-like array or, if only one data
            point is available, ``positions`` can be a scalar number
            representing the 1D coordinate of the data point.

            .. note::
                If the dimensionality of the ``positions`` argument is
                larger than 2, e.g., if it is of the form N1 x N2 x N3 x
                ... x Nn x M, then it will be flattened to form an array
                of size NxM where N = N1 * N2 * ... * Nn.

            .. warning::
                The dimensionality of ``positions`` must match the
                dimensionality of the ``coordinates`` used during the
                initialization of the interpolator.

        n_neighbors : int, optional
            The maximum number of nearest neighbors to use during the
            interpolation.

        eps : float, optional
            Set to use approximate nearest neighbors; the kth neighbor
            is guaranteed to be no further than (1 + ``eps``) times the
            distance to the real *k*-th nearest neighbor. See
            `scipy.spatial.cKDTree.query` for further information.

        power : float, optional
            The power of the inverse distance used for the interpolation
            weights. See the Notes section for more details.

        regularization : float, optional
            The regularization parameter. It may be used to control the
            smoothness of the interpolator. See the Notes section for
            more details.

        conf_dist : float, optional
            The confusion distance below which the interpolator should
            use the value of the closest data point instead of
            attempting to interpolate. This is used to avoid
            singularities at the known data points, especially if
            ``regularization`` is 0.0.

        dtype : data-type, optional
            The data type of the output interpolated values. If `None`
            then the type will be inferred from the type of the
            ``values`` parameter used during the initialization of the
            interpolator.

        Returns
        -------
        result : float or `~numpy.ndarray`
            The interpolated value(s). A scalar is returned when a
            single position is provided; otherwise a 1D array is
            returned.
        """
        n_neighbors = int(n_neighbors)
        if n_neighbors < 1:
            msg = 'n_neighbors must be a positive integer'
            raise ValueError(msg)

        if conf_dist is not None and conf_dist <= 0.0:
            conf_dist = None

        positions = np.asanyarray(positions)
        if positions.ndim == 0:
            if self.coords_ndim != 1:
                msg = ('The dimensionality of the input position does '
                       'not match the dimensionality of the coordinates '
                       'used to initialize the interpolator.')
                raise ValueError(msg)
        elif positions.ndim == 1:
            if self.coords_ndim not in (1, positions.shape[-1]):
                msg = ('The input position was provided as a 1D array, '
                       'but its length does not match the dimensionality '
                       'of the coordinates used to initialize the '
                       'interpolator.')
                raise ValueError(msg)
        elif positions.ndim != 2:
            msg = ('The input positions must be an array_like object '
                   'of dimensionality no larger than 2.')
            raise ValueError(msg)

        positions = np.reshape(positions, (-1, self.coords_ndim))
        n_positions = positions.shape[0]

        distances, idx = self.kdtree.query(positions, k=n_neighbors, eps=eps)

        if n_neighbors == 1:
            result = self.values[idx]
            return result.item() if n_positions == 1 else result

        if dtype is None:
            dtype = self.values.dtype

        valid = np.isfinite(distances)

        safe_distances = np.where(valid, distances, 1.0)
        safe_idx = np.where(valid, idx, 0)

        with np.errstate(invalid='ignore', divide='ignore'):
            weights = np.where(valid,
                               1.0 / (safe_distances ** power
                                      + regularization),
                               0.0)

            if self.weights is not None:
                weights *= np.where(valid, self.weights[safe_idx], 0.0)

            neighbor_values = self.values[safe_idx]
            weights_tot = np.sum(weights, axis=1)
            weighted_sum = np.sum(weights * neighbor_values, axis=1)

            interp_values = np.where(weights_tot > 0.0,
                                     weighted_sum / weights_tot,
                                     np.nan).astype(dtype)

        if conf_dist is not None:
            min_dist = distances[:, 0]
            confused = np.isfinite(min_dist) & (min_dist <= conf_dist)
            if np.any(confused):
                interp_values[confused] = self.values[
                    idx[confused, 0]
                ].astype(dtype)

        if n_positions == 1:
            return interp_values[0]

        return interp_values
