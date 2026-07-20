
from collections import defaultdict

import numpy as np
from astropy.utils import lazyproperty
from scipy.cluster.hierarchy import fclusterdata

from photutils.aperture import CircularAperture
from photutils.utils import make_random_cmap
from photutils.utils._deprecation import deprecated_positional_kwargs
from photutils.utils._repr import make_repr

__all__ = ['SourceGrouper', 'SourceGroups']


class SourceGroups:

    def __init__(self, x, y, groups):
        self.x = np.asarray(x)
        self.y = np.asarray(y)
        self.groups = np.asarray(groups)

        if self.x.shape != self.y.shape or self.x.shape != self.groups.shape:
            msg = 'x, y, and groups must have the same shape'
            raise ValueError(msg)

        self.n_sources = len(self.groups)

        unique_groups, counts = np.unique(self.groups, return_counts=True)
        self._unique_groups = unique_groups
        self._group_counts = counts
        self.n_groups = len(unique_groups)

    def __repr__(self):
        params = ['n_sources', 'n_groups']
        return make_repr(self, params, brackets=True)

    def __len__(self):
        """
        Return the number of sources.
        """
        return self.n_sources

    @lazyproperty
    def size_map(self):
        pass

    @lazyproperty
    def sizes(self):
        pass

    @lazyproperty
    def group_centers(self):
        pass

    def get_group_sources(self, group_id):
        pass

    def plot(self, radius, *, ax=None, cmap=None, seed=0, label_groups=False,
             label_kwargs=None, label_offset=(0, 0), **kwargs):
        pass


class SourceGrouper:

    def __init__(self, min_separation):
        self.min_separation = min_separation

    def __repr__(self):
        return make_repr(self, 'min_separation')

    def _compute_groups(self, x, y):
        pass

    @deprecated_positional_kwargs(since='3.0', until='4.0')
    def __call__(self, x, y, return_groups_object=False):
        """
        Group sources into clusters based on a minimum distance
        criteria.

        Parameters
        ----------
        x, y : 1D float `~numpy.ndarray`
            The 1D arrays of the x and y coordinates of the sources.

        return_groups_object : bool, optional
            If `False` (default), return a 1D array of group IDs.
            If `True`, return a `SourceGroups` object containing the
            grouping results along with analysis methods.

        Returns
        -------
        result : `~numpy.ndarray` or `SourceGroups`
            If ``return_groups_object=False`` (default), returns a 1D
            integer array of group IDs for each source, in the same
            order as the input coordinates.

            If ``return_groups_object=True``, returns a `SourceGroups`
            object containing the grouping results. The object provides:

            - ``groups`` : array of group IDs for each source
            - ``n_sources`` : total number of sources
            - ``n_groups`` : total number of groups
            - ``sizes`` : group size for each source
            - ``group_centers`` : centroid coordinates for each group
            - ``get_group_sources(group_id)`` : retrieve sources in a
              specific group
            - ``plot()`` : visualize the grouping with color-coded
              apertures

        Examples
        --------
        Get group IDs as an array (default behavior):

        >>> from photutils.psf import SourceGrouper
        >>> import numpy as np
        >>> x = np.array([10, 15, 50])
        >>> y = np.array([20, 25, 60])
        >>> grouper = SourceGrouper(min_separation=10)
        >>> group_ids = grouper(x, y)
        >>> print(group_ids)
        [1 1 2]

        Get a SourceGroups object with additional analysis methods:

        >>> groups = grouper(x, y, return_groups_object=True)
        >>> print(groups.n_groups)
        2
        >>> print(groups.groups)
        [1 1 2]
        """
        groups = self._compute_groups(x, y)
        if return_groups_object:
            return SourceGroups(x, y, groups)
        return groups
