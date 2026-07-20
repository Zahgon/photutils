
import warnings

import astropy.units as u
import numpy as np
from astropy.utils.exceptions import AstropyUserWarning
from scipy.ndimage import binary_dilation

from photutils.utils._coords import apply_separation
from photutils.utils._deprecation import (deprecated_getattr,
                                          deprecated_renamed_argument)
from photutils.utils._parameters import (SigmaClipSentinelDefault,
                                         create_default_sigmaclip)
from photutils.utils._progress_bars import add_progress_bar
from photutils.utils._repr import make_repr
from photutils.utils.footprints import circular_footprint

__all__ = ['ImageDepth']

__doctest_requires__ = {('ImageDepth', 'ImageDepth.*'): ['skimage']}

SIGMA_CLIP = SigmaClipSentinelDefault(sigma=3.0, maxiters=10)

_DEPRECATED_ATTRIBUTES = {
    'nsigma': 'n_sigma',
    'napers': 'n_apertures',
    'niters': 'n_iters',
    'napers_used': 'n_apertures_used',
}


class ImageDepth:

    @deprecated_renamed_argument('nsigma', 'n_sigma', '3.0', until='4.0')
    @deprecated_renamed_argument('napers', 'n_apertures', '3.0', until='4.0')
    @deprecated_renamed_argument('niters', 'n_iters', '3.0', until='4.0')
    def __init__(self, aper_radius, *, n_sigma=5.0, mask_pad=0,
                 n_apertures=1000, n_iters=10, overlap=True,
                 overlap_maxiters=100, seed=None, zeropoint=0.0,
                 sigma_clip=SIGMA_CLIP, progress_bar=True):

        if aper_radius <= 0:
            msg = 'aper_radius must be > 0'
            raise ValueError(msg)
        if mask_pad < 0:
            msg = 'mask_pad must be >= 0'
            raise ValueError(msg)

        self.aper_radius = aper_radius
        self.n_sigma = n_sigma
        self.mask_pad = mask_pad
        self.n_apertures = n_apertures
        self.n_iters = n_iters
        self.overlap = overlap
        self.overlap_maxiters = overlap_maxiters
        self.seed = seed
        self.zeropoint = zeropoint
        if sigma_clip is SIGMA_CLIP:
            sigma_clip = create_default_sigmaclip(sigma=SIGMA_CLIP.sigma,
                                                  maxiters=SIGMA_CLIP.maxiters)
        if sigma_clip is not None and not callable(sigma_clip):
            msg = 'sigma_clip must be a callable (e.g., SigmaClip) or None'
            raise TypeError(msg)
        self.sigma_clip = sigma_clip
        self.progress_bar = progress_bar

        self.rng = np.random.default_rng(self.seed)
        self.dilate_radius = int(np.ceil(self.aper_radius + self.mask_pad))
        self.dilate_footprint = circular_footprint(radius=self.dilate_radius)

        self.apertures = []
        self.n_apertures_used = np.array([])
        self.fluxes = []
        self.flux_limits = np.array([])
        self.mag_limits = np.array([])

    def __repr__(self):
        params = ('aper_radius', 'n_sigma', 'mask_pad', 'n_apertures',
                  'n_iters', 'overlap', 'overlap_maxiters', 'seed',
                  'zeropoint', 'sigma_clip', 'progress_bar')
        return make_repr(self, params)

    def __getattr__(self, name):
        return deprecated_getattr(self, name,
                                  _DEPRECATED_ATTRIBUTES,
                                  since='3.0', until='4.0')

    def __call__(self, data, mask):
        """
        Calculate the limiting flux and magnitude of an image.

        Parameters
        ----------
        data : 2D `~numpy.ndarray`
            The 2D array, which should be in flux units (not surface
            brightness units).

        mask : 2D bool `~numpy.ndarray`
            A 2D mask array with the same shape as ``data`` where
            a `True` value indicates the corresponding element of
            ``data`` is masked. The input array should mask both sources
            (e.g., from a segmentation image) and regions without image
            coverage. If `None`, then the entire image will be used.

        Returns
        -------
        flux_limit, mag_limit : float
            The flux and magnitude limits. The flux limit is returned in
            the same units as the input ``data``. The magnitude limit is
            calculated from the flux limit and the input ``zeropoint``.
        """
        from photutils.aperture import CircularAperture

        if mask is None or not np.any(mask):
            all_xycoords = self._make_all_coords_no_mask(data.shape)
        else:
            all_xycoords = self._make_all_coords(mask)

        if len(all_xycoords) == 0:
            msg = ('There are no unmasked pixel values (including the '
                   'masked image borders).')
            raise ValueError(msg)

        n_apertures = self.n_apertures
        if not self.overlap:
            n_apertures2 = 1.5 * self.n_apertures
            n_apertures = int(min(n_apertures2,
                                  0.1 * len(all_xycoords)))

        iter_range = range(self.n_iters)
        if self.progress_bar:
            desc = 'Image Depths'
            iter_range = add_progress_bar(iter_range,
                                          desc=desc)

        flux_limits = []
        apertures = []
        for _ in iter_range:
            if self.overlap:
                xycoords = self._make_coords(all_xycoords, n_apertures)
            else:
                xycoords = self._make_coords(all_xycoords,
                                             n_apertures * 10)
                min_separation = self.aper_radius * 2.0
                xycoords = apply_separation(xycoords, min_separation)
                xycoords = xycoords[0:self.n_apertures]

            apers = CircularAperture(xycoords, r=self.aper_radius)
            apertures.append(apers)
            fluxes, _ = apers.do_photometry(data)
            if self.sigma_clip is not None:
                fluxes = self.sigma_clip(fluxes, masked=False)  # ndarray
            self.fluxes.append(fluxes)
            flux_limits.append(self.n_sigma * np.std(fluxes))

        self.apertures = apertures
        n_apertures_used = np.array([len(apers) for apers in apertures])
        self.n_apertures_used = n_apertures_used
        if np.any(n_apertures_used < self.n_apertures):
            msg = (f'Unable to generate {self.n_apertures} '
                   'non-overlapping apertures in unmasked regions. '
                   'The number of apertures used was less than '
                   f'{self.n_apertures} (see the '
                   '"n_apertures_used" ImageDepth object attribute). '
                   'To fix this, decrease the number of apertures '
                   'and/or aperture size, or increase '
                   '`overlap_maxiters`. Alternatively, you may set '
                   'overlap=True')
            warnings.warn(msg, AstropyUserWarning)

        if isinstance(flux_limits[0], u.Quantity):
            units = True
            self.flux_limits = u.Quantity(flux_limits)
        else:
            units = False
            self.flux_limits = np.array(flux_limits)
        flux_limit = np.mean(self.flux_limits)
        if np.any(self.flux_limits == 0):
            msg = ('One or more flux_limit values was zero. This is '
                   'likely due to constant image values. Check the '
                   'input mask.')
            warnings.warn(msg, AstropyUserWarning)

        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)
            flux_limits = self.flux_limits
            flux_limit_ = flux_limit
            if units:
                flux_limits = flux_limits.value
                flux_limit_ = flux_limit.value
            self.mag_limits = -2.5 * np.log10(flux_limits) + self.zeropoint
            mag_limit = -2.5 * np.log10(flux_limit_) + self.zeropoint

        return flux_limit, mag_limit

    @staticmethod
    def _find_slice_axis(data, axis):
        pass

    def _find_slices(self, data):
        pass

    def _mask_border(self, mask):
        pass

    def _dilate_mask(self, mask):
        pass

    def _make_all_coords_no_mask(self, shape):
        pass

    def _make_all_coords(self, mask):
        pass

    def _make_coords(self, xycoords, napers):
        pass
