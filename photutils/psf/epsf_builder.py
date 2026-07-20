
import copy
import inspect
import warnings
from dataclasses import dataclass

import numpy as np
from astropy.modeling.fitting import TRFLSQFitter
from astropy.nddata import NoOverlapError, PartialOverlapError, overlap_slices
from astropy.stats import SigmaClip
from astropy.utils.decorators import deprecated
from astropy.utils.exceptions import (AstropyDeprecationWarning,
                                      AstropyUserWarning)
from scipy.ndimage import convolve

from photutils.centroids import centroid_com
from photutils.psf.epsf_stars import EPSFStar, EPSFStars, LinkedEPSFStar
from photutils.psf.image_models import ImagePSF
from photutils.psf.utils import _interpolate_missing_data
from photutils.utils._parameters import (SigmaClipSentinelDefault, as_pair,
                                         create_default_sigmaclip)
from photutils.utils._progress_bars import add_progress_bar
from photutils.utils._round import round_half_away
from photutils.utils._stats import nanmedian

__all__ = ['EPSFBuildResult', 'EPSFBuilder', 'EPSFFitter']

SIGMA_CLIP = SigmaClipSentinelDefault(sigma=3.0, maxiters=10)


class _SmoothingKernel:

    QUARTIC_KERNEL = np.array([
        [+0.041632, -0.080816, 0.078368, -0.080816, +0.041632],
        [-0.080816, -0.019592, 0.200816, -0.019592, -0.080816],
        [+0.078368, +0.200816, 0.441632, +0.200816, +0.078368],
        [-0.080816, -0.019592, 0.200816, -0.019592, -0.080816],
        [+0.041632, -0.080816, 0.078368, -0.080816, +0.041632]])

    QUADRATIC_KERNEL = np.array([
        [-0.07428311, 0.01142786, 0.03999952, 0.01142786, -0.07428311],
        [+0.01142786, 0.09714283, 0.12571449, 0.09714283, +0.01142786],
        [+0.03999952, 0.12571449, 0.15428215, 0.12571449, +0.03999952],
        [+0.01142786, 0.09714283, 0.12571449, 0.09714283, +0.01142786],
        [-0.07428311, 0.01142786, 0.03999952, 0.01142786, -0.07428311]])

    @classmethod
    def get_kernel(cls, kernel_type):
        pass

    @staticmethod
    def apply_smoothing(data, kernel_type):
        pass


class _EPSFValidator:

    @staticmethod
    def validate_oversampling(oversampling, *, context=''):
        """
        Validate oversampling parameters.

        Parameters
        ----------
        oversampling : int or tuple
            The oversampling factor(s).

        context : str, optional
            Additional context for error messages.

        Raises
        ------
        ValueError
            If oversampling is invalid.
        """
        if oversampling is None:
            msg = "'oversampling' must be specified"
            raise ValueError(msg)

        try:
            oversampling = as_pair('oversampling', oversampling,
                                   lower_bound=(0, 0))
        except (TypeError, ValueError) as e:
            msg = f'Invalid oversampling parameter - {e}'
            if context:
                msg = f'{context}: {msg}'
            raise ValueError(msg) from None

        return oversampling

    @staticmethod
    def validate_shape_compatibility(stars, oversampling, *, shape=None):
        pass

    @staticmethod
    def validate_stars(stars, *, context=''):
        pass

    @staticmethod
    def validate_center_accuracy(center_accuracy):
        """
        Validate center accuracy parameter.

        Parameters
        ----------
        center_accuracy : float
            The center accuracy threshold.

        Raises
        ------
        ValueError
            If center accuracy is invalid.
        """
        if not isinstance(center_accuracy, (int, float)):
            msg = (f'center_accuracy must be a number, got '
                   f'{type(center_accuracy)}')
            raise TypeError(msg)

        if center_accuracy <= 0.0:
            msg = ('center_accuracy must be positive, got '
                   f'{center_accuracy}. Typical values are 1e-3 to 1e-4.')
            raise ValueError(msg)

        if center_accuracy > 1.0:
            msg = (f'center_accuracy {center_accuracy} seems unusually large. '
                   'Values > 1.0 may prevent convergence. '
                   'Typical values are 1e-3 to 1e-4.')
            warnings.warn(msg, AstropyUserWarning)

    @staticmethod
    def validate_maxiters(maxiters):
        """
        Validate maximum iterations parameter.

        Parameters
        ----------
        maxiters : int
            The maximum number of iterations.

        Raises
        ------
        ValueError, TypeError
            If maxiters is invalid.
        """
        if not isinstance(maxiters, int):
            msg = f'maxiters must be an integer, got {type(maxiters)}'
            raise TypeError(msg)

        if maxiters <= 0:
            msg = 'maxiters must be a positive number'
            raise ValueError(msg)

        maxiters_warn_threshold = 100
        if maxiters > maxiters_warn_threshold:
            msg = (f'maxiters {maxiters} seems unusually large. '
                   f'Values > {maxiters_warn_threshold} may indicate '
                   'convergence issues. Consider checking your data and '
                   'parameters.')
            warnings.warn(msg, AstropyUserWarning)


class _CoordinateTransformer:

    def __init__(self, oversampling):
        self.oversampling = np.asarray(oversampling)

    def star_to_epsf_coords(self, star_x, star_y, epsf_origin):
        pass

    def compute_epsf_shape(self, star_shapes):
        pass

    def compute_epsf_origin(self, epsf_shape):
        pass

    def oversampled_to_undersampled(self, x, y):
        pass

    def undersampled_to_oversampled(self, x, y):
        pass


class _ProgressReporter:

    def __init__(self, enabled, maxiters):
        """
        Initialize a _ProgressReporter.

        Parameters
        ----------
        enabled : bool
            Whether progress reporting is enabled.

        maxiters : int
            The maximum number of iterations.
        """
        self.enabled = enabled
        self.maxiters = maxiters
        self._pbar = None

    def setup(self):
        pass

    def update(self):
        """
        Update the progress bar by one iteration.

        Only updates if progress reporting is enabled and progress bar
        is initialized.
        """
        if self._pbar is not None:
            self._pbar.update()

    def write_convergence_message(self, iteration):
        pass

    def close(self):
        pass


@dataclass
class EPSFBuildResult:

    epsf: 'ImagePSF'
    fitted_stars: 'EPSFStars'
    iterations: int
    converged: bool
    final_center_accuracy: float
    n_excluded_stars: int
    excluded_star_indices: list

    def __iter__(self):
        """
        Allow tuple unpacking for backward compatibility.

        Returns
        -------
        iterator
            An iterator that yields (epsf, fitted_stars) for
            compatibility with existing code that expects a 2-tuple.
        """
        return iter((self.epsf, self.fitted_stars))

    def __getitem__(self, index):
        """
        Allow indexing for backward compatibility.

        Parameters
        ----------
        index : int
            Index to access (0 for epsf, 1 for fitted_stars).

        Returns
        -------
        value
            The ePSF (index 0) or fitted stars (index 1).
        """
        if index == 0:
            return self.epsf
        if index == 1:
            return self.fitted_stars

        msg = 'EPSFBuildResult index must be 0 (epsf) or 1 (fitted_stars)'
        raise IndexError(msg)


@deprecated(since='3.0',
            message=('EPSFFitter is deprecated and will be removed in a '
                     'version 4.0. Use EPSFBuilder with the fitter, '
                     'fit_shape, and fitter_maxiters parameters instead.'))
class EPSFFitter:

    def __init__(self, *, fitter=None, fit_boxsize=5, **fitter_kwargs):

        if fitter is None:
            fitter = TRFLSQFitter()
        self.fitter = fitter
        self.fitter_has_fit_info = hasattr(self.fitter, 'fit_info')
        if fit_boxsize is not None:
            self.fit_boxsize = as_pair('fit_boxsize', fit_boxsize,
                                       lower_bound=(3, 1), check_odd=True)
        else:
            self.fit_boxsize = None

        remove_kwargs = {'x', 'y', 'z', 'weights'}
        self.fitter_kwargs = {
            k: v for k, v in fitter_kwargs.items()
            if k not in remove_kwargs
        }

    def __call__(self, epsf, stars):
        """
        Fit an ePSF model to stars.

        Parameters
        ----------
        epsf : `ImagePSF`
            An ePSF model to be fitted to the stars.

        stars : `EPSFStars` object
            The stars to be fit. The center coordinates for each star
            should be as close as possible to actual centers. For stars
            than contain weights, a weighted fit of the ePSF to the star
            will be performed.

        Returns
        -------
        fitted_stars : `EPSFStars` object
            The fitted stars. The ePSF-fitted center position and flux
            are stored in the ``center`` (and ``cutout_center``) and
            ``flux`` attributes.
        """
        if len(stars) == 0:
            return stars

        if not isinstance(epsf, ImagePSF):
            msg = 'The input epsf must be an ImagePSF'
            raise TypeError(msg)

        fitted_stars = []
        for star in stars:
            if isinstance(star, EPSFStar):
                if star._excluded_from_fit:
                    fitted_star = star
                else:
                    fitted_star = self._fit_star(epsf, star, self.fitter,
                                                 self.fitter_kwargs,
                                                 self.fitter_has_fit_info,
                                                 self.fit_boxsize)

            elif isinstance(star, LinkedEPSFStar):
                fitted_star = []
                for linked_star in star:
                    if linked_star._excluded_from_fit:
                        fitted_star.append(linked_star)
                    else:
                        fitted_star.append(
                            self._fit_star(epsf, linked_star, self.fitter,
                                           self.fitter_kwargs,
                                           self.fitter_has_fit_info,
                                           self.fit_boxsize))

                fitted_star = LinkedEPSFStar(fitted_star)
                fitted_star.constrain_centers()

            else:
                msg = ('stars must contain only EPSFStar and/or '
                       'LinkedEPSFStar objects')
                raise TypeError(msg)

            fitted_stars.append(fitted_star)

        return EPSFStars(fitted_stars)

    def _fit_star(self, epsf, star, fitter, fitter_kwargs,
                  fitter_has_fit_info, fit_boxsize):
        pass


class EPSFBuilder:

    def __init__(self, *, oversampling=4, shape=None,
                 smoothing_kernel='quartic', sigma_clip=SIGMA_CLIP,
                 recentering_func=centroid_com, recentering_boxsize=(5, 5),
                 recentering_maxiters=20, center_accuracy=1.0e-3,
                 fitter=None, fit_shape=5, fitter_maxiters=100,
                 maxiters=10, progress_bar=True):

        self.oversampling = _EPSFValidator.validate_oversampling(
            oversampling, context='EPSFBuilder initialization')

        self.coord_transformer = _CoordinateTransformer(self.oversampling)

        if shape is not None:
            self.shape = as_pair('shape', shape, lower_bound=(0, 0))
        else:
            self.shape = shape

        self.recentering_func = recentering_func
        self.recentering_maxiters = recentering_maxiters
        self.recentering_boxsize = as_pair('recentering_boxsize',
                                           recentering_boxsize,
                                           lower_bound=(3, 1), check_odd=True)
        self.smoothing_kernel = smoothing_kernel

        if isinstance(fitter, EPSFFitter):
            msg = ('Passing an EPSFFitter instance to EPSFBuilder is '
                   'deprecated. Use the fitter, fit_shape, and '
                   'fitter_maxiters parameters instead.')
            warnings.warn(msg, AstropyDeprecationWarning)
            self.fitter = fitter.fitter
            self.fit_shape = fitter.fit_boxsize
            self.fitter_maxiters = None
            self._fitter_kwargs = fitter.fitter_kwargs
        else:
            if fitter is None:
                fitter = TRFLSQFitter()
            if not callable(fitter):
                msg = 'fitter must be a callable astropy Fitter instance'
                raise TypeError(msg)
            self.fitter = fitter

            if fit_shape is not None:
                self.fit_shape = as_pair('fit_shape', fit_shape,
                                         lower_bound=(3, 1),
                                         check_odd=True)
            else:
                self.fit_shape = None

            self.fitter_maxiters = self._validate_fitter_maxiters(
                fitter_maxiters)

            self._fitter_kwargs = {}
            if self.fitter_maxiters is not None:
                self._fitter_kwargs['maxiter'] = self.fitter_maxiters

        self._fitter_has_fit_info = hasattr(self.fitter, 'fit_info')

        _EPSFValidator.validate_center_accuracy(center_accuracy)
        self.center_accuracy_sq = center_accuracy**2

        _EPSFValidator.validate_maxiters(maxiters)
        self.maxiters = maxiters

        self.progress_bar = progress_bar

        if sigma_clip is SIGMA_CLIP:
            sigma_clip = create_default_sigmaclip(sigma=SIGMA_CLIP.sigma,
                                                  maxiters=SIGMA_CLIP.maxiters)
        if not isinstance(sigma_clip, SigmaClip):
            msg = 'sigma_clip must be an astropy.stats.SigmaClip instance'
            raise TypeError(msg)
        self._sigma_clip = sigma_clip

        self._epsf = []

    def __call__(self, stars):
        """
        Build an ePSF from input stars.

        Parameters
        ----------
        stars : `EPSFStars`
            The stars used to build the ePSF.

        Returns
        -------
        result : `EPSFBuildResult`
            The result of the ePSF building process.
        """
        return self.build_epsf(stars)

    def _validate_fitter_maxiters(self, fitter_maxiters):
        """
        Validate the ``fitter_maxiters`` parameter.

        Parameters
        ----------
        fitter_maxiters : int
            Maximum number of fitter iterations to validate.

        Returns
        -------
        fitter_maxiters : int or `None`
            The validated value, or `None` if the fitter does not
            support the ``maxiter`` parameter.
        """
        spec = inspect.signature(self.fitter.__call__)
        has_maxiter = ('maxiter' in spec.parameters
                       or any(p.kind == inspect.Parameter.VAR_KEYWORD
                              for p in spec.parameters.values()))
        if not has_maxiter:
            msg = ("'fitter_maxiters' will be ignored because "
                   'it is not accepted by the input fitter')
            warnings.warn(msg, AstropyUserWarning)
            return None
        return fitter_maxiters

    def _create_initial_epsf(self, stars):
        pass

    def _resample_residual(self, star, epsf, *, out_image=None):
        pass

    def _resample_residuals(self, stars, epsf):
        pass

    def _smooth_epsf(self, epsf_data):
        pass

    def _normalize_epsf(self, epsf_data):
        pass

    def _recenter_epsf(self, epsf, *, centroid_func=None, box_size=None,
                       maxiters=None, center_accuracy=None):
        pass

    def _build_epsf_step(self, stars, *, epsf=None):
        pass

    def _check_convergence(self, stars, centers, fit_failed):
        pass

    def _fit_stars(self, epsf, stars):
        pass

    def _fit_star(self, epsf, star):
        pass

    def _process_iteration(self, stars, epsf, iter_num):
        pass

    def _finalize_build(self, epsf, stars, progress_reporter, iter_num,
                        converged, final_center_accuracy,
                        excluded_star_indices):
        pass

    def build_epsf(self, stars, *, epsf=None):
        pass
