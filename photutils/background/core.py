
import abc
import warnings

import numpy as np
from astropy.stats import SigmaClip, biweight_location, biweight_scale, mad_std

from photutils.utils._deprecation import deprecated_positional_kwargs
from photutils.utils._parameters import (SigmaClipSentinelDefault,
                                         create_default_sigmaclip)
from photutils.utils._repr import make_repr
from photutils.utils._stats import nanmean, nanmedian, nanstd

__all__ = [
    'BackgroundBase',
    'BackgroundRMSBase',
    'BiweightLocationBackground',
    'BiweightScaleBackgroundRMS',
    'MADStdBackgroundRMS',
    'MMMBackground',
    'MeanBackground',
    'MedianBackground',
    'ModeEstimatorBackground',
    'SExtractorBackground',
    'StdBackgroundRMS',
]


SIGMA_CLIP = SigmaClipSentinelDefault(sigma=3.0, maxiters=10)

_SIGMA_CLIP_PARAM_DOC = (
    'sigma_clip : `astropy.stats.SigmaClip` or `None`, optional\n'
    '    A `~astropy.stats.SigmaClip` object that defines the sigma\n'
    '    clipping parameters. If `None` then no sigma clipping will be\n'
    '    performed.'
)


def _insert_sigma_clip_doc(cls):
    pass


def _validate_sigma_clip(sigma_clip):
    """
    Validate and generate the ``sigma_clip`` parameter.

    If ``sigma_clip`` is the sentinel default, a fresh
    `~astropy.stats.SigmaClip` instance is created from its stored
    parameters. `None` is accepted (meaning no sigma clipping). Any
    other value must be a `~astropy.stats.SigmaClip` instance.

    Parameters
    ----------
    sigma_clip : `~photutils.utils._parameters.SigmaClipSentinelDefault`,\
            `~astropy.stats.SigmaClip`, or `None`
        The value supplied to a base-class ``__init__``.

    Returns
    -------
    sigma_clip : `~astropy.stats.SigmaClip` or `None`
        A concrete `~astropy.stats.SigmaClip` instance, or `None`.
    """
    if sigma_clip is SIGMA_CLIP:
        return create_default_sigmaclip(sigma=SIGMA_CLIP.sigma,
                                        maxiters=SIGMA_CLIP.maxiters)
    if not isinstance(sigma_clip, SigmaClip) and sigma_clip is not None:
        msg = 'sigma_clip must be an astropy SigmaClip instance or None'
        raise TypeError(msg)

    return sigma_clip


def _prepare_data(sigma_clip, data, axis):
    pass


def _apply_masked(result, masked):
    pass


class _BackgroundCommonBase:

    @deprecated_positional_kwargs(since='3.0', until='4.0')
    def __init__(self, sigma_clip=SIGMA_CLIP):
        self.sigma_clip = _validate_sigma_clip(sigma_clip)

    def __repr__(self):
        return make_repr(self, ('sigma_clip',))


class BackgroundBase(_BackgroundCommonBase, abc.ABC):

    @deprecated_positional_kwargs(since='3.0', until='4.0')
    def __call__(self, data, axis=None, masked=False):
        return self.calc_background(data, axis=axis, masked=masked)

    @abc.abstractmethod
    def calc_background(self, data, *, axis=None, masked=False):
        """
        Calculate the background value.

        Parameters
        ----------
        data : array_like or `~numpy.ma.MaskedArray`
            The array for which to calculate the background value.

        axis : int or `None`, optional
            The array axis along which the background is calculated. If
            `None`, then the entire array is used.

        masked : bool, optional
            If `True`, then a `~numpy.ma.MaskedArray` is returned. If
            `False`, then a `~numpy.ndarray` is returned, where masked
            values have a value of NaN. The default is `False`.

        Returns
        -------
        result : float, `~numpy.ndarray`, or `~numpy.ma.MaskedArray`
            The calculated background value. If ``masked`` is
            `False`, then a `~numpy.ndarray` is returned, otherwise a
            `~numpy.ma.MaskedArray` is returned. A scalar result is
            always returned as a float.
        """


class BackgroundRMSBase(_BackgroundCommonBase, abc.ABC):

    @deprecated_positional_kwargs(since='3.0', until='4.0')
    def __call__(self, data, axis=None, masked=False):
        return self.calc_background_rms(data, axis=axis, masked=masked)

    @abc.abstractmethod
    def calc_background_rms(self, data, *, axis=None, masked=False):
        """
        Calculate the background RMS value.

        Parameters
        ----------
        data : array_like or `~numpy.ma.MaskedArray`
            The array for which to calculate the background RMS value.

        axis : int or `None`, optional
            The array axis along which the background RMS is calculated.
            If `None`, then the entire array is used.

        masked : bool, optional
            If `True`, then a `~numpy.ma.MaskedArray` is returned. If
            `False`, then a `~numpy.ndarray` is returned, where masked
            values have a value of NaN. The default is `False`.

        Returns
        -------
        result : float, `~numpy.ndarray`, or `~numpy.ma.MaskedArray`
            The calculated background RMS value. If ``masked`` is
            `False`, then a `~numpy.ndarray` is returned, otherwise a
            `~numpy.ma.MaskedArray` is returned. A scalar result is
            always returned as a float.
        """


@_insert_sigma_clip_doc
class MeanBackground(BackgroundBase):

    @deprecated_positional_kwargs(since='3.0', until='4.0')
    def calc_background(self, data, axis=None, masked=False):
        pass


@_insert_sigma_clip_doc
class MedianBackground(BackgroundBase):

    @deprecated_positional_kwargs(since='3.0', until='4.0')
    def calc_background(self, data, axis=None, masked=False):
        pass


@_insert_sigma_clip_doc
class ModeEstimatorBackground(BackgroundBase):

    @deprecated_positional_kwargs(since='3.0', until='4.0')
    def __init__(self, median_factor=3.0, mean_factor=2.0,
                 sigma_clip=SIGMA_CLIP):
        super().__init__(sigma_clip=sigma_clip)
        self.median_factor = median_factor
        self.mean_factor = mean_factor

    def __repr__(self):
        params = ('median_factor', 'mean_factor', 'sigma_clip')
        return make_repr(self, params)

    @deprecated_positional_kwargs(since='3.0', until='4.0')
    def calc_background(self, data, axis=None, masked=False):
        pass


@_insert_sigma_clip_doc
class MMMBackground(ModeEstimatorBackground):

    @deprecated_positional_kwargs(since='3.0', until='4.0')
    def __init__(self, sigma_clip=SIGMA_CLIP):
        super().__init__(median_factor=3.0, mean_factor=2.0,
                         sigma_clip=sigma_clip)


@_insert_sigma_clip_doc
class SExtractorBackground(BackgroundBase):

    @deprecated_positional_kwargs(since='3.0', until='4.0')
    def calc_background(self, data, axis=None, masked=False):
        pass


@_insert_sigma_clip_doc
class BiweightLocationBackground(BackgroundBase):

    @deprecated_positional_kwargs(since='3.0', until='4.0')
    def __init__(self, c=6.0, M=None, sigma_clip=SIGMA_CLIP):
        super().__init__(sigma_clip=sigma_clip)
        self.c = c
        self.M = M

    def __repr__(self):
        params = ('c', 'M', 'sigma_clip')
        return make_repr(self, params)

    @deprecated_positional_kwargs(since='3.0', until='4.0')
    def calc_background(self, data, axis=None, masked=False):
        pass


@_insert_sigma_clip_doc
class StdBackgroundRMS(BackgroundRMSBase):

    @deprecated_positional_kwargs(since='3.0', until='4.0')
    def calc_background_rms(self, data, axis=None, masked=False):
        pass


@_insert_sigma_clip_doc
class MADStdBackgroundRMS(BackgroundRMSBase):

    @deprecated_positional_kwargs(since='3.0', until='4.0')
    def calc_background_rms(self, data, axis=None, masked=False):
        pass


@_insert_sigma_clip_doc
class BiweightScaleBackgroundRMS(BackgroundRMSBase):

    @deprecated_positional_kwargs(since='3.0', until='4.0')
    def __init__(self, c=9.0, M=None, sigma_clip=SIGMA_CLIP):
        super().__init__(sigma_clip=sigma_clip)
        self.c = c
        self.M = M

    def __repr__(self):
        params = ('c', 'M', 'sigma_clip')
        return make_repr(self, params)

    @deprecated_positional_kwargs(since='3.0', until='4.0')
    def calc_background_rms(self, data, axis=None, masked=False):
        pass
