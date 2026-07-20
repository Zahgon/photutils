
from photutils.segmentation.deblend import deblend_sources
from photutils.segmentation.detect import detect_sources
from photutils.utils._deprecation import (deprecated_getattr,
                                          deprecated_positional_kwargs,
                                          deprecated_renamed_argument)
from photutils.utils._parameters import as_pair
from photutils.utils._repr import make_repr

__all__ = ['SourceFinder']

_FINDER_DEPRECATED_ATTRIBUTES = {
    'npixels': 'n_pixels',
    'nlevels': 'n_levels',
    'nproc': 'n_processes',
}


class SourceFinder:

    @deprecated_renamed_argument('npixels', 'n_pixels', '3.0', until='4.0')
    @deprecated_renamed_argument('nlevels', 'n_levels', '3.0', until='4.0')
    @deprecated_renamed_argument('nproc', 'n_processes', '3.0', until='4.0')
    def __init__(self, n_pixels, *, connectivity=8, deblend=True, n_levels=32,
                 contrast=0.001, mode='exponential', relabel=True,
                 n_processes=1, progress_bar=True):
        self.n_pixels = as_pair('n_pixels', n_pixels, check_odd=False)
        self.deblend = deblend
        self.connectivity = connectivity
        self.n_levels = n_levels
        self.contrast = contrast
        self.mode = mode
        self.relabel = relabel
        self.n_processes = n_processes
        self.progress_bar = progress_bar

    def __repr__(self):
        params = ('n_pixels', 'deblend', 'connectivity', 'n_levels',
                  'contrast', 'mode', 'relabel', 'n_processes',
                  'progress_bar')
        return make_repr(self, params)

    def __getattr__(self, name):
        return deprecated_getattr(self, name,
                                  _FINDER_DEPRECATED_ATTRIBUTES,
                                  since='3.0', until='4.0')

    @deprecated_positional_kwargs(since='3.0', until='4.0')
    def __call__(self, data, threshold, mask=None):
        """
        Detect sources, including deblending, in an image using
        segmentation.

        Parameters
        ----------
        data : 2D `~numpy.ndarray`
            The 2D array from which to detect sources. Typically, this
            array should be an image that has been convolved with a
            smoothing kernel.

        threshold : 2D `~numpy.ndarray` or float
            The data value or pixel-wise data values (as an array) to be
            used as the per-pixel detection threshold. If ``data`` is
            a `~astropy.units.Quantity` array, then ``threshold`` must
            have the same units as ``data``. A 2D ``threshold`` array must
            have the same shape as ``data``.

        mask : 2D bool `~numpy.ndarray`, optional
            A boolean mask with the same shape as ``data``, where a
            `True` value indicates the corresponding element of ``data``
            is masked. Masked pixels will not be included in any source.

        Returns
        -------
        segment_image : `~photutils.segmentation.SegmentationImage` or `None`
            A 2D segmentation image, with the same shape as the input data,
            where sources are marked by different positive integer values. A
            value of zero is reserved for the background. If no sources are
            found then `None` is returned.
        """
        segment_img = detect_sources(data, threshold, self.n_pixels[0],
                                     mask=mask, connectivity=self.connectivity)
        if segment_img is None:
            return None

        if self.deblend:
            segment_img = deblend_sources(data, segment_img, self.n_pixels[1],
                                          n_levels=self.n_levels,
                                          contrast=self.contrast,
                                          mode=self.mode,
                                          connectivity=self.connectivity,
                                          relabel=self.relabel,
                                          n_processes=self.n_processes,
                                          progress_bar=self.progress_bar)

        return segment_img
