
import warnings

import numpy as np
from astropy.modeling import Model
from astropy.table import QTable
from astropy.units import Quantity
from astropy.utils.exceptions import AstropyUserWarning
from scipy import interpolate

from photutils.centroids import centroid_com
from photutils.psf.functional_models import CircularGaussianPRF
from photutils.utils import CutoutImage
from photutils.utils._parameters import as_pair

__all__ = ['fit_2dgaussian', 'fit_fwhm']


def _make_mask(image, mask):
    """
    Create a mask for the input image.

    Non-finite values (e.g., NaN or inf) in the ``image`` array are
    automatically masked. If a mask is provided, then the non-finite
    values are combined with the provided mask.

    Parameters
    ----------
    image : 2D `~numpy.ndarray`
        The input image.

    mask : 2D bool `~numpy.array` or None
        A boolean mask with the same shape as ``data``, where a `True`
        value indicates the corresponding element of ``data`` is masked.

    Returns
    -------
    mask : 2D bool `~numpy.ndarray` or `None`
        The mask for the input image. A `True` value indicates the
        corresponding element of ``image`` is masked.
    """
    def warn_nonfinite():
        msg = ('Input data contains unmasked non-finite values '
               '(NaN or inf), which were automatically ignored.')
        warnings.warn(msg, AstropyUserWarning)

    finite_mask = ~np.isfinite(image)

    if mask is not None:
        finite_mask |= mask
        if np.any(finite_mask & ~mask):
            warn_nonfinite()
    else:
        mask = finite_mask
        if np.any(finite_mask):
            warn_nonfinite()
        else:
            mask = None

    return mask


def fit_2dgaussian(data, *, xypos=None, fwhm=None, fix_fwhm=True,
                   fit_shape=None, mask=None, error=None):
    """
    Fit a 2D Gaussian model to one or more sources in an image.

    This convenience function uses a
    `~photutils.psf.CircularGaussianPRF` model to fit the sources using
    the `~photutils.psf.PSFPhotometry` class.

    Non-finite values (e.g., NaN or inf) in the ``data`` array are
    automatically masked.

    Parameters
    ----------
    data : 2D array
        The 2D array of the image. The input array must be background
        subtracted.

    xypos : array-like, optional
        The initial (x, y) pixel coordinates of the sources as a list
        of tuples or a 2D array. If `None`, then one source will be fit
        with an initial position using the center-of-mass centroid of
        the ``data`` array.

    fwhm : float, optional
        The initial guess for the FWHM of the Gaussian PSF model. If
        `None`, then the initial guess is half the mean of the x and y
        sizes of the ``fit_shape`` values.

    fix_fwhm : bool, optional
        Whether to fix the FWHM of the Gaussian PSF model during the
        fitting process.

    fit_shape : int or tuple of two ints, optional
        The shape of the fitting region. If a scalar, then it is assumed
        to be a square. If `None`, then the shape of the input ``data``
        will be used.

    mask : array-like (bool), optional
        A boolean mask with the same shape as the input ``data``, where
        a `True` value indicates the corresponding element of ``data``
        is masked.

    error : 2D array, optional
        The pixel-wise Gaussian 1-sigma errors of the input
        ``data``. ``error`` is assumed to include *all* sources
        of error, including the Poisson error of the sources (see
        `~photutils.utils.calc_total_error`). ``error`` must have the
        same shape as the input ``data``. If a `~astropy.units.Quantity`
        array, then ``data`` must also be a `~astropy.units.Quantity`
        array with the same units.

    Returns
    -------
    result : `~photutils.psf.PSFPhotometry`
        The PSF-fitting photometry results.

    See Also
    --------
    fit_fwhm : Fit the FWHM of one or more sources in an image.

    Notes
    -----
    The source(s) are fit with a `~photutils.psf.CircularGaussianPRF`
    model using the `~photutils.psf.PSFPhotometry` class. The initial
    guess for the flux is the sum of the pixel values within the fitting
    region. If ``fwhm`` is `None`, then the initial guess for the FWHM
    is half the mean of the x and y sizes of the ``fit_shape`` values.

    Examples
    --------
    Fit a 2D Gaussian model to an image containing only one source
    (e.g., a cutout image):

    >>> import numpy as np
    >>> from photutils.psf import CircularGaussianPRF, fit_2dgaussian
    >>> yy, xx = np.mgrid[:51, :51]
    >>> model = CircularGaussianPRF(x_0=22.17, y_0=28.87, fwhm=3.123, flux=9.7)
    >>> data = model(xx, yy)
    >>> fit = fit_2dgaussian(data, fix_fwhm=False, fit_shape=7)
    >>> phot_tbl = fit.results  # doctest: +FLOAT_CMP
    >>> cols = ['x_fit', 'y_fit', 'fwhm_fit', 'flux_fit']
    >>> for col in cols:
    ...     phot_tbl[col].info.format = '.4f'  # optional format
    >>> print(phot_tbl[['id'] + cols])
     id  x_fit   y_fit  fwhm_fit flux_fit
    --- ------- ------- -------- --------
      1 22.1700 28.8700   3.1230   9.7000

    Fit a 2D Gaussian model to multiple sources in an image:

    >>> import numpy as np
    >>> from photutils.detection import DAOStarFinder
    >>> from photutils.psf import (CircularGaussianPRF, fit_2dgaussian,
    ...                            make_psf_model_image)
    >>> model = CircularGaussianPRF()
    >>> data, sources = make_psf_model_image((100, 100), model, 5,
    ...                                      min_separation=25,
    ...                                      model_shape=(15, 15),
    ...                                      flux=(100, 200), fwhm=[3, 8])
    >>> finder = DAOStarFinder(0.1, 5)
    >>> finder_tbl = finder(data)
    >>> xypos = zip(sources['x_0'], sources['y_0'])
    >>> psfphot = fit_2dgaussian(data, xypos=xypos, fit_shape=7,
    ...                          fix_fwhm=False)
    >>> phot_tbl = psfphot.results
    >>> len(phot_tbl)
    5

    Here we show only a few columns of the photometry table:

    >>> cols = ['x_fit', 'y_fit', 'fwhm_fit', 'flux_fit']
    >>> for col in cols:
    ...     phot_tbl[col].info.format = '.4f'  # optional format
    >>> print(phot_tbl[['id'] + cols])
     id  x_fit   y_fit  fwhm_fit flux_fit
    --- ------- ------- -------- --------
      1 61.7787 74.6905   5.6947 147.9988
      2 30.2017 27.5858   5.2138 123.2373
      3 10.5237 82.3776   7.6551 180.1881
      4  8.4214 12.0369   3.2026 192.3530
      5 76.9412 35.9061   6.6600 126.6130
    """
    from photutils.psf.photometry import PSFPhotometry

    mask = _make_mask(data, mask)

    if xypos is None:
        xypos = centroid_com(data, mask=mask)
    if isinstance(xypos, zip):
        xypos = np.array(list(xypos))
    xypos = np.atleast_2d(xypos)

    if fit_shape is None:
        if len(xypos) > 1:
            msg = ('fit_shape is required when fitting multiple sources. If '
                   'fit_shape is not provided, then the fit may not converge '
                   'or may give poor results. For multiple sources, you '
                   'should provide a fit_shape value that is appropriate for '
                   'the size of the sources in your image.')
            raise ValueError(msg)

        fit_shape = data.shape
        fit_shape = tuple(s - 1 if s % 2 == 0 else s for s in fit_shape)

        msg = ('fit_shape is None, so the input data array is assumed to be '
               'a cutout image containing only one source. If your input '
               'data is not a cutout image, then the fit may not converge '
               'or may give poor results. For non-cutout input data, you '
               'should provide a fit_shape value that is appropriate for '
               'the size of the sources in your image. '
               f'Using fit_shape={fit_shape}.')
        warnings.warn(msg, AstropyUserWarning)
    else:
        fit_shape = as_pair('fit_shape', fit_shape, lower_bound=(1, 1),
                            check_odd=True)

    flux_init = []
    for yxpos in xypos[:, ::-1]:
        cutout = CutoutImage(data, yxpos, tuple(fit_shape))
        cutout = cutout.data[np.isfinite(cutout.data)]
        flux_init.append(np.nansum(cutout))

    if isinstance(data, Quantity):
        flux_init <<= data.unit

    init_params = QTable()
    init_params['x'] = xypos[:, 0]
    init_params['y'] = xypos[:, 1]
    init_params['flux'] = flux_init

    if fwhm is None:
        fwhm = np.mean(fit_shape) / 2.0
    init_params['fwhm'] = fwhm

    model = CircularGaussianPRF(fwhm=fwhm)
    model.fwhm.min = 0.0
    if not fix_fwhm:
        model.fwhm.fixed = False

    phot = PSFPhotometry(model, fit_shape)
    _ = phot(data, mask=mask, error=error, init_params=init_params)

    return phot


def fit_fwhm(data, *, xypos=None, fwhm=None, fit_shape=None, mask=None,
             error=None):
    """
    Fit the FWHM of one or more sources in an image.

    This convenience function uses a
    `~photutils.psf.CircularGaussianPRF` model to fit the sources using
    the `~photutils.psf.PSFPhotometry` class.

    Non-finite values (e.g., NaN or inf) in the ``data`` array are
    automatically masked.

    Parameters
    ----------
    data : 2D array
        The 2D array of the image. The input array must be background
        subtracted.

    xypos : array-like, optional
        The initial (x, y) pixel coordinates of the sources as a list
        of tuples or a 2D array. If `None`, then one source will be fit
        with an initial position using the center-of-mass centroid of
        the ``data`` array.

    fwhm : float, optional
        The initial guess for the FWHM of the Gaussian PSF model. If
        `None`, then the initial guess is half the mean of the x and y
        sizes of the ``fit_shape`` values.

    fit_shape : int or tuple of two ints, optional
        The shape of the fitting region. If a scalar, then it is assumed
        to be a square. If `None`, then the shape of the input ``data``
        will be used.

    mask : array-like (bool), optional
        A boolean mask with the same shape as the input ``data``, where
        a `True` value indicates the corresponding element of ``data``
        is masked.

    error : 2D array, optional
        The pixel-wise Gaussian 1-sigma errors of the input
        ``data``. ``error`` is assumed to include *all* sources
        of error, including the Poisson error of the sources (see
        `~photutils.utils.calc_total_error`). ``error`` must have the
        same shape as the input ``data``. If a `~astropy.units.Quantity`
        array, then ``data`` must also be a `~astropy.units.Quantity`
        array with the same units.

    Returns
    -------
    fwhm : `~numpy.ndarray`
        The FWHM of the sources. Note that the returned FWHM values are
        always positive.

    See Also
    --------
    fit_2dgaussian : Fit a 2D Gaussian model to one or more sources in an
                     image.

    Notes
    -----
    The source(s) are fit using the :func:`fit_2dgaussian` function,
    which uses a `~photutils.psf.CircularGaussianPRF` model with the
    `~photutils.psf.PSFPhotometry` class. The initial guess for the
    flux is the sum of the pixel values within the fitting region. If
    ``fwhm`` is `None`, then the initial guess for the FWHM is half the
    mean of the x and y sizes of the ``fit_shape`` values.

    Examples
    --------
    Fit the FWHM of a single source (e.g., a cutout image):

    >>> import numpy as np
    >>> from photutils.psf import CircularGaussianPRF, fit_fwhm
    >>> yy, xx = np.mgrid[:51, :51]
    >>> model = CircularGaussianPRF(x_0=22.17, y_0=28.87, fwhm=3.123, flux=9.7)
    >>> data = model(xx, yy)
    >>> fwhm = fit_fwhm(data, fit_shape=7)
    >>> fwhm  # doctest: +FLOAT_CMP
    array([3.123])

    Fit the FWHMs of multiple sources in an image:

    >>> import numpy as np
    >>> from photutils.detection import DAOStarFinder
    >>> from photutils.psf import (CircularGaussianPRF, fit_fwhm,
    ...                            make_psf_model_image)
    >>> model = CircularGaussianPRF()
    >>> data, sources = make_psf_model_image((100, 100), model, 5,
    ...                                      min_separation=25,
    ...                                      model_shape=(15, 15),
    ...                                      flux=(100, 200), fwhm=[3, 8])
    >>> finder = DAOStarFinder(0.1, 5)
    >>> finder_tbl = finder(data)
    >>> xypos = zip(sources['x_0'], sources['y_0'])
    >>> fwhms = fit_fwhm(data, xypos=xypos, fit_shape=7)
    >>> fwhms  # doctest: +FLOAT_CMP
    array([5.69467204, 5.21376414, 7.65508658, 3.20255356, 6.66003098])
    """
    with warnings.catch_warnings(record=True) as fit_warnings:
        phot = fit_2dgaussian(data, xypos=xypos, fwhm=fwhm, fix_fwhm=False,
                              fit_shape=fit_shape, mask=mask, error=error)

    fit_shape_warning = False
    other_warnings = False
    for warning in fit_warnings:
        if 'fit_shape is None' in str(warning.message):
            warnings.warn(str(warning.message), warning.category)
            fit_shape_warning = True
        else:
            other_warnings = True

    if other_warnings and not fit_shape_warning:
        msg = ('One or more fit(s) may not have converged. Please '
               'carefully check your results. You may need to change '
               'the input "xypos" and "fit_shape" parameters.')
        warnings.warn(msg, AstropyUserWarning)

    return np.array(phot.results['fwhm_fit'])


def _interpolate_missing_data(data, mask, *, method='cubic'):
    """
    Interpolate missing data as identified by the ``mask`` keyword.

    Parameters
    ----------
    data : 2D `~numpy.ndarray`
        An array containing the 2D image.

    mask : 2D bool `~numpy.ndarray`
        A 2D boolean mask array with the same shape as the input
        ``data``, where a `True` value indicates the corresponding
        element of ``data`` is masked. The masked data points are those
        that will be interpolated.

    method : {'cubic', 'nearest'}, optional
        The method of used to interpolate the missing data:

        * ``'cubic'``:  Masked data are interpolated using 2D cubic
          splines. If any masked pixels cannot be interpolated using
          cubic interpolation (e.g., at the edges), they will be filled
          using nearest-neighbor interpolation as a fallback.

        * ``'nearest'``:  Masked data are interpolated using
          nearest-neighbor interpolation.

    Returns
    -------
    data_interp : 2D `~numpy.ndarray`
        The interpolated 2D image. All masked pixels are guaranteed
        to be filled if there are any valid (unmasked) pixels. If all
        pixels are masked, the returned array will contain NaN values.
    """
    data_interp = np.copy(data)

    if len(data_interp.shape) != 2:
        msg = 'data must be a 2D array'
        raise ValueError(msg)

    if mask.shape != data.shape:
        msg = 'mask and data must have the same shape'
        raise ValueError(msg)

    if not np.any(mask):
        return data_interp

    if np.all(mask):
        data_interp[:] = np.nan
        return data_interp

    y, x = np.indices(data_interp.shape)
    xy = np.dstack((x[~mask].ravel(), y[~mask].ravel()))[0]
    z = data_interp[~mask].ravel()

    if method == 'nearest':
        interpol = interpolate.NearestNDInterpolator(xy, z)
    elif method == 'cubic':
        interpol = interpolate.CloughTocher2DInterpolator(xy, z)
    else:
        msg = 'Unsupported interpolation method'
        raise ValueError(msg)

    xy_missing = np.dstack((x[mask].ravel(), y[mask].ravel()))[0]
    data_interp[mask] = interpol(xy_missing)

    if method == 'cubic':
        remaining_mask = ~np.isfinite(data_interp)
        if np.any(remaining_mask):
            xy_valid = np.dstack((x[~remaining_mask].ravel(),
                                  y[~remaining_mask].ravel()))[0]
            z_valid = data_interp[~remaining_mask].ravel()
            interpol_nn = interpolate.NearestNDInterpolator(xy_valid, z_valid)
            xy_remaining = np.dstack((x[remaining_mask].ravel(),
                                      y[remaining_mask].ravel()))[0]
            data_interp[remaining_mask] = interpol_nn(xy_remaining)

    return data_interp


def _validate_psf_model(psf_model):
    """
    Validate the PSF model.

    The PSF model must be a subclass of
    `astropy.modeling.Fittable2DModel`. It must also be two-dimensional
    and have a single output.

    Parameters
    ----------
    psf_model : `astropy.modeling.Fittable2DModel`
        The PSF model to validate.

    Returns
    -------
    psf_model : `astropy.modeling.Model`
        The validated PSF model.

    Raises
    ------
    TypeError
        If the PSF model is not an Astropy Model subclass.

    ValueError
        If the PSF model is not two-dimensional with n_inputs=2 and
        n_outputs=1.
    """
    if not isinstance(psf_model, Model):
        msg = 'psf_model must be an Astropy Model subclass'
        raise TypeError(msg)

    if psf_model.n_inputs != 2 or psf_model.n_outputs != 1:
        msg = ('psf_model must be two-dimensional with '
               'n_inputs=2 and n_outputs=1')
        raise ValueError(msg)

    return psf_model


def _get_psf_model_main_params(psf_model):
    """
    Get the names of the main PSF model parameters corresponding to x,
    y, and flux.

    The PSF model must have parameters called 'x_0', 'y_0', and
    'flux' or it must have 'x_name', 'y_name', and 'flux_name'
    attributes (i.e., output from `make_psf_model`). Otherwise, a
    `ValueError` is raised.

    The PSF model must be a subclass of `astropy.modeling.Model`. It
    must also be two-dimensional and have a single output.

    Parameters
    ----------
    psf_model : `astropy.modeling.Model`
        The PSF model to validate.

    Returns
    -------
    model_params : tuple
        A tuple of the PSF model parameter names. These are always
        returned in the order of (x, y, flux).
    """
    psf_model = _validate_psf_model(psf_model)

    params1 = ('x_0', 'y_0', 'flux')
    params2 = ('x_name', 'y_name', 'flux_name')
    if all(name in psf_model.param_names for name in params1):
        model_params = params1
    elif all(params := [getattr(psf_model, name, None) for name in params2]):
        model_params = tuple(params)
    else:
        msg = 'Invalid PSF model - could not find PSF parameter names'
        raise ValueError(msg)

    return model_params


def _create_call_docstring(*, iterative=False):
    """
    Decorator factory to create the __call__ method docstring for PSF
    photometry methods.

    This decorator factory creates a decorator that provides a base
    docstring for PSF photometry methods and customizes it based on the
    class type (PSFPhotometry vs IterativePSFPhotometry).

    Parameters
    ----------
    iterative : bool, optional
        If True, customize the docstring for IterativePSFPhotometry.
        If False, customize for PSFPhotometry.

    Returns
    -------
    decorator : callable
        A method decorator that updates the method's docstring.
    """
    def decorator(func):
        pass

    return decorator
