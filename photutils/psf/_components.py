
import contextlib
import warnings
import weakref
from copy import deepcopy

import astropy
import astropy.units as u
import numpy as np
from astropy.modeling import Fittable2DModel, Parameter
from astropy.modeling.fitting import TRFLSQFitter
from astropy.nddata import NDData, NoOverlapError, overlap_slices
from astropy.table import QTable, Table, hstack, join
from astropy.utils import minversion
from astropy.utils.exceptions import AstropyUserWarning

from photutils.aperture import CircularAperture
from photutils.datasets import make_model_image as _make_model_image
from photutils.utils._deprecation import DeprecatedColumnQTable
from photutils.utils._misc import _get_meta

from .flags import PSF_FLAGS

__all__ = ['PSFDataProcessor', 'PSFFitter', 'PSFResultsAssembler']


def _apply_bounds_to_param(model, param_name, param_value, bound_value):
    """
    Apply bounds to a specific model parameter.

    This is a general helper function for applying symmetric bounds
    around a parameter value.

    Parameters
    ----------
    model : `~astropy.modeling.Model`
        The model containing the parameter.

    param_name : str
        Name of the parameter to apply bounds to.

    param_value : float
        Current value of the parameter.

    bound_value : float
        The bound offset (parameter will be bounded to
        [param_value - bound_value, param_value + bound_value]).
    """
    if bound_value is not None:
        param_obj = getattr(model, param_name)
        param_obj.bounds = (param_value - bound_value,
                            param_value + bound_value)


def _create_flat_model_class(n_sources, psf_model):
    """
    Create a new flat model class for the given number of sources.

    This function creates a dynamically-generated flat model class that
    avoids CompoundModel by creating a custom model class where all
    parameters are top-level (e.g., x_0, y_0, flux_0, x_1, y_1, flux_1,
    etc.) rather than nested. This eliminates the parameter tree traversal
    and nested structure overhead of CompoundModel.

    The dynamic flat model class directly evaluates each PSF and sums them,
    providing much better performance for large groups.

    Parameters
    ----------
    n_sources : int
        Number of sources in the group.

    psf_model : `~astropy.modeling.Model`
        Base PSF model to be used for all sources.

    Returns
    -------
    model_class : type
        Dynamically created flat model class for the specified number
        of sources.
    """
    base_param_names = list(psf_model.param_names)
    base_psf_model = psf_model.copy()

    class_attrs = {}

    for i in range(n_sources):
        for base_param in base_param_names:
            flat_param_name = f'{base_param}_{i}'
            base_param_obj = getattr(base_psf_model, base_param)
            default_value = base_param_obj.value
            is_fixed = base_param_obj.fixed
            param = Parameter(default=default_value, fixed=is_fixed)
            class_attrs[flat_param_name] = param

    def model_init(self, **kwargs):
        pass

    def evaluate(self, x, y, *params):
        """
        Evaluate the flat PSF group model.

        This efficiently evaluates each PSF with its parameters and sums
        the results, avoiding CompoundModel overhead.
        """
        result = np.zeros_like(x, dtype=float)

        for i in range(self.n_sources):
            source_params = []
            for j, _ in enumerate(self.base_param_names):
                param_idx = i * len(self.base_param_names) + j
                source_params.append(params[param_idx])

            result += self.base_psf_model.evaluate(x, y, *source_params)

        return result

    class_attrs['__init__'] = model_init
    class_attrs['evaluate'] = evaluate
    class_attrs['__doc__'] = (f'Cached flat PSF model for '
                              f'{n_sources} sources.')

    class_name = f'FlatPSFGroupModel_{n_sources}'
    return type(class_name, (Fittable2DModel,), class_attrs)


def _instantiate_flat_model(model_class, sources, psf_model, param_mapper, *,
                            xy_bounds=None):
    """
    Create an instance of a flat model class with source-specific
    parameter values and bounds.

    Parameters
    ----------
    model_class : type
        Flat model class created by `create_flat_model_class`.

    sources : `~astropy.table.Table` or list of `~astropy.table.Row`
        List of source rows from the sources table for the group.

    psf_model : `~astropy.modeling.Model`
        Base PSF model used for parameter defaults.

    param_mapper : object
        Parameter mapper for handling column name mappings and model
        parameters. Must have `alias_to_model_param` and `init_colnames`
        attributes.

    xy_bounds : tuple of float or None, optional
        Bounds for x and y position parameters as (x_bound, y_bound).
        If provided, fitting positions will be constrained to within
        these bounds of the initial values.

    Returns
    -------
    model : `~astropy.modeling.Model`
        Instance of the flat model with parameter values set from sources
        and bounds applied.
    """
    alias_map = param_mapper.alias_to_model_param
    init_colnames = param_mapper.init_colnames

    model = model_class()

    for i, source in enumerate(sources):
        for base_param in psf_model.param_names:
            flat_param_name = f'{base_param}_{i}'

            init_value = None
            for alias, col_name in init_colnames.items():
                if alias_map[alias] == base_param:
                    init_value = source[col_name]
                    if isinstance(init_value, u.Quantity):
                        init_value = init_value.value
                    break

            if init_value is None:
                init_value = getattr(psf_model, base_param).value

            setattr(model, flat_param_name, init_value)

            if (xy_bounds is not None and base_param == alias_map.get('x')
                    and xy_bounds[0] is not None):
                _apply_bounds_to_param(model, flat_param_name, init_value,
                                       xy_bounds[0])
            elif (xy_bounds is not None and base_param == alias_map.get('y')
                  and xy_bounds[1] is not None):
                _apply_bounds_to_param(model, flat_param_name, init_value,
                                       xy_bounds[1])

    return model


_FLAT_MODEL_CACHE = weakref.WeakValueDictionary()


def _get_flat_model(sources, psf_model, param_mapper, *, xy_bounds=None):
    """
    Get or create a flat model for a group of sources.

    This function caches the dynamically-generated flat model classes
    to avoid recreating them for groups with the same characteristics,
    improving performance when processing many groups.

    The caching uses weak references to prevent memory leaks and
    includes the PSF model type in the cache key to avoid collisions
    between different model types with similar parameter names.

    Parameters
    ----------
    sources : `~astropy.table.Table` or list of `~astropy.table.Row`
        List of source rows from the sources table for the group.

    psf_model : `~astropy.modeling.Model`
        Base PSF model to be used for all sources.

    param_mapper : object
        Parameter mapper for handling column name mappings and model
        parameters.

    xy_bounds : tuple of float or None, optional
        Bounds for x and y position parameters as (x_bound, y_bound).

    Returns
    -------
    model : `~astropy.modeling.Model`
        Flat model instance for the group of sources with parameters
        set from the sources and bounds applied.
    """
    n_sources = len(sources)

    model_class = psf_model.__class__
    model_type = f'{model_class.__module__}.{model_class.__name__}'
    cache_key = (n_sources, model_type)

    if cache_key not in _FLAT_MODEL_CACHE:
        model_class = _create_flat_model_class(n_sources, psf_model)
        _FLAT_MODEL_CACHE[cache_key] = model_class

    model_class = _FLAT_MODEL_CACHE[cache_key]

    return _instantiate_flat_model(model_class, sources, psf_model,
                                   param_mapper, xy_bounds=xy_bounds)


class PSFDataProcessor:

    def __init__(self, param_mapper, fit_shape, *, finder=None,
                 aperture_radius=None, local_bkg_estimator=None):
        self.param_mapper = param_mapper
        self.fit_shape = fit_shape
        self.finder = finder
        self.aperture_radius = aperture_radius
        self.local_bkg_estimator = local_bkg_estimator
        self.data_unit = None
        self.finder_results = None

        self._cached_offsets = None
        self._cache_key = None

    def validate_array(self, array, name, *, data_shape=None):
        pass

    def normalize_init_units(self, init_params, colname):
        pass

    def validate_init_params(self, init_params):
        pass

    def get_aper_fluxes(self, data, mask, init_params):
        pass

    def find_sources_if_needed(self, data, mask, init_params):
        pass

    def _convert_finder_to_init(self, sources):
        pass

    def estimate_flux_and_bkg_if_needed(self, data, mask, init_params):
        pass

    def get_fit_offsets(self):
        pass

    def should_skip_source(self, row, data_shape):
        pass

    def get_source_cutout_data(self, row, data, mask, y_offsets, x_offsets):
        pass


class PSFFitter:

    def __init__(self, psf_model, param_mapper, *, fitter=None,
                 fitter_maxiters=100, xy_bounds=None,
                 group_warning_threshold=25):
        self.psf_model = psf_model
        self.param_mapper = param_mapper
        self.fitter = fitter if fitter is not None else TRFLSQFitter()
        self.fitter_maxiters = fitter_maxiters
        self.xy_bounds = xy_bounds
        self.group_warning_threshold = group_warning_threshold

    def make_psf_model(self, sources):
        """
        Create a single PSF model or a flat model for a group of
        sources.

        This method avoids CompoundModel by creating a custom model
        class where all parameters are top-level (e.g., x_0, y_0,
        flux_0, x_1, y_1, flux_1, etc.) rather than nested. This
        eliminates the parameter tree traversal and nested structure
        overhead of CompoundModel.

        The dynamic flat model class directly evaluates each PSF and
        sums them, providing much better performance for large groups.

        Flat model classes are cached based on the number of sources and
        PSF model characteristics to improve performance for repeated
        group sizes.

        Parameters
        ----------
        sources : list of `~astropy.table.Row`
            List of source rows from the sources table for the group.

        Returns
        -------
        model : `~astropy.modeling.Model`
            PSF model for the group of sources, either a single PSF
            model or a flat model for multiple sources.
        """
        n_sources = len(sources)
        if n_sources == 1:
            source = sources[0]
            model = self.psf_model.copy()
            alias_map = self.param_mapper.alias_to_model_param
            init_colnames = self.param_mapper.init_colnames

            for alias, col_name in init_colnames.items():
                model_param = alias_map[alias]
                value = source[col_name]
                if isinstance(value, u.Quantity):
                    value = value.value
                setattr(model, model_param, value)

            if 'id' in source.colnames:
                model.name = source['id']

            self._apply_xy_bounds(model, alias_map)
            return model

        return _get_flat_model(sources, self.psf_model, self.param_mapper,
                               xy_bounds=self.xy_bounds)

    def _apply_bounds_to_param(self, model, param_name, param_value,
                               bound_value):
        """
        Apply bounds to a specific model parameter.

        This method is now a wrapper around the standalone helper
        function.
        """
        _apply_bounds_to_param(model, param_name, param_value, bound_value)

    def _apply_xy_bounds(self, model, alias_map):
        """
        Apply xy_bounds to a model.
        """
        if self.xy_bounds is not None:
            if self.xy_bounds[0] is not None:
                x_param_name = alias_map['x']
                x_param = getattr(model, x_param_name)
                self._apply_bounds_to_param(model, x_param_name,
                                            x_param.value, self.xy_bounds[0])
            if self.xy_bounds[1] is not None:
                y_param_name = alias_map['y']
                y_param = getattr(model, y_param_name)
                self._apply_bounds_to_param(model, y_param_name,
                                            y_param.value, self.xy_bounds[1])

    def run_fitter(self, psf_model, xi, yi, cutout, error):
        pass

    def extract_source_covariances(self, group_cov, num_sources, nfitparam):
        pass

    def split_flat_model(self, flat_model, n_sources):
        pass


class PSFResultsAssembler:

    def __init__(self, param_mapper, fit_shape, *, xy_bounds=None):
        self.param_mapper = param_mapper
        self.fit_shape = fit_shape
        self.xy_bounds = xy_bounds

    def get_fit_error_indices(self, fit_info):
        pass

    def param_errors_to_table(self, fit_param_errs, data_unit):
        pass

    def create_fit_results(self, fit_model_all_params, fit_param_errs,
                           valid_mask, data_unit):
        pass

    def calc_fit_metrics(self, results_tbl, sum_abs_residuals, cen_residuals,
                         reduced_chi2):
        pass

    def define_flags(self, results_tbl, shape, fit_error_indices, fit_info,
                     fitted_models_table, valid_mask, invalid_reasons,
                     init_params):
        pass

    def assemble_results_table(self, init_params, fit_params, data_shape,
                               state, calc_fit_metrics_func, define_flags_func,
                               class_name, metadata_attrs):
        pass


def _make_model_image_docstring(func):
    func.__doc__ = """
        Create a 2D image from the fit PSF models and optional local
        background.

        Parameters
        ----------
        shape : 2 tuple of int
            The shape of the output array.

        psf_shape : 2-tuple of int, optional
            The shape of the region around the center of the fit model
            to render in the output image. If ``psf_shape`` is a scalar
            integer, then a square shape of size ``psf_shape`` will be
            used. If `None`, then the bounding box of the model will be
            used. This keyword must be specified if the model does not
            have a ``bounding_box`` attribute.

        include_local_bkg : bool, optional
            Whether to include the local background in the rendered
            output image. Note that the local background level is
            included around each source over the region defined by
            ``psf_shape``. Thus, regions where the ``psf_shape`` of
            sources overlap will have the local background added
            multiple times. Non-finite local background values (NaN or
            inf) are treated as zero and not included in the output
            image.

        Returns
        -------
        array : 2D `~numpy.ndarray`
            The rendered image from the fit PSF models. This image will
            not have any units.
        """
    return func


def _make_residual_image_docstring(func):
    func.__doc__ = """
        Create a 2D residual image from the fit PSF models and local
        background.

        Parameters
        ----------
        data : 2D `~numpy.ndarray`
            The 2D array on which photometry was performed. This should
            be the same array input when calling the PSF-photometry
            class.

        psf_shape : 2-tuple of int, optional
            The shape of the region around the center of the fit model
            to subtract. If ``psf_shape`` is a scalar integer, then
            a square shape of size ``psf_shape`` will be used. If
            `None`, then the bounding box of the model will be used.
            This keyword must be specified if the model does not have a
            ``bounding_box`` attribute.

        include_local_bkg : bool, optional
            Whether to include the local background in the subtracted
            model. Note that the local background level is subtracted
            around each source over the region defined by ``psf_shape``.
            Thus, regions where the ``psf_shape`` of sources overlap
            will have the local background subtracted multiple times.
            Non-finite local background values (NaN or inf) are not
            subtracted from the residual image.

        Returns
        -------
        array : 2D `~numpy.ndarray`
            The residual image of the ``data`` minus the fit PSF models
            minus the optional``local_bkg``.
        """
    return func


class _ModelImageMaker:

    def __init__(self, psf_model, model_params, *, local_bkg=None,
                 progress_bar=False):
        self.psf_model = psf_model
        self.model_params = model_params
        self.local_bkg = local_bkg
        self.progress_bar = progress_bar

    @_make_model_image_docstring
    def make_model_image(self, shape, *, psf_shape=None,
                         include_local_bkg=False):
        psf_model = self.psf_model
        model_params = self.model_params
        local_bkgs = self.local_bkg
        progress_bar = self.progress_bar

        if include_local_bkg:
            model_params = model_params.copy()
            local_bkgs_clean = local_bkgs.copy()
            nonfinite_mask = ~np.isfinite(local_bkgs_clean)
            if np.any(nonfinite_mask):
                local_bkgs_clean[nonfinite_mask] = 0
            model_params['local_bkg'] = local_bkgs_clean

        try:
            x_name = psf_model.x_name
            y_name = psf_model.y_name
        except AttributeError:
            x_name = 'x_0'
            y_name = 'y_0'

        return _make_model_image(shape, psf_model, model_params,
                                 model_shape=psf_shape,
                                 x_name=x_name, y_name=y_name,
                                 progress_bar=progress_bar)

    @_make_residual_image_docstring
    def make_residual_image(self, data, *, psf_shape=None,
                            include_local_bkg=False):
        pass
