
import contextlib
import inspect
import warnings
from dataclasses import dataclass, field

import astropy.units as u
import numpy as np
from astropy.modeling.fitting import TRFLSQFitter
from astropy.nddata import NDData, StdDevUncertainty
from astropy.table import QTable
from astropy.utils.decorators import deprecated
from astropy.utils.exceptions import AstropyUserWarning

from photutils.background import LocalBackground
from photutils.psf._components import (PSFDataProcessor, PSFFitter,
                                       PSFResultsAssembler,
                                       _make_model_image_docstring,
                                       _make_residual_image_docstring,
                                       _ModelImageMaker)
from photutils.psf.flags import decode_psf_flags
from photutils.psf.utils import (_create_call_docstring,
                                 _get_psf_model_main_params, _make_mask,
                                 _validate_psf_model)
from photutils.utils._deprecation import (deprecated_positional_kwargs,
                                          deprecated_renamed_argument)
from photutils.utils._parameters import as_pair
from photutils.utils._progress_bars import add_progress_bar
from photutils.utils._quantity_helpers import process_quantities
from photutils.utils._repr import make_repr

__all__ = ['PSFPhotometry']


@dataclass
class _PSFParameterMapper:

    psf_model: object
    alias_to_model_param: dict = field(init=False, repr=False)

    VALID_INIT_COLNAMES = {  # noqa: RUF012
        'x': (
            'x_init', 'xinit', 'x', 'x_0', 'x0', 'xcentroid',
            'x_centroid', 'x_peak', 'xcen', 'x_cen', 'xpos', 'x_pos',
            'x_fit', 'xfit',
        ),
        'y': (
            'y_init', 'yinit', 'y', 'y_0', 'y0', 'ycentroid',
            'y_centroid', 'y_peak', 'ycen', 'y_cen', 'ypos', 'y_pos',
            'y_fit', 'yfit',
        ),
        'flux': (
            'flux_init', 'fluxinit', 'flux', 'flux_0', 'flux0',
            'flux_fit', 'fluxfit', 'source_sum', 'segment_flux',
            'kron_flux',
        ),
    }
    MAIN_ALIASES = ('x', 'y', 'flux')

    def __post_init__(self):
        self.alias_to_model_param = self._get_model_params_map()

    def _get_model_params_map(self):
        pass

    @property
    def fitted_param_names(self):
        pass

    def get_init_colname(self, alias):
        pass

    def get_fit_colname(self, alias):
        pass

    def get_err_colname(self, alias):
        pass

    @property
    def init_colnames(self):
        pass

    @property
    def fit_colnames(self):
        pass

    @property
    def err_colnames(self):
        pass

    @property
    def model_param_to_alias(self):
        pass

    def find_column(self, table, param_alias):
        pass

    def rename_table_columns(self, table):
        pass


class PSFPhotometry:

    _DEFAULT_PARAM_VALUE = np.nan

    @deprecated_renamed_argument('localbkg_estimator',
                                 'local_bkg_estimator', '3.0',
                                 until='4.0')
    def __init__(self, psf_model, fit_shape, *, finder=None, grouper=None,
                 fitter=None, fitter_maxiters=100, xy_bounds=None,
                 aperture_radius=None, local_bkg_estimator=None,
                 group_warning_threshold=25, progress_bar=False):

        self.psf_model = _validate_psf_model(psf_model)
        self._param_mapper = _PSFParameterMapper(self.psf_model)

        self.fit_shape = as_pair('fit_shape', fit_shape, lower_bound=(1, 1),
                                 check_odd=True)
        self.finder = self._validate_callable(finder, 'finder')
        self.grouper = self._validate_callable(grouper, 'grouper')
        if fitter is None:
            fitter = TRFLSQFitter()
        self.fitter = self._validate_callable(fitter, 'fitter')
        self.fitter_maxiters = self._validate_maxiters(fitter_maxiters)
        self.xy_bounds = self._validate_bounds(xy_bounds)
        self.aperture_radius = self._validate_radius(aperture_radius)
        self.local_bkg_estimator = self._validate_localbkg(
            local_bkg_estimator, 'local_bkg_estimator')
        self.group_warning_threshold = group_warning_threshold
        self.progress_bar = progress_bar

        self._data_processor = PSFDataProcessor(
            self._param_mapper, self.fit_shape, finder=self.finder,
            aperture_radius=self.aperture_radius,
            local_bkg_estimator=self.local_bkg_estimator,
        )

        self._psf_fitter = PSFFitter(
            self.psf_model, self._param_mapper, fitter=self.fitter,
            fitter_maxiters=self.fitter_maxiters, xy_bounds=self.xy_bounds,
            group_warning_threshold=self.group_warning_threshold,
        )

        self._results_assembler = PSFResultsAssembler(
            self._param_mapper, self.fit_shape, xy_bounds=self.xy_bounds,
        )

        self._attrs = ('psf_model', 'fit_shape', 'finder', 'grouper', 'fitter',
                       'fitter_maxiters', 'xy_bounds', 'aperture_radius',
                       'local_bkg_estimator', 'group_warning_threshold',
                       'progress_bar')

        self._reset_results()

    def _reset_results(self):
        """
        Reset internal state attributes for each __call__.
        """
        self.data_unit = None
        self.finder_results = None
        self.init_params = None
        self.results = None
        self.fit_info = []

        if hasattr(self, '_data_processor'):
            self._data_processor.data_unit = None

        self._state = {
            'valid_mask_by_id': None,
            'fit_param_errs': None,
            'fit_error_indices': None,
            'fitted_models_table': None,
            'n_pixels_fit': None,
            'group_size': None,
            'invalid_reasons': None,
            'sum_abs_residuals': None,
            'cen_residuals': None,
            'reduced_chi2': None,
        }

    def _initialize_source_state_storage(self, n_sources):
        pass

    def _init_model_param_storage(self, n_sources):
        pass

    def _cache_fitted_parameters(self, row_index, model):
        pass

    def _build_fitted_models_table(self):
        pass

    def __repr__(self):
        return make_repr(self, self._attrs)

    @staticmethod
    def _validate_type(obj, name, expected_type):
        """
        Validate that object is of expected type.

        Parameters
        ----------
        obj : object or None
            Object to validate.

        name : str
            Name of the parameter for error messages.

        expected_type : type or tuple of types
            Expected type(s) for the object.

        Returns
        -------
        obj : object or None
            The validated object.

        Raises
        ------
        error_type
            If obj is not None and not an instance of expected_type.
        """
        if obj is not None and not isinstance(obj, expected_type):
            type_name = expected_type.__name__
            msg = f'{name} must be a {type_name} instance'
            raise TypeError(msg)
        return obj

    @staticmethod
    def _validate_callable(obj, name):
        """
        Validate that the input object is callable.

        Parameters
        ----------
        obj : object or None
            Object to validate.

        name : str
            Name of the parameter for error messages.

        Returns
        -------
        obj : object or None
            The validated callable object.

        Raises
        ------
        TypeError
            If obj is not None and not callable.
        """
        if obj is not None and not callable(obj):
            msg = f'{name!r} must be a callable object'
            raise TypeError(msg)
        return obj

    def _validate_bounds(self, xy_bounds):
        """
        Validate the input ``xy_bounds`` value.

        Parameters
        ----------
        xy_bounds : float, tuple of float, or None
            The maximum distance(s) in pixels that fitted sources can be
            from initial positions.

        Returns
        -------
        xy_bounds : ndarray or None
            The validated xy_bounds as a 2-element array, or None if
            input was None.

        Raises
        ------
        ValueError
            If xy_bounds has incorrect shape, dimension, or contains
            invalid values (non-positive or non-finite).
        """
        if xy_bounds is None:
            return xy_bounds

        xy_bounds = np.atleast_1d(xy_bounds)
        if len(xy_bounds) == 1:
            xy_bounds = np.array((xy_bounds[0], xy_bounds[0]))
        if len(xy_bounds) != 2:
            msg = 'xy_bounds must have 1 or 2 elements'
            raise ValueError(msg)
        if xy_bounds.ndim != 1:
            msg = 'xy_bounds must be a 1D array'
            raise ValueError(msg)
        for bound in xy_bounds:
            if bound is not None:
                if bound <= 0:
                    msg = 'xy_bounds must be strictly positive'
                    raise ValueError(msg)
                if not np.isfinite(bound):
                    msg = 'xy_bounds must be finite'
                    raise ValueError(msg)
        return xy_bounds

    @staticmethod
    def _validate_radius(radius):
        """
        Validate the input ``aperture_radius`` value.

        Parameters
        ----------
        radius : float or None
            The aperture radius value to validate.

        Returns
        -------
        radius : float or None
            The validated aperture radius.

        Raises
        ------
        ValueError
            If radius is not None and is not a strictly positive finite
            scalar.
        """
        if radius is not None and (not np.isscalar(radius)
                                   or radius <= 0 or not np.isfinite(radius)):
            msg = 'aperture_radius must be a strictly-positive scalar'
            raise ValueError(msg)
        return radius

    def _validate_localbkg(self, value, name):
        """
        Validate the input ``local_bkg_estimator`` value.

        Parameters
        ----------
        value : LocalBackground or None
            The local background estimator to validate.

        name : str
            Name of the parameter for error messages.

        Returns
        -------
        value : LocalBackground or None
            The validated local background estimator.

        Raises
        ------
        TypeError
            If value is not None and not a LocalBackground instance.
        """
        value = self._validate_type(value, 'local_bkg_estimator',
                                    LocalBackground)
        return self._validate_callable(value, name)

    def _validate_maxiters(self, maxiters):
        """
        Validate the input ``maxiters`` value.

        Parameters
        ----------
        maxiters : int or None
            Maximum number of fitter iterations to validate.

        Returns
        -------
        maxiters : int or None
            The validated maxiters value, or None if the fitter doesn't
            support this parameter.
        """
        spec = inspect.signature(self.fitter.__call__)
        if 'maxiter' not in spec.parameters:
            msg = ("'maxiters' will be ignored because it is not accepted "
                   'by the input fitter __call__ method.')
            warnings.warn(msg, AstropyUserWarning)
            maxiters = None
        return maxiters

    def _sync_data_unit(self):
        pass

    def _find_sources_if_needed(self, data, mask, init_params):
        pass

    def _group_sources(self, init_params):
        pass

    def _build_initial_parameters(self, data, mask, init_params):
        pass

    def _prepare_fit_inputs(self, data, *, mask=None, error=None,
                            init_params=None):
        pass

    def _ungroup_fit_results(self, row_indices, valid_mask, group_model,
                             group_fit_info):
        pass

    def _calculate_residual_metrics(self, row_indices, valid_mask,
                                    npixfit_full, cen_index_full, *,
                                    error=None, xi_all=None, yi_all=None):
        pass

    def _fit_source_groups(self, source_groups, data, mask, error):
        pass

    def _get_fit_error_indices(self):
        pass

    def _create_fit_results(self, fit_model_all_params):
        pass

    def _assemble_fit_results(self):
        pass

    def _fit_sources(self, data, init_params, *, error=None, mask=None):
        pass

    def _calc_fit_metrics(self, results_tbl):
        pass

    def _define_flags(self, results_tbl, shape, init_params):
        pass

    def _assemble_results_table(self, init_params, fit_params, data_shape):
        pass

    @staticmethod
    def _coerce_nddata(data):
        pass

    @_create_call_docstring(iterative=False)
    def __call__(self, data, *, mask=None, error=None, init_params=None):
        self._reset_results()

        try:
            if isinstance(data, NDData):
                data, mask, error = self._coerce_nddata(data)

            data, mask, error, init_params = self._prepare_fit_inputs(
                data, mask=mask, error=error, init_params=init_params,
            )

            if init_params is None:
                return None

            self.init_params = init_params

            fit_params = self._fit_sources(data, init_params, error=error,
                                           mask=mask)

            self.results = self._assemble_results_table(
                init_params, fit_params, data.shape)

        except Exception:
            self._reset_state()
            raise

        self._reset_state()
        return self.results

    def _reset_state(self):
        pass

    @property
    @deprecated('2.3.0', alternative='results')
    def fit_params(self):
        pass

    @staticmethod
    def _results_to_init_params(results_tbl, *, remove_invalid=True,
                                reset_ids=True):
        pass

    @staticmethod
    def _results_to_model_params(results_tbl, param_mapper, *,
                                 remove_invalid=True, reset_ids=True):
        """
        Convert PSF photometry results to PSF model parameters format.

        This method extracts fitted parameters (columns ending with
        '_fit') from the results table and renames them to match the
        PSF model's parameter names (e.g., 'x_fit' → 'x_0', 'flux_fit'
        → 'flux'). This output can be used to reconstruct fitted PSF
        models for visualization or further analysis.

        This is a static helper method to allow it to be called by
        `~photutils.psf.IterativePSFPhotometry`.

        Parameters
        ----------
        results_tbl : `~astropy.table.QTable` or None
            The table of fit results from a previous `PSFPhotometry`
            run. If None, returns None.

        param_mapper : `_PSFParameterMapper`
            The helper class that manages the mapping between aliases
            (e.g., 'x', 'flux') and PSF model parameter names (e.g.,
            'x_0', 'flux').

        remove_invalid : bool, optional
            If `True`, rows containing non-finite fitted values are
            removed. Default is `True`.

        reset_ids : bool, optional
            If `True`, the 'id' column is reset to sequential
            numbering starting from 1. If `False`, the 'id' values are
            preserved from ``results_tbl``. This option is ignored if
            ``remove_invalid`` is `False`. Default is `True`.

        Returns
        -------
        model_params_tbl : `~astropy.table.QTable` or None
            A table with columns renamed to match the PSF model's
            parameter names, suitable for model reconstruction. Returns
            None if ``results_tbl`` is None.

        Notes
        -----
        Only the 'id' column and columns with '_fit' suffix are included
        in the output. All other columns from the results table (e.g.,
        initial parameters, quality metrics, flags) are excluded.
        """
        if results_tbl is None:
            return None

        tbl = QTable()
        for col_name in results_tbl.colnames:
            if col_name == 'id' or '_fit' in col_name:
                alias = col_name.replace('_fit', '')
                model_param_name = param_mapper.alias_to_model_param.get(
                    alias, alias)
                tbl[model_param_name] = results_tbl[col_name]

        if remove_invalid:
            keep = np.all([np.isfinite(tbl[col])
                           for col in tbl.colnames], axis=0)
            tbl = tbl[keep]

            if reset_ids:
                tbl['id'] = np.arange(1, len(tbl) + 1)

        return tbl

    def results_to_init_params(self, *, remove_invalid=True, reset_ids=True):
        pass

    def results_to_model_params(self, *, remove_invalid=True, reset_ids=True):
        """
        Create a table of the fitted model parameters from the results.

        The table columns are named according to the PSF model parameter
        names. It can also be used to reconstruct the fitted PSF models
        for visualization or further analysis.

        Parameters
        ----------
        remove_invalid : bool, optional
            If `True`, rows that contain non-finite fitted values are
            removed.

        reset_ids : bool, optional
            If `True`, the 'id' column will be reset to a sequential
            numbering starting from 1. If `False`, the 'id' column will
            remain unchanged from the results table. This option is
            ignored if ``remove_invalid`` is `False`.
        """
        return self._results_to_model_params(self.results,
                                             self._param_mapper,
                                             remove_invalid=remove_invalid,
                                             reset_ids=reset_ids)

    @deprecated_positional_kwargs(since='3.0', until='4.0')
    def decode_flags(self, return_bit_values=False):
        pass

    def _get_model_image_params(self):
        model_params = self.results_to_model_params(remove_invalid=False)

        keep = np.all([np.isfinite(model_params[col])
                       for col in model_params.colnames], axis=0)
        model_params = model_params[keep]

        local_bkg = self.results['local_bkg'][keep]

        return model_params, local_bkg

    @deprecated_renamed_argument('include_localbkg', 'include_local_bkg',
                                 '3.0', until='4.0')
    @_make_model_image_docstring
    def make_model_image(self, shape, *, psf_shape=None,
                         include_local_bkg=False):
        if self.results is None:
            msg = ('No results available. Please run the PSFPhotometry '
                   'instance first.')
            raise ValueError(msg)

        model_params, local_bkg = self._get_model_image_params()
        maker = _ModelImageMaker(self.psf_model, model_params,
                                 local_bkg=local_bkg,
                                 progress_bar=self.progress_bar)
        return maker.make_model_image(shape, psf_shape=psf_shape,
                                      include_local_bkg=include_local_bkg)

    @deprecated_renamed_argument('include_localbkg', 'include_local_bkg',
                                 '3.0', until='4.0')
    @_make_residual_image_docstring
    def make_residual_image(self, data, *, psf_shape=None,
                            include_local_bkg=False):
        pass
