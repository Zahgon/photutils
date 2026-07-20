
import warnings
from copy import deepcopy

import numpy as np
from astropy.nddata import NDData
from astropy.table import QTable, vstack

from photutils.psf._components import (_make_model_image_docstring,
                                       _make_residual_image_docstring,
                                       _ModelImageMaker)
from photutils.psf.flags import decode_psf_flags
from photutils.psf.photometry import PSFPhotometry
from photutils.psf.utils import _create_call_docstring
from photutils.utils._deprecation import (deprecated_positional_kwargs,
                                          deprecated_renamed_argument)
from photutils.utils._repr import make_repr
from photutils.utils.exceptions import NoDetectionsWarning

__all__ = ['IterativePSFPhotometry']


class IterativePSFPhotometry:

    @deprecated_renamed_argument('localbkg_estimator',
                                 'local_bkg_estimator', '3.0',
                                 until='4.0')
    def __init__(self, psf_model, fit_shape, finder, *, grouper=None,
                 fitter=None, fitter_maxiters=100, xy_bounds=None,
                 maxiters=3, mode='new', aperture_radius=None,
                 local_bkg_estimator=None, group_warning_threshold=25,
                 sub_shape=None, progress_bar=False):

        if finder is None:
            msg = 'finder cannot be None for IterativePSFPhotometry'
            raise ValueError(msg)

        if aperture_radius is None:
            msg = 'aperture_radius cannot be None for IterativePSFPhotometry'
            raise ValueError(msg)

        threshold = group_warning_threshold
        self._psfphot = PSFPhotometry(psf_model, fit_shape, finder=finder,
                                      grouper=grouper, fitter=fitter,
                                      fitter_maxiters=fitter_maxiters,
                                      xy_bounds=xy_bounds,
                                      aperture_radius=aperture_radius,
                                      local_bkg_estimator=local_bkg_estimator,
                                      group_warning_threshold=threshold,
                                      progress_bar=progress_bar)

        self.maxiters = self._validate_maxiters(maxiters)

        if mode not in ['new', 'all']:
            msg = "mode must be 'new' or 'all'"
            raise ValueError(msg)
        if mode == 'all' and grouper is None:
            msg = "grouper must be input for the 'all' mode"
            raise ValueError(msg)
        self.mode = mode

        self.sub_shape = sub_shape

        self._reset_results()

    def _reset_results(self):
        """
        Reset these attributes for each __call__.
        """
        self.fit_results = []
        self.results = None

    def __repr__(self):
        params = ('psf_model', 'fit_shape', 'finder', 'grouper', 'fitter',
                  'fitter_maxiters', 'xy_bounds', 'maxiters', 'mode',
                  'local_bkg_estimator', 'aperture_radius', 'sub_shape',
                  'progress_bar')
        overrides = {
            'psf_model': self._psfphot.psf_model,
            'fit_shape': self._psfphot.fit_shape,
            'finder': self._psfphot.finder,
            'grouper': self._psfphot.grouper,
            'fitter': self._psfphot.fitter,
            'fitter_maxiters': self._psfphot.fitter_maxiters,
            'xy_bounds': self._psfphot.xy_bounds,
            'local_bkg_estimator': self._psfphot.local_bkg_estimator,
            'aperture_radius': self._psfphot.aperture_radius,
            'progress_bar': self._psfphot.progress_bar,
        }
        return make_repr(self, params, overrides=overrides)

    @staticmethod
    def _validate_maxiters(maxiters):
        if (not np.isscalar(maxiters) or maxiters <= 0
                or ~np.isfinite(maxiters)):
            msg = 'maxiters must be a strictly-positive scalar'
            raise ValueError(msg)
        if maxiters != int(maxiters):
            msg = 'maxiters must be an integer'
            raise ValueError(msg)
        return maxiters

    @staticmethod
    def _emit_warnings(recorded_warnings):
        pass

    @staticmethod
    def _move_column(table, colname, colname_after):
        pass

    def _measure_init_fluxes(self, data, mask, sources):
        pass

    def _prepare_next_iteration_sources(self, residual_data, mask, new_sources,
                                        orig_sources):
        pass

    @_create_call_docstring(iterative=True)
    def __call__(self, data, *, mask=None, error=None, init_params=None):
        if isinstance(data, NDData):
            data_, mask, error = PSFPhotometry._coerce_nddata(data)
            return self.__call__(data_, mask=mask, error=error,
                                 init_params=init_params)

        self._reset_results()

        with warnings.catch_warnings(record=True) as rwarn0:
            phot_tbl = self._psfphot(data, mask=mask, error=error,
                                     init_params=init_params)
            self.fit_results.append(deepcopy(self._psfphot))

        if phot_tbl is None:
            self._emit_warnings(rwarn0)
            return None

        residual_data = data
        with warnings.catch_warnings(record=True) as rwarn1:
            phot_tbl['iter_detected'] = 1
            if self.mode == 'all':
                iter_detected = np.ones(len(phot_tbl), dtype=int)

            iter_num = 2
            while iter_num <= self.maxiters and phot_tbl is not None:
                residual_data = self._psfphot.make_residual_image(
                    residual_data, psf_shape=self.sub_shape)

                with warnings.catch_warnings():
                    warnings.simplefilter('ignore', NoDetectionsWarning)

                    new_sources = self._psfphot.finder(residual_data,
                                                       mask=mask)
                    if new_sources is None:  # no new sources detected
                        break

                finder_results = new_sources.copy()

                data_processor = self._psfphot._data_processor
                new_sources = data_processor._convert_finder_to_init(
                    new_sources)

                if self.mode == 'all':
                    init_params = self._prepare_next_iteration_sources(
                        residual_data, mask, new_sources,
                        self._psfphot.results_to_init_params())

                    residual_data = data

                    current_iter = (np.ones(len(new_sources), dtype=int)
                                    * iter_num)
                    iter_detected = np.concatenate((iter_detected,
                                                    current_iter))
                elif self.mode == 'new':
                    init_params = new_sources

                new_tbl = self._psfphot(residual_data, mask=mask, error=error,
                                        init_params=init_params)
                self._psfphot.finder_results = finder_results
                self.fit_results.append(deepcopy(self._psfphot))

                if self.mode == 'all':
                    new_tbl['iter_detected'] = iter_detected
                    phot_tbl = new_tbl

                elif self.mode == 'new':
                    new_tbl['id'] += np.max(phot_tbl['id'])
                    new_tbl['group_id'] += np.max(phot_tbl['group_id'])
                    new_tbl['iter_detected'] = iter_num
                    new_tbl.meta = {}  # prevent merge conflicts on date
                    phot_tbl = vstack([phot_tbl, new_tbl])

                iter_num += 1

            phot_tbl = self._move_column(phot_tbl, 'iter_detected',
                                         'group_size')

        phot_tbl.meta['psf_class'] = self.__class__.__name__
        phot_tbl.meta['maxiters'] = self.maxiters
        phot_tbl.meta['mode'] = self.mode
        phot_tbl.meta['sub_shape'] = self.sub_shape

        recorded_warnings = rwarn0 + rwarn1
        self._emit_warnings(recorded_warnings)

        self.results = phot_tbl

        return phot_tbl

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
        return self._psfphot._results_to_model_params(
            self.results, self._psfphot._param_mapper,
            remove_invalid=remove_invalid, reset_ids=reset_ids)

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
        if not self.fit_results:
            msg = ('No results available. Please run the '
                   'IterativePSFPhotometry instance first.')
            raise ValueError(msg)

        model_params, local_bkg = self._get_model_image_params()
        maker = _ModelImageMaker(self._psfphot.psf_model, model_params,
                                 local_bkg=local_bkg,
                                 progress_bar=self._psfphot.progress_bar)
        return maker.make_model_image(shape, psf_shape=psf_shape,
                                      include_local_bkg=include_local_bkg)

    @deprecated_renamed_argument('include_localbkg', 'include_local_bkg',
                                 '3.0', until='4.0')
    @_make_residual_image_docstring
    def make_residual_image(self, data, *, psf_shape=None,
                            include_local_bkg=False):
        pass
