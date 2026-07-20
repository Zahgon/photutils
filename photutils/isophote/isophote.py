
import astropy.units as u
import numpy as np

from photutils.isophote.harmonics import (first_and_second_harmonic_function,
                                          fit_first_and_second_harmonics,
                                          fit_upper_harmonic)
from photutils.utils._deprecation import (create_empty_deprecated_qtable,
                                          deprecated_getattr,
                                          deprecated_positional_kwargs)
from photutils.utils._misc import _get_meta

__all__ = ['Isophote', 'IsophoteList']

_DEPRECATED_ATTRIBUTES = {
    'grad_error': 'gradient_err',
    'grad_r_error': 'gradient_rel_err',
    'niter': 'n_iter',
    'ndata': 'n_data',
    'nflag': 'n_flag',
}


class Isophote:

    def __init__(self, sample, n_iter, valid, stop_code):
        self.sample = sample
        self.n_iter = n_iter
        self.valid = valid
        self.stop_code = stop_code

        if sample.geometry.sma > 0:
            self.intens = sample.mean
            self.rms = np.std(sample.values[2])
            self.int_err = self.rms / np.sqrt(sample.actual_points)
            self.pix_stddev = self.rms * np.sqrt(sample.sector_area)
            self.grad = sample.gradient
            self.gradient_err = sample.gradient_err

            self.gradient_rel_err = sample.gradient_rel_err
            self.sarea = sample.sector_area
            self.n_data = sample.actual_points
            self.n_flag = sample.total_points - sample.actual_points

            (self.tflux_e, self.tflux_c, self.npix_e,
             self.npix_c) = self._compute_fluxes()

            self._compute_errors()

            (self.a3, self.b3, self.a3_err,
             self.b3_err) = self._compute_deviations(sample, 3)
            (self.a4, self.b4, self.a4_err,
             self.b4_err) = self._compute_deviations(sample, 4)

    def __getattr__(self, name):
        return deprecated_getattr(self, name, _DEPRECATED_ATTRIBUTES,
                                  since='3.0', until='4.0')

    @staticmethod
    def _raise_sma_error(err):
        pass

    def __lt__(self, other):
        try:
            return self.sma < other.sma
        except AttributeError as err:
            self._raise_sma_error(err)

    def __gt__(self, other):
        try:
            return self.sma > other.sma
        except AttributeError as err:
            self._raise_sma_error(err)

    def __le__(self, other):
        try:
            return self.sma <= other.sma
        except AttributeError as err:
            self._raise_sma_error(err)

    def __ge__(self, other):
        try:
            return self.sma >= other.sma
        except AttributeError as err:
            self._raise_sma_error(err)

    def __eq__(self, other):
        try:
            return self.sma == other.sma
        except AttributeError as err:
            self._raise_sma_error(err)

    def __ne__(self, other):
        try:
            return self.sma != other.sma
        except AttributeError as err:
            self._raise_sma_error(err)

    def __str__(self):
        return str(self.to_table())

    @property
    def sma(self):
        pass

    @property
    def eps(self):
        pass

    @property
    def pa(self):
        pass

    @property
    def x0(self):
        pass

    @property
    def y0(self):
        pass

    def _compute_fluxes(self):
        """
        Compute integrated flux inside ellipse, as well as inside a
        circle defined with the same semimajor axis.

        Pixels in a square section enclosing circle are scanned; the
        distance of each pixel to the isophote center is compared both
        with the semimajor axis length and with the length of the
        ellipse radius vector, and integrals are updated if the pixel
        distance is smaller.
        """
        sma = self.sample.geometry.sma
        x0 = self.sample.geometry.x0
        y0 = self.sample.geometry.y0
        xsize = self.sample.image.shape[1]
        ysize = self.sample.image.shape[0]

        imin = max(0, int(x0 - sma - 0.5) - 1)
        jmin = max(0, int(y0 - sma - 0.5) - 1)
        imax = min(xsize, int(x0 + sma + 0.5) + 1)
        jmax = min(ysize, int(y0 + sma + 0.5) + 1)

        if (jmax - jmin > 1) and (imax - imin) > 1:
            y, x = np.mgrid[jmin:jmax, imin:imax]
            radius, angle = self.sample.geometry.to_polar(x, y)
            radius_e = self.sample.geometry.radius(angle)

            midx = (radius <= sma)
            values = self.sample.image[y[midx], x[midx]]
            tflux_c = np.ma.sum(values)
            npix_c = np.ma.count(values)

            midx2 = (radius <= radius_e)
            values = self.sample.image[y[midx2], x[midx2]]
            tflux_e = np.ma.sum(values)
            npix_e = np.ma.count(values)
        else:
            tflux_e = 0.0
            tflux_c = 0.0
            npix_e = 0
            npix_c = 0

        return tflux_e, tflux_c, npix_e, npix_c

    def _compute_deviations(self, sample, n):
        """
        Compute deviations from a perfect ellipse, based on the
        amplitudes and errors for harmonic "n".

        Note that we first subtract the first and second harmonics from
        the raw data.
        """
        try:
            up_coeffs, up_inv_hessian = fit_upper_harmonic(sample.values[0],
                                                           sample.values[2],
                                                           n)

            a = up_coeffs[1] / self.sma / abs(sample.gradient)
            b = up_coeffs[2] / self.sma / abs(sample.gradient)

            def errfunc(x, phi, order, intensities):
                return (x[0] + x[1] * np.sin(order * phi)
                        + x[2] * np.cos(order * phi) - intensities)

            up_var_residual = np.std(errfunc(up_coeffs, self.sample.values[0],
                                             n, self.sample.values[2]),
                                     ddof=len(up_coeffs))**2
            up_covariance = up_inv_hessian * up_var_residual

            ce = np.sqrt(np.diag(up_covariance))

            gre = (self.gradient_rel_err
                   if self.gradient_rel_err is not None else 0.8)

            a_err = abs(a) * np.sqrt((ce[1] / up_coeffs[1])**2 + gre**2)
            b_err = abs(b) * np.sqrt((ce[2] / up_coeffs[2])**2 + gre**2)

        except Exception:  # we want to catch everything
            a = b = a_err = b_err = None

        return a, b, a_err, b_err

    def _compute_errors(self):
        """
        Compute parameter errors based on the diagonal of the covariance
        matrix of the four harmonic coefficients for harmonics n=1 and
        n=2.0.
        """
        try:
            coeffs, covariance = fit_first_and_second_harmonics(
                self.sample.values[0], self.sample.values[2])
            model = first_and_second_harmonic_function(self.sample.values[0],
                                                       coeffs)
            var_residual = np.std(self.sample.values[2] - model,
                                  ddof=len(coeffs)) ** 2
            errors = np.sqrt(np.diagonal(covariance * var_residual))

            eps = self.sample.geometry.eps
            pa = self.sample.geometry.pa

            ea = abs(errors[2] / self.grad)
            eb = abs(errors[1] * (1.0 - eps) / self.grad)
            self.x0_err = np.sqrt((ea * np.cos(pa))**2 + (eb * np.sin(pa))**2)
            self.y0_err = np.sqrt((ea * np.sin(pa))**2 + (eb * np.cos(pa))**2)
            self.ellip_err = (abs(2.0 * errors[4] * (1.0 - eps) / self.sma
                                  / self.grad))
            if abs(eps) > np.finfo(float).resolution:
                self.pa_err = (abs(2.0 * errors[3] * (1.0 - eps) / self.sma
                                   / self.grad / (1.0 - (1.0 - eps)**2)))
            else:
                self.pa_err = 0.0
        except Exception:  # we want to catch everything
            self.x0_err = self.y0_err = self.pa_err = self.ellip_err = 0.0

    def fix_geometry(self, isophote):
        pass

    def sampled_coordinates(self):
        pass

    def to_table(self):
        pass


class CentralPixel(Isophote):

    def __init__(self, sample):
        super().__init__(sample, 0, valid=True, stop_code=0)

        self.intens = sample.mean

        self.rms = None
        self.int_err = 0.0
        self.pix_stddev = None
        self.grad = 0.0
        self.gradient_err = None
        self.gradient_rel_err = None
        self.sarea = None
        self.n_data = sample.actual_points
        self.n_flag = sample.total_points - sample.actual_points

        self.tflux_e = self.tflux_c = self.npix_e = self.npix_c = None

        self.a3 = self.b3 = 0.0
        self.a4 = self.b4 = 0.0
        self.a3_err = self.b3_err = 0.0
        self.a4_err = self.b4_err = 0.0

        self.ellip_err = 0.0
        self.pa_err = 0.0
        self.x0_err = 0.0
        self.y0_err = 0.0

    def __eq__(self, other):
        try:
            return self.sma == other.sma
        except AttributeError as err:
            self._raise_sma_error(err)

    @property
    def eps(self):
        pass

    @property
    def pa(self):
        pass


class IsophoteList:

    def __init__(self, iso_list):
        self._list = iso_list

    def __getattr__(self, name):
        return deprecated_getattr(self, name, _DEPRECATED_ATTRIBUTES,
                                  since='3.0', until='4.0')

    def __len__(self):
        return len(self._list)

    def __delitem__(self, index):
        self._list.__delitem__(index)

    def __setitem__(self, index, value):
        self._list.__setitem__(index, value)

    def __getitem__(self, index):
        if isinstance(index, slice):
            return IsophoteList(self._list[index])
        return self._list.__getitem__(index)

    def __iter__(self):
        return self._list.__iter__()

    def sort(self):
        """
        Sort the list of isophotes by semimajor axis length.
        """
        self._list.sort()

    def insert(self, index, value):
        """
        Insert an isophote at a given index.

        Parameters
        ----------
        index : int
            The index where to insert the isophote.
        value : `~photutils.isophote.Isophote`
            The isophote to be inserted.
        """
        self._list.insert(index, value)

    def append(self, value):
        """
        Append an isophote to the list.

        Parameters
        ----------
        value : `~photutils.isophote.Isophote`
            The isophote to be appended.
        """
        self.insert(len(self) + 1, value)

    def extend(self, value):
        """
        Extend the list with the isophotes from another
        `~photutils.isophote.IsophoteList` instance.

        Parameters
        ----------
        value : `~photutils.isophote.IsophoteList`
            The isophotes to be appended.
        """
        self._list.extend(value._list)

    def __iadd__(self, value):
        self.extend(value)
        return self

    def __add__(self, value):
        temp = self._list[:]  # shallow copy
        temp.extend(value._list)
        return IsophoteList(temp)

    def get_closest(self, sma):
        pass

    def _collect_as_array(self, attr_name):
        pass

    def _collect_as_list(self, attr_name):
        pass

    @property
    def sample(self):
        pass

    @property
    def sma(self):
        pass

    @property
    def intens(self):
        pass

    @property
    def int_err(self):
        pass

    @property
    def eps(self):
        pass

    @property
    def ellip_err(self):
        pass

    @property
    def pa(self):
        pass

    @property
    def pa_err(self):
        pass

    @property
    def x0(self):
        pass

    @property
    def x0_err(self):
        pass

    @property
    def y0(self):
        pass

    @property
    def y0_err(self):
        pass

    @property
    def rms(self):
        pass

    @property
    def pix_stddev(self):
        pass

    @property
    def grad(self):
        pass

    @property
    def gradient_err(self):
        pass

    @property
    def gradient_rel_err(self):
        pass

    @property
    def sarea(self):
        pass

    @property
    def n_data(self):
        pass

    @property
    def n_flag(self):
        pass

    @property
    def n_iter(self):
        pass

    @property
    def valid(self):
        pass

    @property
    def stop_code(self):
        pass

    @property
    def tflux_e(self):
        pass

    @property
    def tflux_c(self):
        pass

    @property
    def npix_e(self):
        pass

    @property
    def npix_c(self):
        pass

    @property
    def a3(self):
        pass

    @property
    def b3(self):
        pass

    @property
    def a4(self):
        pass

    @property
    def b4(self):
        pass

    @property
    def a3_err(self):
        pass

    @property
    def b3_err(self):
        pass

    @property
    def a4_err(self):
        pass

    @property
    def b4_err(self):
        pass

    @deprecated_positional_kwargs(since='3.0', until='4.0')
    def to_table(self, columns='main'):
        pass

    def get_names(self):
        pass


def _get_properties(isophote_list):
    pass


def _isophote_list_to_table(isophote_list, *, columns='main'):
    pass
