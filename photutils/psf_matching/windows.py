
import warnings

import numpy as np
from astropy.utils.exceptions import AstropyUserWarning

__all__ = [
    'CosineBellWindow',
    'HanningWindow',
    'SplitCosineBellWindow',
    'TopHatWindow',
    'TukeyWindow',
]


def _distance_grid(shape):
    pass


class SplitCosineBellWindow:

    def __init__(self, alpha, beta):
        if not (0.0 <= alpha <= 1.0):
            msg = ('alpha must be between 0.0 and 1.0, inclusive. '
                   f'Got: {alpha}')
            raise ValueError(msg)
        if not (0.0 <= beta <= 1.0):
            msg = ('beta must be between 0.0 and 1.0, inclusive. '
                   f'Got: {beta}')
            raise ValueError(msg)

        if alpha + beta > 1.0:
            msg = ('alpha + beta > 1.0; the taper region will be '
                   'clipped to the array boundary.')
            warnings.warn(msg, AstropyUserWarning)

        self.alpha = float(alpha)
        self.beta = float(beta)

    def __repr__(self):
        return (f'{self.__class__.__name__}('
                f'alpha={self.alpha!r}, beta={self.beta!r})')

    def __str__(self):
        return self.__repr__()

    def __call__(self, shape):
        """
        Generate the window function for the given shape.

        Parameters
        ----------
        shape : tuple of int
            The size of the output array along each axis.

        Returns
        -------
        result : 2D `~numpy.ndarray`
            The window function as a 2D array.
        """
        dist = _distance_grid(shape)

        max_r = (min(shape) - 1.0) / 2.0
        r_inner = self.beta * max_r
        taper_width = self.alpha * max_r
        r_outer = r_inner + taper_width

        if taper_width > 0:
            r = dist - r_inner
            result = 0.5 * (1.0 + np.cos(np.pi * r / taper_width))
        else:
            result = np.ones(shape, dtype=float)

        result[dist < r_inner] = 1.0
        result[dist > r_outer] = 0.0

        return result


class HanningWindow(SplitCosineBellWindow):

    def __init__(self):
        super().__init__(alpha=1.0, beta=0.0)

    def __repr__(self):
        return f'{self.__class__.__name__}()'

    def __str__(self):
        return self.__repr__()


class TukeyWindow(SplitCosineBellWindow):

    def __init__(self, alpha):
        super().__init__(alpha=alpha, beta=1.0 - alpha)

    def __repr__(self):
        return (f'{self.__class__.__name__}'
                f'(alpha={self.alpha!r})')

    def __str__(self):
        return self.__repr__()


class CosineBellWindow(SplitCosineBellWindow):

    def __init__(self, alpha):
        super().__init__(alpha=alpha, beta=0.0)

    def __repr__(self):
        return (f'{self.__class__.__name__}'
                f'(alpha={self.alpha!r})')

    def __str__(self):
        return self.__repr__()


class TopHatWindow(SplitCosineBellWindow):

    def __init__(self, beta):
        super().__init__(alpha=0.0, beta=beta)

    def __repr__(self):
        return (f'{self.__class__.__name__}'
                f'(beta={self.beta!r})')

    def __str__(self):
        return self.__repr__()
