
from functools import partial

import numpy as np
from astropy.units import Quantity

from photutils.utils._optional_deps import HAS_BOTTLENECK

_STAT_NAMES = (
    'nansum', 'nanmin', 'nanmax', 'nanmean', 'nanmedian', 'nanstd', 'nanvar',
)

if HAS_BOTTLENECK:
    import bottleneck as bn

    def _move_tuple_axes_last(array, axis):
        pass

    def _apply_bottleneck(function, array, axis=None, **kwargs):
        pass

    bn_funcs = {
        name: partial(_apply_bottleneck, getattr(bn, name))
        for name in _STAT_NAMES
    }
    np_funcs = {name: getattr(np, name) for name in _STAT_NAMES}

    class _DtypeDispatch:

        def __init__(self, func_name):
            self.func_name = func_name

        def __repr__(self):
            return f'_DtypeDispatch({self.func_name!r})'

        def __call__(self, *args, **kwargs):
            dt = args[0].dtype
            if dt.kind == 'f' and dt.itemsize == 8:
                return bn_funcs[self.func_name](*args, **kwargs)
            return np_funcs[self.func_name](*args, **kwargs)

    (nansum, nanmin, nanmax, nanmean, nanmedian, nanstd, nanvar) = (
        _DtypeDispatch(name) for name in _STAT_NAMES
    )

else:
    (nansum, nanmin, nanmax, nanmean, nanmedian, nanstd, nanvar) = (
        getattr(np, name) for name in _STAT_NAMES
    )
