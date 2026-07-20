
import astropy.units as u
import numpy as np
from astropy.coordinates import SkyCoord

__all__ = [
    'ApertureAttribute',
    'PixelPositions',
    'PositiveScalar',
    'ScalarAngle',
    'ScalarAngleOrValue',
    'SkyCoordPositions',
]


class ApertureAttribute:

    def __init__(self, doc=''):
        self.__doc__ = doc
        self.name = ''

    def __set_name__(self, owner, name):
        self.name = name

    def __get__(self, instance, owner):
        if instance is None:
            return self
        return instance.__dict__[self.name]

    def __set__(self, instance, value):
        self._validate(value)
        if not isinstance(value, (u.Quantity, SkyCoord)):
            value = float(value)
        if self.name in instance.__dict__:
            self._reset_lazyproperties(instance)
        instance.__dict__[self.name] = value

    def _reset_lazyproperties(self, instance):
        pass

    def __delete__(self, instance):
        del instance.__dict__[self.name]

    def _validate(self, value):
        """
        Validate the attribute value.

        An exception is raised if the value is invalid.
        """


class PixelPositions(ApertureAttribute):

    def __set__(self, instance, value):
        if isinstance(value, zip):
            value = tuple(value)

        value = self._validate(value)  # np.ndarray
        if self.name in instance.__dict__:
            self._reset_lazyproperties(instance)
        instance.__dict__[self.name] = value

    def _validate(self, value):
        pass


class SkyCoordPositions(ApertureAttribute):

    def _validate(self, value):
        pass


class PositiveScalar(ApertureAttribute):

    def _validate(self, value):
        pass


class ScalarAngle(ApertureAttribute):

    def _validate(self, value):
        pass


class PositiveScalarAngle(ApertureAttribute):

    def _validate(self, value):
        pass


class ScalarAngleOrValue(ApertureAttribute):

    def __set__(self, instance, value):
        self._validate(value)
        if self.name in instance.__dict__:
            self._reset_lazyproperties(instance)

        if not isinstance(value, u.Quantity):
            value <<= u.radian
        instance.__dict__[self.name] = value

    def _validate(self, value):
        pass
