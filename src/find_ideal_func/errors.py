"""
Custom Exceptions for the find_ideal_func module.
This module defines custom exceptions used throughout the find_ideal_func package.
"""


class NoMatchingError(Exception):
    """Raise when no testpoint is possible to map to any of the four ideal functions."""


class DatavalidatorError(Exception):
    """Raise when the data validation fails."""


class IdealFunctionError(Exception):
    """Base class for exceptions related to ideal functions."""


class IdealFunctionNotFoundError(IdealFunctionError):
    """Raise when the ideal function is not found in the database."""
