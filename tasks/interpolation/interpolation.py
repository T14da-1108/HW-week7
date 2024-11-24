from typing import Callable
import numpy as np
import numpy.typing as npt
from scipy.interpolate import lagrange


def interpolate_polynomial(x_values: npt.NDArray[np.float64], y_values: npt.NDArray[np.float64])\
        -> np.poly1d:
    """
    Return interpolating function using Lagrange polynomial.
    :param x_values: initial x-values of a function
    :param y_values: initial y-values, that is f(x), of a function
    :return: interpolating polynomial as a numpy.poly1d object
    """
    # Use scipy's lagrange function to compute the Lagrange polynomial
    lagrange_poly = lagrange(x_values, y_values)

    # Convert the result to a numpy.poly1d object for better compatibility and usability
    return np.poly1d(lagrange_poly)
