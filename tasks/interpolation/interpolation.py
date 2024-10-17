from typing import Callable

import numpy as np
import numpy.typing as npt
import scipy


def interpolate_polynomial(x_values: npt.NDArray[np.float64], y_values: npt.NDArray[np.float64])\
        -> np.poly1d:
    """
    Return interpolating function using Lagrange polynomial.
    :param x_values: initial x-values of a function
    :param y_values: initial y-values, that is f(x), of a function
    :return: interpolating polynomial
    """

    # keep in mind that Lagrange interpolation has a lot of
    # drawbacks when applied to real world data, for instance see
    # Runge's phenomenon
