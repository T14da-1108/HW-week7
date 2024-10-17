from typing import Any

import numpy as np
import numpy.typing as npt
import scipy


def solve_linear_system(coeffs: npt.NDArray[Any], consts: npt.NDArray[np.float64])\
        -> tuple[npt.NDArray[np.float64] | None, bool]:
    """
    Solve a system of linear equations $Ax = b$ by calculating inverse matrix, e.g. $A^{-1}$.
    :param coeffs: square matrix of coefficients, A in the aforementioned equation
    :param consts: vector of constants, b in the aforementioned equation
    :return: if there exists inverse matrix and, consequently, a solution, return `(<SOLUTION>, True)`
             where `<SOLUTION>` is a `numpy` array. Otherwise, return `(None, False)`.
    """

    # Hints:
    # * matrix multiplication in numpy is NOT *, rather something else
    # * numerical algorithms are not so precise (this is also related to scipy):
    #   https://github.com/numpy/numpy/issues/10471
