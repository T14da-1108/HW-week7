from typing import Any
import numpy as np
import numpy.typing as npt


def solve_linear_system(coeffs: npt.NDArray[Any], consts: npt.NDArray[np.float64]) \
        -> tuple[npt.NDArray[np.float64] | None, bool]:
    """
    Solve a system of linear equations Ax = b using the inverse matrix method.

    :param coeffs: square matrix of coefficients, A in the equation Ax = b
    :param consts: vector of constants, b in the equation Ax = b
    :return: a tuple:
             - the solution as a numpy array if the system can be solved,
               otherwise None
             - a boolean indicating whether the inverse matrix exists
    """
    try:
        # Compute the determinant to check if the matrix is invertible
        det = np.linalg.det(coeffs)
        if np.isclose(det, 0):
            # If determinant is zero (or close to zero), the matrix is not invertible
            return None, False

        # Calculate the inverse of the matrix
        inverse_matrix = np.linalg.inv(coeffs)

        # Compute the solution x = A^(-1) * b
        solution = inverse_matrix @ consts  # Matrix multiplication

        return solution, True
    except np.linalg.LinAlgError:
        # Handle cases where the matrix cannot be inverted
        return None, False
