from typing import Callable
from scipy.optimize import minimize


def find_minimum(f: Callable[[float], float]) -> tuple[float, float]:
    """
    Find the local minimum of a function using 0 as the starting point.

    :param f: the function whose minimum we're looking for
    :return: a tuple containing the minimum value and argument that maps to
             such value in this order
    """
    # Use scipy.optimize.minimize to find the local minimum
    result = minimize(f, x0=0)  # x0=0 is the initial guess

    # Extract the minimum value and the corresponding x value
    x_min = result.x[0]  # Location of the minimum
    y_min = result.fun   # Value of the function at the minimum

    return y_min, x_min
