from typing import Callable
from scipy.integrate import quad, IntegrationWarning
import warnings


def integrate_1(f: Callable[[float], float]) -> float:
    """
    Integrates the given function `f` over the interval [0, 1].

    :param f: function to be integrated
    :return: value of the integral of function f wrt. x over closed interval [0, 1]
    """
    # Ensure warnings are not suppressed, so they can propagate to pytest
    with warnings.catch_warnings():
        warnings.simplefilter("always", IntegrationWarning)
        result, _ = quad(f, 0, 1)  # Perform numerical integration over [0, 1]
    return result
