from typing import Callable


def integrate_1(f: Callable[[float], float]) -> float:
    """
    :param f: function to be integrated
    :return: value of the integral of function f wrt. x over closed interval [0, 1]
    """
