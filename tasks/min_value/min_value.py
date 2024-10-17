from typing import Callable


def find_minimum(f: Callable[[float], float]) -> tuple[float, float]:
    """
    Find the local minimum of a function using 0 as the starting point.
    :param f: the function whose minimum we're looking for
    :return: a tuple containing the minimum value and argument that maps to
             such value in this order
    """
